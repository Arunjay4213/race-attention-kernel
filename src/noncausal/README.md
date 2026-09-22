# Non-causal RACE Attention (CUDA)

RACE Attention (arXiv 2510.04008, Algorithm 1, global normalization) as bf16 CUDA kernels: a three-kernel forward and a five-launch backward, with an fp64-capable PyTorch reference, an autograd wrapper, and tests.

Status: the kernels compile cleanly with nvcc 12.8 and 13 for sm_80, sm_89 and sm_90 (no warnings, no spills, at most 64 registers per thread), and the torch extension builds, but they have **never run on a GPU**.
The references, the backward derivation check, the CPU emulations of the kernels' data flow, and the index-coverage tests all pass on CPU.
`tests/test_forward_cuda.py`, then `tests/test_backward_cuda.py`, are the first things to run on a GPU.

## Math

For each table l = 1..L with fixed planes W_l ∈ R^{P×d} (rows ~ N(0, I), never trained) and a trainable scalar β:

- uₜ = tanh(w_{l,t} · x), the same W for queries and keys, no 1/√d and no normalization of x (as in the authors' code).
- φ_l(x)[r] = softmax_r(β ∑ₜ c_{r,t} uₜ) over the R = 2^P corners c_r ∈ {±1}^P, computed exactly as ∏ₜ σ(±2βuₜ).
- A_l[r] = ∑ⱼ φ_l(kⱼ)[r] and B_l[r,:] = ∑ⱼ φ_l(kⱼ)[r] vⱼ.
- Oᵢ = (∑_l ∑_r φ_l(qᵢ)[r] B_l[r,:]) / (∑_l ∑_r φ_l(qᵢ)[r] A_l[r]); the paper's 1/L factors cancel.

The authors' repo divides the logits by `scale`, so β = 1/scale there.
Corner r has +1 on plane t iff bit (P−1−t) of r is set, the `itertools.product` order.
If Den underflows to exactly 0 (only possible for very large β), the output row is 0.

## Layout

| path | what |
|---|---|
| `reference.py` | PyTorch reference: `race_forward_reference`, `corner_probs_softmax`, `corner_probs_bernoulli`, `bucket_sums`, and the decomposed backward `race_backward_reference` |
| `BACKWARD_DERIVATION.md` | the backward's vector-Jacobian product, derived step by step, and what is saved |
| `ops.py` | `race_attention(q, k, v, W, beta)` and `RaceAttentionFunction`, the autograd wrapper |
| `kernels/race_fwd.h` | plain C++ forward launcher API (no torch), tensor and workspace layouts |
| `kernels/race_fwd.cu` | the forward's query and tree-reduce kernels and the forward launchers |
| `kernels/race_bwd.h` | plain C++ backward launcher API and backward workspace layout |
| `kernels/race_bwd.cu` | the backward's query-gradient, key-gradient and β-reduce kernels and launchers |
| `kernels/race_internal.cuh` | shared launch constants, dispatch, and the bucket build kernel (forward A/B and backward dA/dB) |
| `kernels/race_common.cuh` | device helpers: bf16 vector I/O, warp all-reduce, sigmoid pair, corner probability, corner reduce-scatter |
| `kernels/torch_binding.cpp` | `forward`, `forward_debug(...) -> (o, A, B)`, `forward_train(...) -> (o, bucket_totals)`, `backward(...) -> (dq, dk, dv, dbeta)` |
| `build.py` | JIT build via `torch.utils.cpp_extension.load`, `load_extension()` |
| `tests/test_reference.py` | CPU tests of the forward reference |
| `tests/test_backward_reference.py` | CPU fp64 check of the backward derivation against autograd and finite differences |
| `tests/test_kernel_emulation.py` | CPU checks of the forward kernels' index math and an fp32 emulation of their summation order |
| `tests/test_backward_emulation.py` | the same for the backward: reduce-scatter, layouts, fp32 data flow |
| `tests/test_forward_cuda.py` | forward GPU tests (skipped without CUDA) |
| `tests/test_backward_cuda.py` | backward and autograd GPU tests (skipped without CUDA) |
| `tests/numerics.py` | input generation and the derived tolerances |
| `bench/bench_forward.py` | forward timing against the HBM roofline |
| `bench/bench_backward.py` | backward and forward + backward timing against the HBM roofline |

Inputs: q, k, v are [B, H, N, d] bf16 contiguous CUDA tensors with identical shapes, W is [L, P, d] fp32, β is a one-element tensor (CPU or CUDA).
Limits: d ∈ {64, 128}, P ∈ {1..5}, L ∈ {1..4}, B·H ≤ 65535, N ≤ 2³¹ − 2049.
The binding's functions are not differentiable; `ops.race_attention` is.

## Kernels

1. `bucket_build_kernel`: grid (tiles·L, B·H), 256 threads, 2048 keys per CTA.
   Warps stage 16 tokens at a time (φ and v in double-buffered shared memory), then each thread folds the stage into the (r, c) slots of B it owns in registers.
2. `tree_reduce_pass_kernel`: sums the per-tile partials in place, 8 tiles per launch with a fixed pairwise tree, ⌈log₈(tiles)⌉ launches.
   The order depends only on N, so results are bitwise reproducible.
3. `query_kernel`: grid (⌈N/1024⌉, B·H); loads the final A, B and all planes into dynamic shared memory (up to 80 KB, opted in above 48 KB), one warp per query token, one reciprocal per token, bf16 store.

## Workspace

One fp32 buffer of shape [tiles, B·H, L, R·(d+1)], tiles = ⌈N/2048⌉.
In each slice, B[r][c] is at r·d + c and A[r] is at R·d + r.
After the reduce, tile 0 holds the totals over all N keys, which is what `forward_debug` returns.
Size example: N = 2²⁰, B·H = 4, L = 4, P = 4, d = 128 gives 512 · 4 · 4 · 2064 floats ≈ 68 MB.

## Backward

`ops.race_attention(q, k, v, W, beta)` is differentiable in q, k, v and β (W is fixed; passing a W that requires grad raises).
The derivation is in `BACKWARD_DERIVATION.md`.

Saved for backward: q, k, v, W, β and the reduced A, B as `bucket_totals` [B, H, L, R·(d+1)] fp32, L·R·(d+1) floats per head (33 KB at P = 4, L = 4, d = 128).
O and the per-token Den are not saved: Den is recomputed from A, and O is never needed because g · Num comes out of the same dot products B_l[r] · g that the gradient with respect to φ(q) needs.
Double backward raises (`once_differentiable`).

The backward is five launches on one stream (`race_backward` in `kernels/race_bwd.h`):

1. `query_grad_kernel`: grid (⌈N/1024⌉, B·H), one warp per query token, A, B and the planes in dynamic shared memory (up to 85 KB, opted in above 48 KB).
   Three passes over the tables per token: hash, then y = B · dO with Den and g · Num, then dφ, the chain rule through φ and tanh, and dq.
   Writes dq, a per-token pair (1/Den, dDen), and one β partial per CTA.
2. `bucket_build_kernel<D, P, true>`: the forward's bucket build over (q, dO) with the per-token pair as weights, giving per-tile dB and dA.
3. `tree_reduce_pass_kernel`: the forward's deterministic tree over the dA/dB tiles.
4. `key_grad_kernel`: grid (⌈N/1024⌉, B·H), one warp per key token, dA, dB and the planes in shared memory; writes dk, dv and one β partial per CTA.
5. `beta_grad_reduce_kernel`: one CTA sums all β partials in a fixed order.

Backward workspace (one fp32 buffer): the dA/dB tiles, the same size as the forward workspace, then 8 bytes per query token, then 2·B·H·⌈N/1024⌉ β partials.
dβ is summed over every batch element and head, since β is one scalar.
Like the forward, the backward uses no atomics and is bitwise reproducible.

## Build, test, benchmark

From this directory, with the project venv:

```bash
python build.py                                   # JIT build, prints the ptxas report
python -m pytest -q tests/test_reference.py tests/test_kernel_emulation.py   # CPU, seconds
python -m pytest -q tests/test_backward_reference.py tests/test_backward_emulation.py   # CPU, ~10 s
python -m pytest -q tests/test_forward_cuda.py    # GPU
python -m pytest -q tests/test_backward_cuda.py   # GPU
python bench/bench_forward.py                     # N = 2^14 .. 2^20, d = 128, B·H = 4
python bench/bench_backward.py                    # same grid, backward and forward + backward
```

Compile-only check without a GPU or torch:

```bash
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c kernels/race_fwd.cu
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c kernels/race_bwd.cu
```

## Tolerances

The GPU test compares the bf16 output with the fp64 reference on the same bf16-rounded inputs using |O − O_ref| ≤ 2⁻⁸|O_ref| + 10⁻⁵·max(1, β)·max|v|.
2⁻⁸ is the bf16 unit roundoff of the final rounding.
The absolute term covers fp32 error, which the CPU emulation measures at 1.5×10⁻⁸ to 1.4×10⁻⁷ of max|v|, a margin of more than 100×.
The debug A and B are checked at 3×10⁻⁵ relative to their natural scale, which the emulation uses at most 1.4% of.
The full derivation is in `tests/numerics.py`.

The backward's bf16 dq, dk, dv are checked against the fp64 reference VJP with |G − G_ref| ≤ 2⁻⁸|G_ref| + 5×10⁻⁵·max(1, β)·S_G, and the fp32 dβ with 10⁻⁶·max(1, β)·S_β.
S_G is a per-head error scale, not max|G_ref|: dq and dk pass through tanh′ = 1 − u², which fp32 evaluates with an absolute error near 2⁻²³ while whole heads can be saturated, and through dφ = y/Den + A·dDen, whose two terms can nearly cancel.
So S_dq and S_dk are the gradient with respect to u pushed through |W|, with both terms of dφ taken in absolute value, S_dv is max|dv_ref|, and S_β is the matching sum for dβ.
The CPU emulation of the backward's fp32 data flow uses at most 7% of these bounds over P ∈ {1..5}, L ∈ {1..4}, β from 1/√128 to 4 and N from 2 to 5000, and a single query token dropped from dA/dB still fails them.

## Where to look first if the GPU tests fail

- `stage_tokens` / `accumulate_stage` in `race_internal.cuh`: the double-buffer barrier invariant and the masking of tail tokens.
- `query_kernel`: the dynamic shared-memory layout (`QuerySmem`) and the `cudaFuncSetAttribute` opt-in.
- The alignment assumptions of the vector loads: bf16 rows need 8-byte alignment, which the binding checks.
- Performance, not correctness: the query kernel's 80 KB of shared memory at P = 5, L = 4 allows only one CTA per SM on sm_89 (L40S), and `bucket_build_kernel<128, 5>` sits at 60 registers.

For the backward:

- `corner_reduce_scatter` (`race_common.cuh`): the lane and corner bookkeeping of the grouped folds. `tests/test_backward_emulation.py` runs the same algorithm lane by lane, so a mismatch there points at the CUDA transcription.
- The `__syncwarp` placement in `query_grad_kernel` and `key_grad_kernel`: lanes 0..R−1 and 0..P−1 write per-warp scratch that every lane reads after the next `__syncwarp`.
- `reciprocal` in `race_bwd.cu` (hardware approximation plus one Newton step, used for the sigmoids and 1/Den): if dq or dk is off by about 1 ulp everywhere, compare it with `1.0f / x`.
- The weighted staging in `bucket_build_kernel<D, P, true>`: the per-token pair (1/Den, dDen) scales the value row and the mass.
- The launch bound `__launch_bounds__(256, 4)` on the per-token kernels caps them at 64 registers; if a future nvcc spills there, the ptxas report will say so.
