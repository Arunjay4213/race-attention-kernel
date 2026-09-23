# Non-causal RACE Attention (CUDA)

RACE Attention (arXiv 2510.04008, Algorithm 1, global normalization) as bf16 CUDA kernels: a three-kernel forward in two variants (fp32 cores, and tensor cores with `tensor_cores=True`) and a five-launch backward, with an fp64-capable PyTorch reference, an autograd wrapper, and tests.

Status: validated on an NVIDIA A10G (sm_86, 99 KB opt-in shared memory) with torch 2.10 and CUDA 12.8.
All 2170 tests in `tests/` pass there (CPU and GPU, forward in both variants and backward), and compute-sanitizer memcheck, racecheck, synccheck and initcheck report no errors on `infra/sanitize_cases.py noncausal`.
The kernels compile with nvcc 12.8 for sm_80, sm_86, sm_89 and sm_90 with no spills; the fp32-core kernels use at most 64 registers per thread, the tensor-core kernels up to 202 (table below).
Measurements of the fp32-core path and the one bug found on the GPU are in `benchmarks/a10g_first_run.md`; the tensor-core path is measured in the last section of this file.
The fp32-core path stays the default; `ops.race_attention` uses it.

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
| `kernels/race_fwd_tc.h` | launcher API of the tensor-core forward (same shapes and workspace as `race_fwd.h`) |
| `kernels/race_fwd_tc.cu` | the tensor-core bucket build and query pass (bf16 wmma, fp32 accumulation) |
| `kernels/race_bwd.h` | plain C++ backward launcher API and backward workspace layout |
| `kernels/race_bwd.cu` | the backward's query-gradient, key-gradient and β-reduce kernels and launchers |
| `kernels/race_internal.cuh` | shared launch constants, dispatch, and the bucket build kernel (forward A/B and backward dA/dB) |
| `kernels/race_common.cuh` | device helpers: bf16 vector I/O, warp all-reduce, sigmoid pair, corner probability, corner reduce-scatter |
| `kernels/torch_binding.cpp` | `forward`, `forward_debug(...) -> (o, A, B)`, `forward_train(...) -> (o, bucket_totals)`, each with `tensor_cores=False`, and `backward(...) -> (dq, dk, dv, dbeta)` |
| `build.py` | JIT build via `torch.utils.cpp_extension.load`, `load_extension()` |
| `tests/test_reference.py` | CPU tests of the forward reference |
| `tests/test_backward_reference.py` | CPU fp64 check of the backward derivation against autograd and finite differences |
| `tests/test_kernel_emulation.py` | CPU checks of the forward kernels' index math and an fp32 emulation of their summation order |
| `tests/test_backward_emulation.py` | the same for the backward: reduce-scatter, layouts, fp32 data flow |
| `tests/test_forward_cuda.py` | forward GPU tests (skipped without CUDA) |
| `tests/test_backward_cuda.py` | backward and autograd GPU tests (skipped without CUDA) |
| `tests/numerics.py` | input generation and the derived tolerances |
| `bench/bench_forward.py` | forward timing against the HBM roofline, `--variant fp32`, `tc` or `both` |
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

### Tensor-core forward (`tensor_cores=True`)

`race_fwd_tc.cu` replaces kernels 1 and 3; the tree reduce and the workspace are shared with the fp32 path, so `forward_debug` and `forward_train` return the same layouts and the backward accepts either forward's totals.
Both kernels hash one stage of 32 tokens at a time for all L tables at once (256 threads, 8 warps):

1. Projection Z = X Wᵀ, [32 × d] × [d × NP] with NP = L·P rounded up to 16, on bf16 wmma (m16n16k16) with fp32 accumulation.
   W is split into hi = bf16(W) and lo = bf16(W − hi), two MMAs per k-step, so the planes are exact to 2⁻¹⁶ relative.
   The 2 × NP/16 output fragments are split over the k dimension so all 8 warps work; the hash step adds the parts in a fixed order.
2. Hash on CUDA cores: tanh, the sigmoid pair, and the Bernoulli product per (token, stacked corner l·R + r), rounded to bf16.
3. `bucket_build_tc_kernel`, grid (tiles, B·H), 2048 keys per CTA: B += Φᵀ V, [L·R × 32] × [32 × d], with the B fragments in registers for the whole tile (warp w owns column tile w mod d/16).
   A is summed on CUDA cores from the same rounded Φ, one fixed corner per thread.
   K and V arrive by 16-byte `cp.async`, V double-buffered, zero-filled past N.
4. `query_tc_kernel`, grid (⌈N/1024⌉, B·H): Num = Φ_Q B, [32 × L·R] × [L·R × d], with B split into hi + lo and held as register-resident fragments for the whole CTA; Den = Φ_Q A on CUDA cores from the same rounded Φ_Q and fp32 A; one reciprocal per token and a 16-byte bf16 store.

All shared memory is static (at most 40064 bytes), and every kernel compiles without spills.
q, k and v must be 16-byte aligned for this path.
The only new rounding is that of Φ to bf16, which A and Den share with B and Num, so it perturbs the weights of a convex combination instead of scaling O; `tests/numerics.py` derives the bound.

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
python bench/bench_forward.py                     # N = 2^14 .. 2^20, d = 128, B·H = 4, both variants
python bench/bench_backward.py                    # same grid, backward and forward + backward
```

Compile-only check without a GPU or torch:

```bash
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c kernels/race_fwd.cu
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c kernels/race_fwd_tc.cu
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c kernels/race_bwd.cu
```

## Tolerances

The GPU test compares the bf16 output with the fp64 reference on the same bf16-rounded inputs using |O − O_ref| ≤ 2⁻⁸|O_ref| + 10⁻⁵·max(1, β)·max|v|.
2⁻⁸ is the bf16 unit roundoff of the final rounding.
The absolute term covers fp32 error, which the CPU emulation measures at 1.5×10⁻⁸ to 1.4×10⁻⁷ of max|v|, a margin of more than 100×.
The debug A and B are checked at 3×10⁻⁵ relative to their natural scale, which the emulation uses at most 1.4% of.
The full derivation is in `tests/numerics.py`.

The tensor-core forward is checked with |O − O_ref| ≤ 2⁻⁸|O_ref| + (1 + 2⁻⁸)·E_tc + the same absolute term, where E_tc = (2δ + δ²)/(1 − δ)²·(M + |O_ref|) plus a 2⁻¹⁶ term for the hi + lo split of B.
δ bounds the relative perturbation of every Φ entry: 2⁻⁸ from the bf16 rounding, raised by the exactly computed effect of the hi + lo planes (at most 1.1×10⁻³ at β = 4 on the test inputs); M is the Φ-weighted mean of |v|.
The bound holds for every sign pattern of the roundings and needs no first-order approximation, because A and Den use the same rounded Φ as B and Num.
It is about 1.2×10⁻³ to 3.2×10⁻³ of max|v|, far looser than the fp32 path's, while the CPU emulation of the tensor-core data flow measures errors of at most 2×10⁻⁴ of max|v| before the final rounding.
At β = 0 every Φ is exactly 2⁻ᴾ, so there the tensor-core path must meet the fp32 tolerance, which sees a single dropped key.
A and B from `forward_debug` are checked with δ·A and δ·∑ⱼ φⱼ|vⱼ| added to the fp32 bounds.

The backward's bf16 dq, dk, dv are checked against the fp64 reference VJP with |G − G_ref| ≤ 2⁻⁸|G_ref| + 5×10⁻⁵·max(1, β)·S_G, and the fp32 dβ with 10⁻⁶·max(1, β)·S_β.
S_G is a per-head error scale, not max|G_ref|: dq and dk pass through tanh′ = 1 − u², which fp32 evaluates with an absolute error near 2⁻²³ while whole heads can be saturated, and through dφ = y/Den + A·dDen, whose two terms can nearly cancel.
So S_dq and S_dk are the gradient with respect to u pushed through |W|, with both terms of dφ taken in absolute value, S_dv is max|dv_ref|, and S_β is the matching sum for dβ.
The CPU emulation of the backward's fp32 data flow uses at most 7% of these bounds over P ∈ {1..5}, L ∈ {1..4}, β from 1/√128 to 4 and N from 2 to 5000, and a single query token dropped from dA/dB still fails them.
On totals from the tensor-core forward the backward is checked twice: against the fp64 VJP evaluated at those totals with the unchanged tolerance (up to 98% used, as for the fp32 path), and against the exact VJP with the tolerance plus a bound, computed in fp64 without approximation, on how far the totals' 2⁻⁸ rounding moves each gradient.
Without that bound the round trip exceeds the backward tolerance by up to 7.2× (dq), 2.8× (dk), 11.5× (dv) and 90× (dβ); with it, it uses at most 4.3%, 7.3%, 56% and 0.1%.

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

## Tensor-core forward on the A10G

Date: 2026-09-22.
Hardware and software as in the status line (A10G, 600 GB/s peak HBM, 80 SMs, torch 2.10, nvcc 12.8, ncu 2026.1 from CUDA 13.2).
Every number below was measured on that box on the same day; the GPU is shared with other work, but no other process was on it during the timed runs, and two full benchmark runs agreed to within 1%.

### Forward time, d = 128, B·H = 4

Median of 20 runs after 5 warmups (`bench/bench_forward.py`); GB/s counts only Q, K, V read once and O written once, so % of peak is a lower bound.

| P, L | N | fp32 ms | fp32 % peak | tc ms | tc % peak | speedup |
|---|---|---|---|---|---|---|
| 2, 2 | 2¹⁴ | 0.488 | 22.9% | 0.206 | 54.3% | 2.37× |
| 2, 2 | 2¹⁶ | 0.804 | 55.7% | 0.588 | 76.1% | 1.37× |
| 2, 2 | 2¹⁸ | 2.566 | 69.8% | 2.305 | 77.7% | 1.11× |
| 2, 2 | 2²⁰ | 9.854 | 72.6% | 8.757 | 81.7% | 1.13× |
| 4, 4 | 2¹⁴ | 1.294 | 8.6% | 0.335 | 33.4% | 3.86× |
| 4, 4 | 2¹⁶ | 3.047 | 14.7% | 0.685 | 65.3% | 4.45× |
| 4, 4 | 2¹⁸ | 10.520 | 17.0% | 2.488 | 71.9% | 4.23× |
| 4, 4 | 2²⁰ | 41.245 | 17.4% | 9.450 | 75.7% | 4.36× |

At P = 4, L = 4 the tensor-core path takes the forward from 17% to 76% of peak HBM bandwidth at N = 2²⁰, 4.4× faster.
At P = 2, L = 2 the fp32 path was already mostly memory-bound, and the gain is 11% to 13% at N ≥ 2¹⁸.

### ncu, B·H = 4, N = 2¹⁸, d = 128, P = 4, L = 4

`ncu --set full` (with sudo, for the performance counters) on one forward per variant.
The tree reduce is the same in both (3 launches, 52 µs in total, 88% DRAM in the first pass).

| kernel | µs | DRAM % | SM % | tensor pipe % | occupancy achieved / theoretical | regs | smem bank conflicts | top stalls |
|---|---|---|---|---|---|---|---|---|
| bucket_build<128,4> (fp32) | 4499 | 24.6 | 71.0 | 0 | 63.5 / 66.7 | 63 | 0.3% of wavefronts | short scoreboard |
| query_kernel<128,4> (fp32) | 5773 | 18.0 | 51.9 | 0 | 31.6 / 33.3 | 43 | 2.8% | short scoreboard |
| bucket_build_tc<128,4,4> | 1203 | 86.3 | 45.4 | 17.1 | 33.1 / 33.3 | 102 | 6.6% | barrier, wait |
| query_tc<128,4,4> | 1229 | 82.0 | 48.8 | 25.5 | 32.1 / 33.3 | 128 | 19.3% | math pipe throttle, wait |

Both tensor-core kernels are memory-bound (82% to 86% of DRAM throughput, under half of the SM throughput), which is the regime section 1 of `docs/noncausal_design.md` predicts once the GEMM-shaped stages leave the fp32 pipe.
Theoretical occupancy is 2 CTAs (16 warps) per SM, limited by registers and shared memory together.
The byte floor at this size is 1.79 ms (Q, K, V, O at 600 GB/s) against 2.49 ms measured.
Candidates for the rest, not yet measured one by one: the query kernel's shared-memory bank conflicts (19% of its wavefronts), the barriers between the stage phases (the build's top stall), and the last partial wave of the build (512 CTAs on 160 slots).

### ptxas on sm_86 (nvcc 12.8, torch build flags)

Registers, static shared memory in bytes, and spill bytes per instantiation; no image spills on sm_80, sm_86, sm_89 or sm_90.

| d | P | L | build regs | build smem | build spill | query regs | query smem | query spill |
|---|---|---|---|---|---|---|---|---|
| 64 | 1..3 | 1..4 | 76-78 | 22784-25600 | 0 | 75-79 | 22016-24832 | 0 |
| 64 | 4 | 1, 2, 3, 4 | 75, 78, 94, 92 | 23552-26624 | 0 | 75, 80, 110, 120 | 22784-25984 | 0 |
| 64 | 5 | 1, 2, 3, 4 | 80, 95, 106, 121 | 23808-27648 | 0 | 79, 120, 150, 168 | 23040-27776 | 0 |
| 128 | 1..3 | 1..4 | 80-93 | 35072-37888 | 0 | 79-111 | 34816-37632 | 0 |
| 128 | 4 | 1, 2, 3, 4 | 83, 94, 102, 102 | 35840-38912 | 0 | 80, 111, 118, 128 | 35584-38784 | 0 |
| 128 | 5 | 1, 2, 3, 4 | 98, 102, 132, 202 | 36096-39936 | 0 | 111, 126, 162, 186 | 35840-40064 | 0 |

The largest configurations keep the most register-resident fragments: at d = 128, P = 5, L = 4 each build warp holds 8 accumulator tiles and each query warp 16 B fragments, so those run one CTA per SM.

