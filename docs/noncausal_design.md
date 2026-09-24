# Non-causal RACE CUDA kernel: regime analysis, verification, and design

This is the design I worked out before writing the non-causal kernels in `src/noncausal/`.
It covers the roofline regime of the forward pass, the numerical check of the Bernoulli factorization the kernels depend on, the three-kernel decomposition, and an outline of the backward.
I keep it as the reference for why the code looks the way it does.
The code departs from it in a few places, each for a concrete reason, and the last section lists those departures.
The regime numbers in section 1 come from `src/regime_analysis.py`, and the factorization and normalization checks come from `src/verify_factorization.py`.

## 0. Facts checked before the design

I checked the starting assumptions against the reference repository and the paper (arXiv 2510.04008).
Three of them matter for the design.

1. **The authors did not leave the CUDA kernel as future work.**
   The paper claims the opposite: contribution III advertises "custom OpenMP/CUDA kernels", and the causal-masking remark says the LM experiments are "implemented efficiently in OpenMP/CUDA (Algorithm 2)".
   What the paper leaves as future work is the causal *theory* (bias-variance analysis), not the kernel.
   What the repo actually ships for the CUDA claim is dead code: `kernels/gpu/*.cu` has no bindings and zero references (traced in `analysis/hot-path.md`).
   So the honest framing is "shipped CUDA kernels are unwired prototypes", not "no GPU implementation was attempted".
2. **The paper's algorithm is not the repo's.**
   Paper Algorithm 1: `Num = (1/L) Σ_l Φ_Q B`, `Den = (1/L) Σ_l Φ_Q A`, `O = Num/Den` (global normalization).
   Repo code (both causal and non-causal): `O = Σ_l Φ_Q (B/(A+1e-6))`, per-bucket normalization, no 1/L.
   Measured gap (`verify_factorization.py` check 2): after restoring the 1/L, they agree to ~1e-4 at the repo's frozen β = 1/√d, but diverge as β grows: 0.7% at β=0.5, 3.7% at β=1, 11% at β=2.
   The paper trains β, so this is not academic.
   **Decision: the paper's Algorithm 1 is the ground truth.**
   It is what the theory analyzes, and it is cheaper in-kernel: one divide per token instead of R·L divides plus per-bucket eps.
3. **W is fixed random (register_buffer, never a Parameter): no dW in backward. β is trainable per the paper ("we treat β as a trainable parameter"), so backward needs dβ (a scalar reduction).**
   The shipped causal code freezes β at 1/√d (the learnable `logit_temp` is commented out); the non-causal scaling variant has it learnable.
   Also confirmed: the CPU kernel is OpenMP (3 `#pragma omp` in `race_pref.cpp`).

Targets: primary target **A100-SXM-40GB (sm_80, 1.56 TB/s, 108 SMs)** because that is the GPU I can actually run (Colab); all tables also carry **H100-SXM-80GB (sm_90, 3.35 TB/s, 132 SMs)** numbers.
Toolkit CUDA 12.x, PyTorch 2.x `load_inline` extension, bf16 I/O with fp32 accumulation.

## 1. Regime analysis

Per token per head (a token is one query row plus one key/value row), M=1, 1 MAC = 2 FLOPs, bf16 I/O, Q/K/V read exactly once, O written once, R-space intermediates never in HBM.

| stage | P=2, L=2, d=128 | P=4, L=4, d=128 |
|---|---|---|
| projection u = tanh(Wx), Q and K sides | 2,048 | 8,192 |
| corner probs (Bernoulli product form) | 32 | 256 |
| bucket aggregation A, B (key side) | 2,056 | 16,448 |
| query mixing Num, Den, O | 2,192 | 16,640 |
| **total FLOPs/token/head** | **6,328** (+16 SFU) | **41,536** (+64 SFU) |
| bytes/token/head (bf16) | 1,024 | 1,024 |
| arithmetic intensity | 6.2 FLOP/B | 40.6 FLOP/B |

Ridge points: A100 fp32 cores 12.5 FLOP/B, tensor-core bf16 201; H100 fp32 20, TC 295.

**Verdict:**

- P=2, L=2: **memory-bound** everywhere.
  Success metric = % of achievable HBM bandwidth.
  Floor at N=1M, 4 heads (4.10 GB moved): **2.63 ms A100 / 1.22 ms H100**.
- P=4, L=4: AI = 40.6 sits **above the fp32-core ridge** on both GPUs, so a plain CUDA-core kernel is compute-bound there (fp32-core floor 8.5 ms A100 / 2.5 ms H100 vs the same 2.6/1.2 ms memory floor).
  The three heavy stages (projection, aggregation, mixing) are all tiny GEMMs, so the tensor-core version returns to memory-bound (AI far below the TC ridge).
- Success metric therefore: **% of HBM bandwidth, with the explicit caveat that v1 (fp32 cores) at P=4, L=4 is measured against the fp32 compute roofline instead**, and the TC rewrite is what closes the gap to bandwidth.

## 2. Bernoulli factorization

Verified numerically for P ∈ {1..5}, β ∈ {1/√128, 1, 4}, 4096 random tokens: max abs deviation between the direct length-R softmax and Π_t [p_t or 1−p_t], p_t = σ(2βu_t), is **4.8e-7** (fp32 rounding).
It is an identity (the partition function separates into Π_t 2cosh(βu_t)), so it holds for all β.
Safe to build on: the kernel computes P sigmoids and a product tree (~2R muls), never a length-R softmax with max-reduce and R exps.

## 3. Kernel decomposition

Three kernels per pass; no N×R tensor ever exists in HBM.

**Kernel 1: bucket build (the reduction over N).**
Grid (⌈N/TILE_N⌉, L, H·M), CTA = 256 threads, TILE_N ≈ 2048.
Shared memory per CTA: fp32 accumulators A[R] and B[R][d] stored padded as B[R][d+1] (stride 129 is odd, so both row walks and fixed-d column walks are bank-conflict-free), plus that table's W (P×d fp32) and a staging buffer.
Budget at the worst case R=32, d=128: B+A = 16.6 KB, W ≤ 2.5 KB, staging ≈ 10 KB → ~29 KB, under the 48 KB default; per-head-per-table B fits in smem with room for 2+ CTAs/SM.
Per token: coalesced float4/bf16x2 loads of K and V, u via warp-level dots (`__shfl_down_sync` only, no implicit warp sync), P sigmoids, φ by product tree in registers.
Accumulation into B without atomics: stage a batch of ~16 tokens' (φ, v) in smem, then threads own fixed (r, d) slots and loop the batch serially.
This is `B += Φᵀ_batch V_batch`, a tiny GEMM, which is exactly the shape the later tensor-core version replaces with wmma.
Each CTA writes one partial A/B to a workspace of shape [nCTA, R, d+1].
Its size is N/TILE_N × R × d, e.g. ~65 MB at N=1M with TILE_N=2048 (489 tiles × L=4 × H=4 partials of 8.3 KB each); this is O(N/TILE_N · Rd), not O(N·R).

**Kernel 2: deterministic reduce.**
Fixed-shape pairwise tree over the nCTA partials → final A[L,H,R], B[L,H,R,d].
Reduction order depends only on (N, TILE_N), never on scheduling → bitwise run-to-run determinism.

Strategies considered for the N-reduction:

| strategy | deterministic | notes |
|---|---|---|
| per-CTA smem partials + tree reduce kernel (chosen) | **yes, bitwise** | fixed order; extra 100-200 MB workspace; error ~√TILE·2⁻²⁴ ≈ 3e-6 rel |
| global fp32 atomicAdd | no | schedule-dependent summation order; rejected on the training-reproducibility requirement |
| cooperative-groups grid sync, single kernel | yes | couples grid size to occupancy, breaks at N=12M grid sizes, harder to tune; not worth it for a 2-kernel saving |
| split-N + atomic second stage | no | same atomics problem, just smaller |

fp32 accumulation error at N=10⁷ with the chosen scheme: per-CTA sums of ≤2048 O(1) terms (~2.7e-6 relative) plus a 13-level tree (~2e-7) → ~3e-6 relative.
The naive single serial sum would be ~100x worse (~2e-4).
Kahan summation is not needed.

**Kernel 3: query pass.**
Grid over N tiles; load the final A, B into smem once per CTA.
Smem budget: B[L,R,d+1] worst case (L=4, R=32, d=128) is 66 KB, over the 48 KB static default, so kernel 3 uses the dynamic shared-memory opt-in (`cudaFuncAttributeMaxDynamicSharedMemorySize`; ceiling 164 KB/SM on A100, 228 KB on H100).
At the typical P=4, L=4 config it is 33 KB and needs no opt-in.
Per token: coalesced Q load, u, φ_Q, `Num = Σ_l φ·B` and `Den = Σ_l φ·A` accumulated in registers, one divide, coalesced bf16 store of O.
Embarrassingly parallel, trivially deterministic.

**Implementation rules:**

- No `__syncthreads` under divergence (all syncs at uniform batch boundaries).
- `_sync` variants everywhere.
- Padding as stated above.
- V loads vectorized (d=128 = 32 float4 or 64 bf16x2 per row).
- `-Xptxas -v` output and a ≤64 registers/thread budget reported with the first compile.
- No `--use_fast_math` (tanh/sigmoid stay accurate); any `__tanhf`-style intrinsic would be a flagged, measured change.
- Tail tiles handle N ∈ {127, 129} by masked loads contributing φ = 0.

**Backward (designed after the forward; derivation before code):**
W fixed → no dW; dβ is a scalar reduction (deterministic tree, same machinery as kernel 2).
Structure mirrors forward: a query-side pass produces dΦ_Q terms and per-bucket adjoints (dB = Σ_i φ_Q,i ⊗ g_i/Den_i, dA likewise), then a key-side pass turns dB/dA into dK, dV.
The product-form φ has the clean Jacobian ∂φ_r/∂u_t = β φ_r (v_{r,t} − (2p_t − 1)), where 2p_t − 1 = E_φ[v_t].
This composes with tanh' = 1 − u², and the full derivation is written out and cross-checked (softmax route vs Bernoulli route) before any backward code.

**Reference and test harness:**
A clarity-first reference implementation and much of the harness already existed for the causal path.
The non-causal reference is ~30 lines on top of `src/verify_factorization.py`.
It comes with the harness: a P × L × d × N cross-product including N ∈ {127, 128, 129}, plus `gradcheck`.

## Decisions

1. Ground truth: the paper's Algorithm 1 (global normalization, 1/L, trainable β), not the repo's behavior (per-bucket, frozen β).
2. β is a run-time argument of the kernels, since the paper trains it.
3. Primary benchmark GPU: A100 (Colab), with H100 numbers analytic.
4. Scope and order: non-causal forward first, then the backward, then the causal v2 kernels; causal is excluded from v1.

## What the implementation changed

The kernels in `src/noncausal/kernels/` follow this design with these differences (the reasons are in the header of `race_fwd.cu` and in `src/noncausal/README.md`):

- The bucket build keeps B in registers (each thread owns fixed (r, c) slots for the whole tile) instead of a shared B[R][d+1].
  The staging and slot ownership of section 3 already make every slot private to one thread, so shared memory would only add a load and a store per FMA.
  The R·(d+1) shape survives as the workspace slice (B rows, then A).
- The build grid is (tiles·L, B·H), with the table index fastest in blockIdx.x, so the L CTAs of a tile share K and V through L2 instead of reading HBM L times.
- The build stages 16 tokens at a time, with φ and v in double-buffered shared memory.
- Each reduce launch folds 8 tiles with a fixed pairwise tree.
  The whole reduction is still one binary tree over tiles, but it takes ⌈log₈(tiles)⌉ launches instead of ⌈log₂(tiles)⌉.
  The order depends only on N, so results are still bitwise reproducible.
- β is read from device memory (a one-element tensor), so a trainable β on the GPU never needs a host sync (`.item()`) per call.
- Lane r computes φ[r] directly as a P-term product (lanes in parallel) instead of one thread building all R values with a product tree.
- The query kernel uses grid (⌈N/1024⌉, B·H) with one warp per query token.
  It loads the final A, B and all planes into dynamic shared memory (up to 80 KB, opted in above 48 KB).
- The workspace is one fp32 buffer of shape [tiles, B·H, L, R·(d+1)], tiles = ⌈N/2048⌉.
  In each slice, B[r][c] is at r·d + c and A[r] is at R·d + r.
  Example: N = 2²⁰, B·H = 4, L = 4, P = 4, d = 128 gives 512 · 4 · 4 · 2064 floats ≈ 68 MB.
- If Den underflows to exactly 0 (only possible for very large β), the output row is 0.
- The backward is five launches: a query-gradient kernel, the forward's bucket build run over (q, dO) with a per-token pair (1/Den, dDen) as weights, the forward's tree reduce over the dA/dB tiles, a key-gradient kernel, and a fixed-order β reduce.
  It saves q, k, v, W, β and the reduced A, B, and saves neither O nor the per-token Den: Den is recomputed from A, and g · Num comes out of the same dot products B_l[r] · g that the gradient with respect to φ(q) needs.
  The derivation is in `src/noncausal/BACKWARD_DERIVATION.md`.
  Like the forward, it uses no atomics and is bitwise reproducible.
- The backward computes the sigmoids and 1/Den with a hardware reciprocal approximation plus one Newton step.
- The per-token kernels use `__launch_bounds__(256, 4)`, which caps them at 64 registers per thread, and every kernel compiles with no spills.
- The tensor-core forward (`kernels/race_fwd_tc.cu`, `tensor_cores=True`) replaces the bucket build and the query pass and keeps the tree reduce and the workspace layout.
  One CTA hashes all L tables of a stage of 32 tokens, so the projection is one [32 × d] × [d × L·P] product and K, Q and V are read from HBM once; the build grid is (tiles, B·H) instead of (tiles·L, B·H).
  W enters the projection as bf16 hi + lo (two MMAs), Φ is rounded to bf16 for the build and query fragments, and the reduced B enters the query as hi + lo.
  A and Den are summed on CUDA cores from the same rounded Φ, so the rounding perturbs convex weights instead of scaling O; `tests/numerics.py` derives the resulting bound.
  On the A10G it runs P = 4, L = 4 at 76% of HBM peak (N = 2²⁰, 4.4× the fp32-core path), and ncu shows both of its kernels memory-bound (82% to 86% DRAM throughput), as section 1 predicts.
  The fp32-core path stays the default and the backward is unchanged.
- The per-token hash dot products use a `__shfl_xor_sync` butterfly, which leaves the sum in every lane, rather than the `__shfl_down_sync` tree named in section 3; the butterfly gives all lanes the same bitwise value, which the tensor-core and backward paths rely on.
