# Causal RACE v2: chunk-parallel CUDA forward - design

This is the design I worked out before writing the causal v2 kernels in `src/causal_v2/`.
It plays the same role for the causal kernels that `docs/noncausal_design.md` plays for the non-causal ones, and I keep it as the reference for why the code looks the way it does.
The code departs from it in a few places, each for a concrete reason, and the last section lists those departures.
Scope: the forward pass in two builds (v2a on fp32 CUDA cores, v2b on bf16 tensor cores), a backward outline, and the test and benchmark plan.
No GPU was used for the design.
Every number is either paper arithmetic (formulas in Appendix A) or the output of small CPU experiments (listed in Appendix B).

## 0. Decisions at a glance

| question | decision | section |
|---|---|---|
| ground truth | paper Algorithm 1 made causal (global normalization): O_i = ∑_{j≤i} k_ij v_j / ∑_{j≤i} k_ij, k_ij = ⟨Φ_Q,i, Φ_K,j⟩, β a runtime argument | 2 |
| decomposition | two levels: tiles of T_blk tokens carry the scanned state along the sequence; sub-chunks of C tokens inside a tile use masked matmuls | 3 |
| kernels | K1 tile sums (the non-causal bucket build with TILE_N = T_blk), K2 exclusive scan over tiles, K3 output pass | 4, 5, 6 |
| C (sub-chunk) | 64, a template parameter (32 for config B where shared memory is short) | 9 |
| T_blk (state tile) | 2048 for config A, 4096 for config B, chosen at launch, smaller when T is small | 3 |
| store Φ in HBM or recompute | recompute | 6.4 |
| build order | v2a first (fp32 CUDA cores), then v2b (wmma, bf16 in, fp32 accumulate) | 6.5 |
| workspace at T = 2M, 8 streams | 130 MiB (config A), 258 MiB (config B) | 8 |
| predicted floor, config A, T = 2M, 8 streams | L40S 15.5 ms, A100 8.6 ms, H100 4.0 ms (v2b, memory-bound) | 9 |
| predicted floor, config B, T = 2M, 8 streams | L40S 31.1 ms, A100 17.3 ms, H100 8.0 ms (v2b, memory-bound at peak rates) | 9 |

Two findings shaped this design.
Each is explained where it occurs.

1. One saved state per C-token chunk (the textbook chunked linear attention layout) is the wrong granularity for RACE.
   The state has S·(d+1) floats, which is about as large as the chunk it summarizes, so for C ≤ 128 (config A) and C ≤ 256 (config B) the state traffic is larger than all Q/K/V/O traffic, and the scan is not a small fraction of the time (section 3.2).
   The fix is to separate the matmul granularity (C) from the state granularity (T_blk): state records every T_blk = 2048-4096 tokens, masked matmuls every C = 64 tokens (section 3.3).
2. The per-bucket normalization of src/race_baseline.py is also chunk-parallel.
   The division depends only on A, which folds into a per-query weight, so the same masked-matmul structure works (verified to 2e-15 in fp64).
   So feasibility does not force the ground-truth choice; Algorithm 1 is still the choice, for the reasons in section 2.3.

## 1. Notation and the two benchmark configurations

- N = B·H·M independent streams (M = 1 ensemble here), T tokens per stream, d the head dimension.
- P hash bits per table (the repo calls this K), R = 2^P buckets per table, L tables, S = L·R features per token.
- W ∈ R^{(L·P) × d} is the fixed random plane matrix (the repo buffer planes_T, transposed), fp32, never trained.
- For one row x (a query or key): u = tanh(W x) ∈ R^{L·P}, p = σ(2β u), and for table l and corner r, φ_l,r(x) = ∏_{t=1..P} (p_l,t if bit t of r is +1, else 1 - p_l,t).
  Bit t of r counts from the most significant bit, which reproduces the repo's itertools.product([-1, +1]) corner order.
  This product form equals the repo's softmax over corners (4.8e-7 in fp32 in docs/noncausal_design.md section 2; 3.6e-15 in fp64 against race_baseline for the causal path, Appendix B).
- Φ(x) ∈ R^S is φ for all L tables concatenated, table-major.
- β: the repo freezes β = 1/√d (it computes tanh(proj)/√d before the corner softmax); the paper trains β.
- k_ij = ⟨Φ_Q,i, Φ_K,j⟩ = ∑_l ∑_r φ_Q,i,l,r · φ_K,j,l,r, always > 0.
- Causal prefix state after token i: B_i = ∑_{j≤i} Φ_K,j ⊗ v_j ∈ R^{S×d} and A_i = ∑_{j≤i} Φ_K,j ∈ R^S.
  The pair (A, B) is "the state"; it is S·(d+1) floats per stream.
- "Stream" always means one (batch, head) sequence; the letter B always means the state above.

| | config A (the benchmarked layer; v1's config) | config B (large) |
|---|---|---|
| d | 64 | 128 |
| P, R | 4, 16 | 5, 32 |
| L | 4 | 4 |
| S = L·R | 64 | 128 |
| state S·(d+1) fp32 | 4,160 floats = 16,640 B | 16,512 floats = 66,048 B |

Benchmark shape everywhere below: N = 8 streams (batch 1, 8 heads, M = 1) and T = 2^21 = 2,097,152 tokens ("2M"), so N·T = 16,777,216 token rows.

## 2. Ground truth

### 2.1 The two candidates

Per-bucket (repo, src/race_baseline.py): O_i = ∑_l ∑_r φ_Q,i,l,r · B_i,l,r / (A_i,l,r + 1e-6), with no 1/L.
Algorithm 1 made causal (global): O_i = (∑_l Φ_Q,i,l · B_i,l) / (∑_l Φ_Q,i,l · A_i,l) = ∑_{j≤i} k_ij v_j / ∑_{j≤i} k_ij.
The paper's 1/L factor multiplies both the numerator and the denominator, so it cancels in the global form.
The global form is exactly normalized causal linear attention with the feature map Φ of dimension S.
That is why the standard chunked linear attention method applies to it without changes.

### 2.2 Finding: both forms are chunk-parallel

For the per-bucket form, write O_i = ∑_s w_i,s · B_i,s with w_i,s = φ_Q,i,s / (A_i,s + eps).
B_i is a sum over keys, so O_i = w_i · B_prev + ∑_{j in chunk, j≤i} (w_i · Φ_K,j) v_j.
This is the same masked matmul with Φ_Q replaced by w.
The only extra work is A_i,s for every query (a C×S inclusive running sum per chunk) and S divisions per token.
Checked in fp64: the chunked per-bucket form matches the cumsum per-bucket form to 1.8e-15 (config A) and 2.2e-15 (config B).
So the statement that per-bucket normalization needs the 5-D tensor is true only of the cumsum implementation, not of the math.

### 2.3 Decision: Algorithm 1 (global normalization)

1. It is the same estimator the non-causal kernels compute, so K1, the Num/Den query mixing, and the Python references are shared.
2. It is what the paper's theory analyzes, and the paper trains β; the two forms diverge quickly as β grows (table below).
3. It is cheaper: one reciprocal per token instead of S divisions plus a C×S running sum per chunk, and no eps.
4. Its output is a convex combination of the values (weights k_ij / ∑_j k_ij are ≥ 0 and sum to 1).
   So the output is bounded by the values, and a constant V must come back unchanged, which gives a sharp kernel test (section 7).
   The repo form sums L separate convex combinations, so it carries a factor L and an eps bias.
5. Its backward is the standard linear attention backward with one extra column (section 11).

The one reason not to: the authors' released LM checkpoints were trained with the per-bucket form.
Loading them would need the per-bucket variant.
Because it has the same chunk structure (2.2), it can be added later as a template flag ("v2c") without redesign.
It is not part of the first build.

Measured gap between the two forms (fp64, config A shapes, the repo's own planes, T = 1024; RMS over all tokens):

| β | ‖O_alg1 - O_repo/L‖ / ‖O_alg1‖ |
|---|---|
| 1/√64 = 0.125 (repo, frozen) | 2.0e-3 |
| 0.5 | 3.0e-2 |
| 1.0 | 1.45e-1 |
| 2.0 | 3.75e-1 |

Where the gap sits at β = 1/√d (T = 2048, max per-row relative gap):

| token positions | max per-row relative gap |
|---|---|
| 0 | 1.7e-5 (only the eps term) |
| 1-15 | 4.4e-3 |
| 16-63 | 3.0e-3 |
| 64-511 | 2.1e-3 |
| 512-2047 | 6.3e-4 |

The causal gap is larger than the non-causal one (1e-4) because early queries see only a few keys, and per-bucket averaging and global averaging differ most when each bucket holds few keys.

### 2.4 What the equivalence test needs

1. **A reference variant.**
   A new fp64 function, alg1_causal_reference(Q, K, V, planes, β), under src/.
   It reuses BatchedACE's buffers (planes_T, protos_T) and its softmax-over-corners φ, not the Bernoulli product form, so the kernel test also checks the factorization.
   It takes β as a parameter: logits = β · tanh(x planes_T) @ protos_T.
   For T ≤ 4096 it builds the dense masked T×T matrix tril(Φ_Q Φ_Kᵀ); for longer T it uses the chunked form, which matched the dense one to 2.5e-16.
2. **A bridge test, Python only.**
   The per-bucket variant of the same reference at β = 1/√d must match race_baseline.BatchedACE to ≤ 1e-12 in fp64 (measured 3.6e-15).
   This proves the new reference uses the same planes, corner order and β convention as the repo.
   The Algorithm-1-versus-repo gap is then reported (tables above), not asserted.
3. **Kernel tests compare against the Algorithm 1 reference, never directly against race_baseline.**
   The reference gap (2e-3 RMS) is the same size as bf16 output rounding (1.5e-3 RMS).
   A tolerance loose enough to accept the gap would also accept real bugs of that size.
   Example: an off-by-one causal mask (dropping the diagonal) changes O_i by about |v_i - O_i| / (i+1), which is below 2e-3 for i above ~500.
4. **Inputs.**
   Q, K, V are generated in bf16 and upcast to fp64 for the reference, so input rounding is not counted as kernel error.
5. **β.**
   Test β ∈ {1/√d, 0.5, 1, 2}; benchmark at the frozen 1/√d (the configuration of the measured 131,072-token wall).
6. **Tolerance.**
   Let F = max|bf16(ref) - ref| (the unavoidable output rounding) and F_rms its RMS.
   v2a: max error ≤ 1.1·F and RMS ≤ 1.05·F_rms.
   v2b: max error ≤ 2.5·F and RMS ≤ 2·F_rms.
   The CPU simulation (section 7) gives v2a = 1.00·F and v2b = 1.15·F at β = 1/√d.

## 3. The decomposition

### 3.1 The chunk identity

Take a block of consecutive rows a..a+C-1 of one stream, and let (A_prev, B_prev) be the state after row a-1 (zero if a = 0).
For a query i in the block:
Num_i = Φ_Q,i · B_prev + ∑_{a≤j≤i} k_ij v_j, and Den_i = Φ_Q,i · A_prev + ∑_{a≤j≤i} k_ij.
In matrix form, with Φ_Q, Φ_K ∈ R^{C×S} and V ∈ R^{C×d} holding the block's rows:

- G = tril(Φ_Q Φ_Kᵀ), a C×C matrix whose mask keeps j ≤ i, including the diagonal j = i.
- Num = Φ_Q B_prev + G V (C×d), Den = Φ_Q A_prev + G 1 (C), O = Num / Den row by row.
- Then the state moves past the block: B ← B + Φ_Kᵀ V and A ← A + Φ_Kᵀ 1.

The carry term uses the state before the block (exclusive), and the mask includes the diagonal, so query i sees exactly keys 0..i.
That matches the reference, whose cumsum includes token t itself.
Checked in fp64 against the dense masked T×T form: 2.5e-16 (config A) and 3.9e-16 (config B), also with a tile length (320) that is not a power of two.

### 3.2 Why one saved state per C-token chunk is the wrong granularity

In the textbook layout, K1 writes one state record per C-token chunk, K2 scans the records, and K3 reads one record per chunk.
A record is S·(d+1)·4 bytes and is touched 4 times (K1 write, K2 read, K2 write, K3 read), so the state traffic is 16·S·(d+1)/C bytes per token.
The Q/K/V/O traffic of this design is 12d bytes per token (K1 reads K and V, K3 reads Q, K, V and writes O).

Arithmetic at T = 2M, N = 8, on A100 (1555 GB/s):

| config | C | chunks per stream | workspace | state bytes/token | Q/K/V/O bytes/token | state / QKVO | K2 time (2·ws/BW) | Q/K/V/O time |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 32,768 | 4.06 GiB | 1,040 | 768 | 135% | 5.61 ms | 8.29 ms |
| A | 128 | 16,384 | 2.03 GiB | 520 | 768 | 68% | 2.81 ms | 8.29 ms |
| A | 256 | 8,192 | 1.02 GiB | 260 | 768 | 34% | 1.40 ms | 8.29 ms |
| B | 64 | 32,768 | 16.1 GiB | 4,128 | 1,536 | 269% | 22.3 ms | 16.6 ms |
| B | 128 | 16,384 | 8.06 GiB | 2,064 | 1,536 | 134% | 11.1 ms | 16.6 ms |
| B | 256 | 8,192 | 4.03 GiB | 1,032 | 1,536 | 67% | 5.57 ms | 16.6 ms |

So with one state per chunk the scan is not a small fraction of the time.
At config B the scan alone takes longer than all Q/K/V/O traffic, and the workspace would use 40% of an A100-40GB.
The reason is sizes: a state record is 4·S·(d+1) bytes, while the chunk's own K and V are 4·C·d bytes, so the ratio is about S/C.
With S = 64-128 and C = 64-256 the two are the same size.
Raising C to thousands would fix the ratio but the intra-chunk matmul cost grows linearly with C (section 9), so C cannot carry both jobs.

### 3.3 The two-level fix: state tiles of T_blk tokens, sub-chunks of C tokens

- A **tile** is T_blk consecutive tokens of one stream (T_blk a multiple of C).
  K1 writes one state record per tile and K2 scans tiles.
- A **sub-chunk** is C tokens.
  One K3 CTA owns one tile, walks its sub-chunks in order, keeps the running (A, B) on chip, and applies section 3.1 to each sub-chunk.
- Cost: the product Φ_Kᵀ[V|1] is computed twice, once in K1 for the tile total and once in K3 for the running state.
  That is +2·S·(d+1) FLOPs per token, cheap on tensor cores.
- State traffic falls to 16·S·(d+1)/T_blk bytes per token.

| config | T_blk | tiles per stream | K3 CTAs | workspace | state bytes/token (% of 12d) | K2 time L40S / A100 / H100 |
|---|---|---|---|---|---|---|
| A | 1024 | 2,048 | 16,384 | 260 MiB | 65.0 (8.5%) | 0.63 / 0.35 / 0.16 ms |
| A | **2048** | 1,024 | 8,192 | 130 MiB | 32.5 (4.2%) | 0.32 / 0.18 / 0.08 ms |
| A | 4096 | 512 | 4,096 | 65 MiB | 16.2 (2.1%) | 0.16 / 0.09 / 0.04 ms |
| A | 8192 | 256 | 2,048 | 32 MiB | 8.1 (1.1%) | 0.08 / 0.04 / 0.02 ms |
| B | 1024 | 2,048 | 16,384 | 1,032 MiB | 258 (16.8%) | 2.51 / 1.39 / 0.65 ms |
| B | 2048 | 1,024 | 8,192 | 516 MiB | 129 (8.4%) | 1.25 / 0.70 / 0.32 ms |
| B | **4096** | 512 | 4,096 | 258 MiB | 64.5 (4.2%) | 0.63 / 0.35 / 0.16 ms |
| B | 8192 | 256 | 2,048 | 129 MiB | 32.2 (2.1%) | 0.31 / 0.17 / 0.08 ms |

Choice: T_blk = 2048 for config A and 4096 for config B.
Both keep state traffic near 4% of the Q/K/V/O traffic and K2 near 2% of the total time (0.18 ms of an 8.6 ms A100 floor for A; 0.35 ms of 17.3 ms for B).
Both still give thousands of CTAs (8,192 and 4,096), which is 38 waves on A100 in both cases (section 10).
Larger T_blk saves little more and starts to cost parallelism at shorter T.

For small problems the launcher picks T_blk = clamp(pow2_floor(N·T / (4 · SMs · CTAs_per_SM)), C, T_blk_default) so the GPU is still filled.
Example: N = 8, T = 4096 on A100 gives T_blk = C = 64 and 512 CTAs; the workspace is then tiny in absolute terms because T is small.

### 3.4 Why not a single-pass scan

- **Decoupled look-back** (the CUB single-pass scan) lets each tile add a mix of predecessor aggregates and inclusive prefixes, and which mix it uses depends on timing.
  Different mixes mean different summation orders, so results are not bitwise reproducible.
  Rejected for the same reproducibility requirement as the non-causal design.
- **Chained scan** (each tile waits for its predecessor's inclusive prefix, then adds its own total) is deterministic, but it is a serial chain of 1,024 dependent L2 round trips per stream and needs forward-progress care (tile tickets).
  It also saves no bytes: the CTA still reads the tile's K and V twice (once for the tile total, once for the outputs), because 216 resident tiles × 512 KB of K and V (config A) is 110 MB, far more than the 40 MB L2.
- So any design that is parallel over T and scans states moves at least 12d bytes per token.
  The single-pass minimum of 8d (read Q, K, V once, write O once) is only reachable by a sequential per-stream scan, which is v1's shape and has no parallelism over T.
  Section 9 reports both floors so the 1.5x gap is visible.

## 4. Kernel 1: tile sums

Math: for tile t of stream n, B_t = ∑_{j in tile} Φ_K,j ⊗ v_j and A_t = ∑_{j in tile} Φ_K,j.

- This is exactly the non-causal bucket build with TILE_N = T_blk.
  Same grid (tiles, L, streams), same 256-thread CTA, same shared-memory accumulators A[R] and B[R][d+1] (padded), same product-form φ, same batch-of-16 staging with fixed (r, d) ownership and no atomics.
  The only difference is what happens to the per-CTA partials afterwards: the non-causal path reduces them to one total, the causal path keeps every tile's partial for the scan.
  So v2a's K1 is the non-causal K1 binary, unchanged.
  src/noncausal/ did not exist yet when I wrote this design, so it depends only on the design in docs/noncausal_design.md section 3.
- **Workspace layout (to be shared with the non-causal kernels).**
  Two tensors: ws_B of shape [N][nTiles][L][R][d] fp32 and ws_A of shape [N][nTiles][L][R] fp32, instead of one [..][R][d+1] tensor.
  Rows of d floats are 16-byte aligned (float4 loads) and have a leading dimension that is a multiple of 4 floats, which wmma::load_matrix_sync requires for fp32 data; rows of d+1 floats satisfy neither.
  The +1 padding belongs in shared memory (bank conflicts), not in global memory.
  If the non-causal K1 ships with [R][d+1], K2 reads that layout through its stride and writes the split layout; it costs nothing measurable.
- **Tail rows.**
  Rows j ≥ T must contribute φ = 0, set explicitly.
  Zero-filling K is not enough: φ(0) = 1/R per bucket, not 0.
- **v2b K1.**
  The tensor-core K1 is the K3-TC kernel instantiated with EMIT_OUTPUT = false: it loads only K and V, computes the K-side projection and φ_K, and runs the state-update matmul over the whole tile.
  If the non-causal kernels get their own tensor-core K1 first, v2b reuses that instead.
  Either way A_t must be summed from exactly the bf16-rounded φ values that multiplied v in B_t (the consistency rule of section 7.4).

## 5. Kernel 2: exclusive scan over tiles

Math: prefix_t = ∑_{t' < t} total_t' for t = 0..nTiles-1, per stream, written in place (exclusive).
K2 also writes the inclusive total of the last tile to a small "final state" buffer [N][S][d+1].
The forward output does not need it, but it is the natural hand-off for inference caching and it gives a cross-check against the non-causal totals (section 12.3).

Three ways to run it (T = 2M, N = 8, config A with T_blk = 2048: 1,024 tiles per stream, 4,160 floats per record):

| option | parallelism | HBM traffic | deterministic | estimated A100 time |
|---|---|---|---|---|
| one CTA per stream, loop over tiles, threads over record elements | 8 CTAs on 8 SMs | 2·ws | yes | ~2.4 ms (A), ~4.7 ms (B): only 8/108 of the GPU pulls data |
| **element-parallel: one thread per record element, loop over tiles in order (chosen)** | N·S·(d+1) = 33,280 threads (A) / 132,096 (B): 136 / 520 CTAs, every SM takes part | 2·ws | yes, bitwise | 0.18 ms (A), 0.35 ms (B) at full bandwidth |
| tree scan over the tile axis (Blelloch / Hillis-Steele) | log2(1024) = 10 levels | ≥ 4·ws (up-sweep and down-sweep) | yes, if the tree is fixed | slower: doubles traffic to buy parallelism that is not needed |

Why a sequential walk over tiles is fine: the parallelism is already large across the S·(d+1) elements of each record, so the tile axis only has to be walked once, in order.
The loads for later tiles do not depend on the running sum, so unrolling the tile loop by 8 keeps about 8 loads in flight per thread.
That is 33,280 × 8 × 4 B ≈ 1.06 MB in flight, close to the ~1.1 MB that Little's law asks for on A100 (1555 GB/s × ~700 ns latency).
If Nsight shows K2 below ~70% of bandwidth, unroll deeper or switch to float4 loads (4 elements per thread).
Grid: x over record elements in groups of 256 (one element per thread, covering the S·d elements of ws_B and the S elements of ws_A), y over streams.
Fallback if N·S·d is small and T huge (for example N = 1): split the tile axis into G groups, scan each group, then add fixed group offsets; still a fixed order, still deterministic.

## 6. Kernel 3: the output pass

### 6.1 Grid and per-CTA algorithm

- Grid (nTiles, N): blockIdx.x is the tile, so consecutive CTAs are neighboring tiles of one stream; blockIdx.y is the stream.
  256 threads (8 warps) by default.
- Prologue: load this tile's exclusive prefix (A, B) from the workspace into on-chip state; load W into shared memory (fp32 for v2a, bf16 hi and lo for v2b).
- Loop over the sub-chunks of the tile, in order (the last one of the sequence may be partial):
  1. Load the C rows of Q, K, V (bf16, 16-byte vector loads, coalesced).
  2. U_Q = Q Wᵀ and U_K = K Wᵀ; then Φ_Q, Φ_K from P sigmoids and the product tree; rows ≥ T get Φ_K = 0.
  3. G = tril(Φ_Q Φ_Kᵀ), diagonal included.
  4. Den = Φ_Q · A + rowsum(G).
  5. Num = Φ_Q B + G V.
  6. O = Num / Den, rounded to bf16, written to global memory.
  7. B ← B + Φ_Kᵀ V and A ← A + colsum(Φ_K).
- Every output row is written by exactly one CTA, so there are no atomics, and the order of every sum is fixed by the code, so the result is deterministic.

### 6.2 v2a: fp32 CUDA cores

Shared memory, config A, C = 64 (77.8 KiB):

| buffer | type and shape | bytes | why this shape |
|---|---|---|---|
| Xq, Xk staging | bf16 [C][d+2] each | 16,896 | row stride (d+2)/2 = 33 words is odd, so 32 lanes reading one column of 32 rows hit 32 different banks |
| G (aliases Xq, Xk after step 2) | fp32 [C][C+1] | (16,640) | Q and K rows are dead once Φ exists; odd stride again |
| Xv staging | bf16 [C][d+2] | 8,448 | as above |
| W | fp32 [L·P][d] | 4,096 | read as a broadcast (all lanes of a warp read the same address) |
| Φ_Q, Φ_K | fp32 [C][S+1] each | 33,280 | odd stride |
| state | fp32 [S][d+1], A in column d | 16,640 | odd stride; the pad column holds A, so the state is one array |
| Den | fp32 [C] | 256 | |

Global loads are 16-byte vectors; the shared-memory stores are 4-byte (bf16x2) words, because an odd-word row stride is not 16-byte aligned.

Thread mapping, config A, 256 threads:

| step | mapping | notes |
|---|---|---|
| 1 load | 3 tensors × 64 rows × 8 lanes of 16 B = 1,536 vector loads, 6 per thread | each row of 64 bf16 is one 128-byte line |
| 2 projection and φ | thread = (token, table): C·L = 256 pairs per side; a warp is 32 consecutive tokens of one table | W reads are warp broadcasts; X reads are conflict-free (odd stride); per thread P dot products of length d, P tanh and sigmoid, a product tree to R values, one row segment of Φ written |
| 3 G | 16×16 thread grid, 4×4 register micro-tile per thread (rows ty+16a, columns tx+16b), k over S | computed dense and masked on write (j > i gives 0); dense costs 2x the masked FLOPs but keeps v2a simple |
| 4 Den | threads 0..C-1, one row each: Φ_Q[i]·A + ∑_j G[i][j] | odd strides keep it conflict-free |
| 5, 6 Num and output | same 16×16 grid over C×d, k over S (carry, reading the state) then over C (intra, reading G and V), one set of accumulators | this is the single product [Φ_Q ∣ G] · [B ; V] with k = S + C; the results are divided by Den and stored as bf16 directly: 16 lanes write 16 consecutive bf16 = one full 32-byte sector per row |
| 7 update | 16×16 grid over S×d, k over C; the 16 threads with tx = 0 also update column d (A) | each state element has one owner, so the read-modify-write needs no atomics |

Barriers per sub-chunk: after steps 1, 2, 3, 4, 5 (the update must not start while step 5 still reads the state), and 7 (the next load must not overwrite X while it is still read): 6 per 64 tokens, about 0.09 per token, against v1's 4 per token.
Launch bounds: __launch_bounds__(256, 2) for config A.

Config B does not fit at C = 64 (188 KiB, larger than A100's 163 KiB per-CTA limit).
v2a config B uses C = 32 (131.2 KiB: 1 CTA/SM on A100 and H100).
On L40S (99 KiB per CTA) config B additionally needs DV_SPLIT = 2 (95.2 KiB): each CTA owns half of the V, B and O columns and recomputes Φ and G.
The two halves are adjacent in the grid, so the second read of Q and K usually hits L2.
DV_SPLIT is an L40S-only fallback, not part of the default build.

### 6.3 v2b: tensor cores (wmma, bf16 in, fp32 accumulate)

Products moved to tensor cores (all m16n16k16):

| product | shape per sub-chunk | a operand | b operand | notes |
|---|---|---|---|---|
| U = X Wᵀ | (C×d)(d×LPpad) | X row-major | W col-major, stored [LPpad][d] | done twice, with W_hi and W_lo |
| G = Φ_Q Φ_Kᵀ | (C×S)(S×C) | Φ_Q row-major | Φ_K read as col-major | only tiles with ti ≥ tj: 10 of 16 at C = 64 |
| carry Φ_Q B | (C×S)(S×d) | Φ_Q row-major | B_hi (and B_lo) row-major | B_lo only with PRECISE_CARRY |
| intra G V | (C×C)(C×d) | G_bf16 row-major | V row-major | only k-tiles tj ≤ ti |
| update B += Φ_Kᵀ V | (S×C)(C×d) | Φ_K read as col-major (gives Φ_Kᵀ) | V row-major | the fp32 accumulators stay in registers for the whole tile |

CUDA cores keep: tanh, sigmoid and the product tree; the mask, bf16 rounding and row sums of G; Den = Φ_Q · A + rowsum; A += colsum(Φ_K); the division; bf16 conversions.
No transposed copy is ever made: reading a row-major [rows][S] buffer as a col-major operand gives its transpose for free.

Precision choices (numbers in section 7):

- **W split into bf16 hi and lo, required.**
  U = X W_hiᵀ + X W_loᵀ with W_hi = bf16(W) and W_lo = bf16(W - W_hi).
  A single bf16 W gives up to 1.3-1.7% row error at β = 1; hi/lo gives 2e-5.
  X is already exact in bf16, so only W needs the split; W_hi and W_lo are computed once on the host because W never changes.
- **Φ_Q, Φ_K and G rounded to bf16, accepted.**
  This is the same kind of error FlashAttention accepts when it rounds the probabilities P to bf16 before P·V.
  Total error after output rounding is 1.0-1.8x the bf16 rounding floor (max) and 1.3-1.5x (RMS) for β from 1/√d to 2 (section 7.4).
- **PRECISE_CARRY (B split into B_hi + B_lo for the carry), default on.**
  It makes the constant-V test exact again (error 3.6e-4 → 7.6e-7) and removes the carry's share of the error (up to 1.4e-3 at β = 1).
  It costs +17% (config A) and +20% (config B) tensor-core FLOPs, which hide under memory time for config A on every GPU.
  For config B on A100 and H100 the model says it may cost 6-8% (section 9.4), so it is a template flag, measured before a final default is fixed for B.
- **Den uses exactly the rounded values Num uses**: row sums of the bf16 G, and A summed from the bf16 Φ_K.
  Then the weights applied to V are exactly the weights summed in Den, so the output stays a convex combination of values.

Shared memory, C = 64, 8 warps, PRECISE_CARRY on:

| buffer | type and shape | config A bytes | config B bytes | why |
|---|---|---|---|---|
| Xq / Φ_Q, Xk / Φ_K | bf16 [C][max(d,S)+8] each | 18,432 | 34,816 | Φ overwrites X in place; each warp only overwrites the 16 rows it projected itself |
| Xv | bf16 [C][d+8] | 9,216 | 17,408 | |
| W_hi, W_lo | bf16 [LPpad][d] each | 4,096 | 16,384 | LPpad = 16 (A); config B pads each table's P = 5 columns to 8, so LPpad = 32 and no table straddles two 16-column tiles |
| G_bf16 | bf16 [C][C+8] | 9,216 | 9,216 | tiles above the diagonal are never read |
| B_hi, B_lo | bf16 [S][d+8] each | 18,432 | 69,632 | the bf16 copies of the state used by the carry |
| per-warp scratch | fp32 [16][20] × 8 warps | 10,240 | 10,240 | see below |
| A, Den, row partials | fp32 [S], [C], [C/16][C] | 1,536 | 1,792 | |
| **total** | | **71,168 (69.5 KiB)** | **159,488 (155.8 KiB)** | |

Why the paddings and the scratch:

- Leading dimensions d+8, S+8, C+8 (in bf16 elements) are multiples of 8 elements (16 bytes), which wmma requires.
  The +8 shifts each row by 4 banks, so the 8 rows touched by one 128-bit fragment load fall in 32 different banks.
  With a leading dimension of exactly d = 64 (32 words), all rows would start in the same bank: an 8-way conflict.
- Fragment pointers must be 32-byte aligned; this holds automatically because tile offsets are multiples of 16 rows × ldm × 2 bytes and 16 columns × 2 bytes.
- The element layout inside a wmma fragment is opaque (not documented), so element-wise work cannot be done on fragments directly.
  A warp stores a fragment to its own 16×20 fp32 scratch, its lanes do the element-wise work (tanh and product tree, mask and rounding, division, hi/lo split), and write the results.
  No other warp touches that scratch, so __syncwarp is enough.
  The ldm of 20 floats is a multiple of 4 floats as required for fp32 fragments.
- The fp32 master copy of B lives in accumulator fragments in registers for the whole tile: config A has 16 fragments over 8 warps (2 per warp, 16 registers per thread); config B has 64 (8 per warp, 64 registers per thread).
  After each update a warp stores its fragments to scratch and writes the hi and lo bf16 copies for the next sub-chunk's carry.
- In the prologue, B_prev is loaded into the accumulators straight from ws_B with wmma::load_matrix_sync (ldm = d floats, a multiple of 4); this is the reason for the split workspace layout in section 4.

Warp mapping, C = 64, 8 warps:

| step | work units | assignment |
|---|---|---|
| projection | 2 sides × 4 row-tiles | warp w: side w/4, rows 16·(w mod 4); 8 MMAs (A) or 32 MMAs (B) per warp including hi and lo |
| φ | 16 tokens × L tables per warp | lanes read the warp's own scratch; 2 (token, table) pairs per lane; bf16 Φ written into the warp's own rows |
| G | 10 tiles on or below the diagonal | warps 0-7 take tiles 0-7, warps 0-1 also take tiles 8-9; diagonal tiles get the element mask; each tile's row sums go to rowpart[tj][i] |
| Den | 64 rows | warp w takes rows 8w..8w+7; lanes over s, __shfl_xor_sync tree; then ∑ over rowpart[tj][i] for tj = 0..ti in fixed order |
| Num | 16 output tiles (A), 32 (B) | config A: warp w takes column tile w mod 4 and row tiles {w/4, 3 - w/4}, so every warp does 1+4 or 2+3 = 5 intra k-steps; config B: warp w takes column tile w and all 4 row tiles (1+2+3+4 = 10 each) |
| output | per tile: scratch, divide by Den, round, 16-byte stores | 16 rows × 32 bytes per tile, full 32-byte sectors |
| update | 16 state tiles (A), 64 (B) | 2 or 8 per warp, k = C/16 = 4 steps |

Barriers per sub-chunk: after the loads, after Φ, after G, after Den, after Num (the update rewrites B_hi and B_lo, which Num reads), and after the update: 6.

Register estimate: config A ≈ 72 per thread (B accumulators 16, Num accumulators 16, operand fragments 8, addresses), so 3 CTAs/SM by registers and 2 by shared memory on A100.
Config B ≈ 130 per thread (64 + 32 + 8 + addresses), 1 CTA/SM, which matches its shared memory.
Launch bounds: (256, 2) for A, (256, 1) for B.
Config B with 16 warps halves the per-thread accumulators but needs 165.8 KiB, which does not fit A100; it is an H100-only tuning option.

Build steps:

- **v2b-1**: everything above, single-buffered.
- **v2b-2**: cp.async double buffering of the next sub-chunk's Q, K, V (config A: 96.5 KiB, 1 CTA/SM on A100 and 2 on H100; config B: H100 only), plus tuning trials (NWARPS, C = 128, tanh.approx).
  Do this only if Nsight Compute shows memory-latency stalls; with 2-3 CTAs per SM, other CTAs already hide much of the load latency.
- **v2b-3 (optional)**: raw mma.sync PTX with documented fragment layouts, which lets G and the state stay in registers as operands (the FlashAttention-2 technique) and removes the scratch round trips.

### 6.4 Recompute Φ or store it in HBM

Only Φ_K is computed twice (in K1 and in K3).
Φ_Q is computed only in K3 and used once, so storing it can never help.
Storing Φ_K in bf16 means K1 writes 2S bytes per token and K3 reads 2S bytes instead of reading K (2d bytes): a net change of +4S - 2d bytes per token.

| | config A | config B |
|---|---|---|
| extra HBM bytes per token | +128 (+16% of 800) | +256 (+16% of 1,600) |
| K3 work saved per token | K-side projection 4,096 tensor FLOPs + 32 tanh/sigmoid (~1,280 FLOP-equivalents) + 128 FLOPs of product tree | 16,384 tensor FLOPs + ~1,600 + 256 |
| A100 time of the extra bytes | 128 B / 1555 GB/s = 82 ps | 165 ps |
| A100 time of the saved work, at peak | 13 ps (tensor) + 72 ps (fp32) = 85 ps | 53 + 95 = 148 ps |
| extra HBM held at 2M, 8 streams | 2.0 GiB | 4.0 GiB |

Decision: recompute.
The raw rates are close, but v2b is memory-bound on all three GPUs, so extra bytes add time directly, while the saved compute was already hidden under the memory time.
Storing also adds an O(T·S) tensor (A100 config A ceiling falls from 9.9M to 8.0M tokens) and would push bf16-rounded Φ into v2a, whose purpose is an fp32 answer.

### 6.5 Which to build first: v2a

1. v2a's error against the fp64 reference is about 1e-7, four orders below bf16 rounding, so a wrong mask, boundary, scan offset or tail shows up as a clear failure instead of hiding in bf16 noise.
2. All the new structure (tiles, scan, sub-chunk loop, carry, masks, tails, 64-bit indexing, workspace layout) is already in v2a.
   v2b changes only how the five products are computed, so v2b's test becomes "v2b agrees with v2a within bf16 tolerance" on top of the reference test.
3. v2a reuses the non-causal K1 and the non-causal query-pass structure directly.
4. On the dev GPU (L40S: 91.6 TFLOP/s fp32 but only 864 GB/s) v2a config A is predicted to be memory-bound (9.5 ms of fp32 work against a 15.5 ms memory floor), so v2a already gives a real bandwidth number.
5. On A100 and H100, v2a is compute-bound (44.7 and 13.0 ms against 8.6 and 4.0 ms memory floors), so v2b is what the bandwidth claim needs; that is expected and is the reason v2b exists.

## 7. Numerics

### 7.1 Accumulation and rounding

- Every sum is fp32: register accumulators in v2a, tensor-core fp32 accumulators in v2b, fp32 in K2.
- Inputs are bf16, which convert to fp32 exactly.
- The output is rounded once, at the end, with round-to-nearest (__float2bfloat16_rn).
- No --use_fast_math: tanhf, expf and the division stay IEEE-accurate.
  If profiling shows the transcendental functions dominate the CUDA-core time, tanh.approx.f32 is a flagged, measured experiment, not a default.

### 7.2 Boundaries

- K2 is exclusive: tile t starts from the sum of tiles 0..t-1.
- Inside K3 the carry is exclusive (the state before the sub-chunk) and the mask is inclusive (j ≤ i), so query i sees keys 0..i, like the reference.
- T = 1: Num and Den are built from the same k_00, so O_0 = v_0 up to one fp32 rounding of the ratio.

### 7.3 Determinism

- K1 sums each tile in a fixed order, K2 walks tiles in a fixed order, K3 has one owner per output row, and nothing uses atomics.
- A given sequence of mma instructions accumulates in a fixed order on a given GPU.
- Result: bitwise identical output run to run on the same GPU, binary, T_blk and C.
- Changing T_blk or C changes the summation order, so results then differ by rounding (not bitwise); a different GPU architecture may also differ.

### 7.4 Measured error (CPU simulation of the kernel arithmetic, fp64 reference, same bf16 inputs)

Before output rounding, max|err| / max|ref| (config A: T = 2048; config B: T = 1024; C = 64, T_blk = 256):

| variant | A, β = 1/√d | A, β = 1 | B, β = 1/√d | B, β = 1 |
|---|---|---|---|---|
| v2a (all fp32) | 1.0e-7 | 3.5e-7 | 1.9e-7 | 2.7e-7 |
| v2b, single bf16 carry | 1.3e-3 | 2.1e-3 | 9.4e-4 | 1.6e-3 |
| v2b, PRECISE_CARRY | 1.3e-3 | 2.1e-3 | 9.4e-4 | 1.6e-3 |
| for scale: bf16 output rounding alone | 1.8e-3 | 1.7e-3 | 1.45e-3 | 1.45e-3 |

After output rounding, as a multiple of the floor F (max) and F_rms (RMS):

| variant | A β=1/√d | A β=1 | A β=2 | B β=1/√d | B β=1 | B β=2 |
|---|---|---|---|---|---|---|
| v2a max / RMS | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| v2b max / RMS | 1.15 / 1.30 | 1.82 / 1.50 | 1.46 / 1.45 | 1.00 / 1.27 | 1.00 / 1.53 | 1.10 / 1.52 |
| v2b PRECISE_CARRY max / RMS | 1.15 / 1.29 | 1.82 / 1.44 | 1.46 / 1.32 | 1.00 / 1.27 | 1.00 / 1.48 | 1.10 / 1.39 |

Constant V = 0.7 (the output must equal 0.7), before output rounding, max|O - 0.7|:

| variant | A, β = 1/√d | A, β = 1 | B, β = 1/√d | B, β = 1 |
|---|---|---|---|---|
| v2a | 3.7e-7 | 3.7e-7 | 5.8e-7 | 6.1e-7 |
| v2b, single bf16 carry | 3.6e-4 | 1.3e-3 | 2.5e-4 | 1.2e-3 |
| v2b, PRECISE_CARRY | 7.6e-7 | 2.1e-6 | 8.8e-7 | 1.7e-6 |

Where the v2b error comes from (config A, each rounding applied alone, max per-row relative error before output rounding):

| rounding applied alone | β = 1/√d | β = 1 |
|---|---|---|
| Φ_Q, Φ_K to bf16 | 6.1e-4 | 2.3e-3 |
| G to bf16 (before G·V) | 2.5e-3 | 2.4e-3 |
| carry state B to bf16 (single) | 2.8e-4 | 1.4e-3 |
| W to bf16 (single) | 1.8e-4 | 1.3e-2 |
| W as bf16 hi + lo | 3.0e-7 | 1.7e-5 |
| for scale: bf16 output rounding | 2.2e-3 | 2.3e-3 |

Reading of these tables:

- v2a is limited only by the output rounding.
- In v2b the G rounding is the largest term; it is largest in the first rows, where a query has few keys and the rounding errors do not average out.
- W must be split (1.3% error at β = 1 otherwise); the split makes it negligible.
- PRECISE_CARRY barely changes random-V error but fixes constant V, because with it Num and Den see the same weights to fp32 accuracy.
- Removing the remaining G and Φ rounding would need hi/lo splits of the operands themselves (3 MMAs per product), which is fp32 emulation; not planned.

### 7.5 Long sequences

The running sums grow with position, so fp32 drift matters at 2M.
A reduced-width model (S·d = 16 outputs, true fp32 sequential sums in numpy, T = 2^21) gives, as the worst relative error of the prefix state over positions ≥ 10^4:

| accumulation scheme | worst relative error |
|---|---|
| v1: one fp32 running sum over all 2M tokens | 1.1e-4 |
| v2: sums of ≤ 2048 tokens inside a tile, plus a sum of ≤ 1024 tile totals (T_blk = 2048) | 1.0e-5 |
| v2 with T_blk = 4096 | 1.6e-5 |

The two-level scheme is about 10x more accurate than v1 because the error of a sequential fp32 sum grows with its length (about √n·2^-24 typical, n·2^-24 worst), and v2 never runs a sum longer than max(T_blk, nTiles).
Both are far below bf16 output rounding (about 2e-3).
At full width and T = 16,384 the v2a simulation error stays at 4.5e-7, the same as at T = 2048.

### 7.6 Tail cases to test

| case | what it exercises |
|---|---|
| T = 1 | a single partial sub-chunk; O_0 must equal v_0 |
| T = 2, 15, 17, 63 | T < C: one partial sub-chunk, partial G diagonal tile |
| T = 64, 65, 127, 129 | exact C, one past, one short of 2C |
| T = T_blk - 1, T_blk, T_blk + 1 | last tile of length 1; exclusive-scan boundary |
| T = 3·T_blk + 17 | several full tiles and a partial last tile with a partial sub-chunk |
| T = 10,000 | not a multiple of anything |
| N = 1, 3, 8 | stream indexing, odd N |
| config B, N = 8, T = 2^21 | N·T·d = 2^31 elements, one past INT_MAX: all global offsets must be 64-bit |

The last case is not hypothetical: 8 × 2,097,152 × 128 = 2,147,483,648 = 2^31, so a 32-bit offset overflows on the very last element of the 2M benchmark.

## 8. Memory

HBM use at T = 2M, N = 8:

| item | formula | config A | config B |
|---|---|---|---|
| Q, K, V, O (bf16) | 4·N·T·d·2 | 8.00 GiB | 16.0 GiB |
| ws_B + ws_A | N·(T/T_blk)·S·(d+1)·4 | 130 MiB (T_blk 2048) | 258 MiB (T_blk 4096) |
| final state | N·S·(d+1)·4 | 130 KiB | 516 KiB |
| W, W_hi, W_lo | L·P·d·(4+2+2) | 8 KiB | 20 KiB |
| **total** | | **8.13 GiB** | **16.25 GiB** |

There is no O(T·S) tensor anywhere.
Per token over all 8 streams: 4,161 bytes (config A) and 8,321 bytes (config B).

Capacity, taking usable memory as the card's memory minus 1.5 GiB for the CUDA context and allocator:

| GPU | config A at 2M uses | config A ceiling | ceiling / 131,072 | config B at 2M uses | config B ceiling |
|---|---|---|---|---|---|
| L40S-48GB | 17% | 12.0M tokens | 92x | 35% | 6.0M |
| A100-40GB | 21% | 9.9M tokens | 76x | 42% | 5.0M |
| H100-80GB | 10% | 20.3M tokens | 155x | 21% | 10.1M |

So the 2M target fits on A100-40GB with about 4x headroom (config A), and H100-80GB tops out near 20M tokens.
These ceilings are for the forward interface (Q, K, V, O and the workspace resident, nothing else); an earlier estimate of ~9M for A100 used a more conservative usable-memory figure and is the same result.
For contrast, the naive one-state-per-chunk layout of section 3.2 would need 1-16 GiB of workspace for the same run.

## 9. Roofline and the choice of C

### 9.1 FLOPs per token per stream, by stage (C = 64)

Same conventions as docs/noncausal_design.md section 1: 1 MAC = 2 FLOPs, M = 1, a token is one query row plus one key/value row, tanh and sigmoid counted separately.

| stage | formula | config A | config B |
|---|---|---|---|
| projection, K side in K1 | 2·L·P·d | 2,048 | 5,120 |
| projection, Q and K in K3 | 4·L·P·d | 4,096 | 10,240 |
| φ product trees, 3 sides | 6·L·R | 384 | 768 |
| tile sum in K1: Φ_Kᵀ[V∣1] | 2·S·(d+1) | 8,320 | 33,024 |
| carry in K3: Φ_Q [B∣A] | 2·S·(d+1) | 8,320 | 33,024 |
| state update in K3: Φ_Kᵀ[V∣1] | 2·S·(d+1) | 8,320 | 33,024 |
| intra G = Φ_Q Φ_Kᵀ, dense | 2·C·S | 8,192 | 16,384 |
| intra G [V∣1], dense | 2·C·(d+1) | 8,320 | 16,512 |
| divide | d | 64 | 128 |
| **total, dense intra (the v2a count)** | | **48,064** | **148,224** |
| intra with 16×16 causal tile skipping (v2b) | f(C)·(2CS + 2C(d+1)), f(64) = 10/16 | 10,320 (not 16,512) | 20,560 (not 32,896) |
| tanh + sigmoid, 3 sides | 3·2·L·P operations | 96 (~3,840 FLOP-eq) | 120 (~4,800 FLOP-eq) |

The FLOP-equivalent for the transcendental functions assumes about 40 fp32-pipe instructions per accurate tanh + sigmoid pair; this is an estimate to be checked in Nsight Compute.

Bytes per token per stream: 12d + 16·S·(d+1)/T_blk = 768 + 32.5 = 800.5 (A) and 1,536 + 64.5 = 1,600.5 (B).
The single-pass minimum (8d) is 512 (A) and 1,024 (B).

What runs where in v2b, per token (PRECISE_CARRY on, C = 64):

| | formula | config A | config B |
|---|---|---|---|
| tensor-core FLOPs | 3·(2·2·LPpad·d) + 4·(2·S·d) + f(C)·(2CS + 2Cd) | 55,296 | 200,704 |
| CUDA-core FLOP-eq | 6LR + 80·3LP + 4S + 2C + d | 4,672 | 6,336 |

(The projection term counts hi and lo and the padded LPpad; the 4 state products are K1 tile sum, carry hi, carry lo, update.)

### 9.2 Arithmetic intensity as a function of C

| C | f(C) | v2a FLOP/B, A | v2b tensor FLOP/B, A | v2a FLOP/B, B | v2b tensor FLOP/B, B |
|---|---|---|---|---|---|
| 32 | 0.750 | 54.5 | 64.0 | 85.3 | 120.3 |
| 64 | 0.625 | 64.8 | 69.1 | 95.6 | 125.4 |
| 128 | 0.563 | 85.5 | 79.3 | 116.2 | 135.6 |
| 256 | 0.531 | 126.7 | 99.8 | 157.3 | 156.1 |

Ridge points (FLOP per byte where compute time equals memory time):

| GPU | fp32 CUDA cores | bf16 tensor cores |
|---|---|---|
| L40S (864 GB/s, 91.6 TFLOP/s fp32, 362 bf16) | 106 | 419 |
| A100-40GB (1555 GB/s, 19.5, 312) | 12.5 | 201 |
| H100-SXM (3350 GB/s, 66.9, ~660 via mma.sync) | 20 | 197 |

H100's tensor figure is about 2/3 of its 989 TFLOP/s wgmma peak, because wmma compiles to the older mma.sync instructions, which cannot reach the full Hopper rate.
Readings: v2a is memory-bound on L40S for C ≤ 128 and compute-bound on A100/H100 for every C; v2b sits below the tensor-core ridge on all three GPUs for every C, by 1.3-6x.

### 9.3 Time versus C (T = 2M, N = 8)

| C | v2a floor L40S / A100 / H100 (ms), config A | v2b tensor time A100 (ms), A | v2b tensor time A100 (ms), B |
|---|---|---|---|
| 32 | 15.5 / 37.6 / 10.9 | 2.75 | 10.35 |
| 64 | 15.5 / 44.7 / 13.0 | 2.97 | 10.79 |
| 128 | 15.5 / 58.9 / 17.2 | 3.41 | 11.67 |
| 256 | 18.6 / 87.3 / 25.4 | 4.29 | 13.44 |

For comparison, the A100 memory floor is 8.64 ms (A) and 17.27 ms (B).

### 9.4 Predicted floors at T = 2M, N = 8 (C = 64)

Floor = max(memory time, tensor time, CUDA-core time) at peak rates.
"Expected" uses 85% of peak bandwidth, 50% of the mma peak (small 16×16 tiles fed from shared memory) and 60% of the fp32 peak; these efficiencies are assumptions to be replaced by measurements.

| kernel | GPU | memory | tensor | CUDA cores | floor | expected |
|---|---|---|---|---|---|---|
| v2b, config A | L40S | 15.54 | 2.56 | 0.86 | **15.5 ms** | 18.3 ms |
| v2b, config A | A100 | 8.64 | 2.97 | 4.02 | **8.6 ms** | 10.2 ms |
| v2b, config A | H100 | 4.01 | 1.41 | 1.17 | **4.0 ms** | 4.7 ms |
| v2b, config B | L40S (C = 32) | 31.08 | 8.92 | 1.15 | **31.1 ms** | 36.6 ms |
| v2b, config B | A100 | 17.27 | 10.79 | 5.45 | **17.3 ms** | 21.6 ms (tensor-limited; 20.3 ms without PRECISE_CARRY) |
| v2b, config B | H100 | 8.02 | 5.10 | 1.59 | **8.0 ms** | 10.2 ms (tensor-limited; 9.4 ms without PRECISE_CARRY) |
| v2a, config A | L40S / A100 / H100 | | | 9.5 / 44.7 / 13.0 | 15.5 / 44.7 / 13.0 ms | 18.3 / 74.4 / 21.7 ms |
| v2a, config B (C = 32) | L40S / A100 / H100 | | | 25.0 / 117.5 / 34.3 | 31.1 / 117.5 / 34.3 ms | 36.6 / 196 / 57 ms |
| single-pass minimum (8d bytes) | L40S / A100 / H100 | | | | A: 9.9 / 5.5 / 2.6 ms, B: 19.9 / 11.1 / 5.1 ms | |

Config A (the layer that hit the 131,072-token wall) is memory-bound with room to spare on every GPU, so the success metric is % of HBM bandwidth.
Config B on A100 and H100 is close to balanced once realistic tensor-core efficiency is included; for it, report both % of HBM and tensor-pipe utilization.
The CUDA-core line for config A on A100 (4.0 ms at peak, ~6.7 ms at 60%) is mostly the transcendental estimate; it is the first thing to check in Nsight Compute.

### 9.5 Choice of C = 64

- **Tensor time grows with C** (table 9.3), but slowly for config B because the state products and the projection, not the intra term, dominate its tensor work.
  C = 64 keeps config A's tensor time near 1/3 of the memory time on A100, so a realistic 40-50% wmma efficiency still hides under memory.
- **Shared memory grows with C** (tiles ∝ C, G ∝ C²): config A v2b goes from 69.5 KiB at C = 64 to 124.8 KiB at C = 128, which drops A100 from 2 CTAs/SM to 1.
- **C = 32 buys little**: 7% less tensor work for config A, but the causal tile skip gets worse (3 of 4 G tiles instead of 10 of 16), the per-sub-chunk fixed costs (6 barriers, rewriting B_hi/B_lo, S·d elements each) double per token, and 3 G tiles cannot keep 8 warps busy.
- **v2a**: on L40S C does not matter (memory-bound up to C = 128); on A100 C = 32 would be 16% faster, but v2a is not the A100 performance kernel, and one C for both builds keeps layouts and tests shared.
  Config B uses C = 32 in v2a for shared-memory reasons only.

## 10. Parallelism and occupancy

### 10.1 Grids at T = 2M, N = 8

| kernel | grid | CTAs | resident at once on A100 |
|---|---|---|---|
| v1 (race_fused_fwd.cu) | N | 8 | 8, on 8 of 108 SMs (7%) |
| v2 K1 (non-causal K1, one CTA per tile and table) | (1024, 4, 8) | 32,768 | up to 8 per SM (2048-thread limit), 864 |
| v2 K2 | (⌈S·(d+1)/256⌉, 8) | 136 (A), 520 (B) | all |
| v2 K3, config A (T_blk 2048) | (1024, 8) | 8,192 | 2 per SM, 216 |
| v2 K3, config B (T_blk 4096) | (512, 8) | 4,096 | 1 per SM, 108 |

K3 runs 8,192 / 216 = 38 waves (config A) and 4,096 / 108 = 38 waves (config B), so the partial last wave costs at most about 1/38 ≈ 3%.
v1's structure, by contrast, uses 7% of the SMs and serializes each token behind 4 barriers.
A rough v1 estimate: about 3,000 cycles per token (step 5's 64-iteration loop with a division each, the single-warp softmax, 4 barriers, and an unprefetched global load per token), so about 2.2 µs per token at 1.41 GHz and about 4.6 s for T = 2M.
This is an estimate, not a measurement.
Against the expected v2b time of ~10 ms on A100, that is a speedup of several hundred times.

### 10.2 Shared memory, registers and CTAs per SM

Per-SM limits: L40S (sm_89) 100 KiB shared, 99 KiB per CTA, 1,536 threads; A100 (sm_80) 164 KiB, 163 KiB, 2,048 threads; H100 (sm_90) 228 KiB, 227 KiB, 2,048 threads; 64K registers on all three; 1 KiB of shared memory is reserved per resident CTA.

| K3 build | shared memory per CTA | L40S CTAs/SM | A100 CTAs/SM | H100 CTAs/SM | registers per thread (est.) |
|---|---|---|---|---|---|
| v2a config A, C = 64 | 77.8 KiB | 1 | 2 | 2 | ~64 |
| v2a config B, C = 32 | 131.2 KiB | does not fit | 1 | 1 | ~100 |
| v2a config B, C = 32, DV_SPLIT = 2 | 95.2 KiB | 1 | 1 | 2 | ~70 |
| v2b config A, C = 64 | 69.5 KiB | 1 | 2 | 3 | ~72 |
| v2b config A, C = 64, double-buffered | 96.5 KiB | 1 | 1 | 2 | ~72 |
| v2b config B, C = 64 | 155.8 KiB | does not fit | 1 | 1 | ~130 |
| v2b config B, C = 64, no PRECISE_CARRY | 121.8 KiB | does not fit | 1 | 1 | ~130 |
| v2b config B, C = 32, no PRECISE_CARRY (L40S build) | 88.9 KiB | 1 | 1 | 2 | ~100 |

Every K3 build is above the 48 KiB static limit, so each instantiation needs the dynamic shared memory opt-in once before its first launch:
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), plus cudaFuncAttributePreferredSharedMemoryCarveout = 100 so the L1/shared split gives the maximum shared memory.
The launcher reads the per-device limits from cudaGetDeviceProperties and refuses a configuration that does not fit, instead of failing at launch.
K1 (the non-causal design, ~29 KiB worst case) and K2 (no shared memory) stay under 48 KiB.
With 2 CTAs of 256 threads, A100 runs 16 warps per SM (25% occupancy); that is enough here because each warp has independent work (4×4 register micro-tiles in v2a, several MMAs in flight in v2b) and other CTAs cover the load phases.
On L40S, config B is the tight case: it runs only as v2a with DV_SPLIT = 2 or v2b with C = 32 and without PRECISE_CARRY (88.9 KiB).

## 11. Backward outline (design only)

### 11.1 Math

Let g_i = ∂ℒ/∂O_i, and write O_i = N_i / D_i (N_i the numerator vector, D_i the denominator).

- ∂ℒ/∂N_i = g_i / D_i and ∂ℒ/∂D_i = -⟨g_i, O_i⟩ / D_i.
- Define ĝ_i = [g_i / D_i, -⟨g_i, O_i⟩ / D_i] ∈ R^{d+1} and v̂_j = [v_j, 1] ∈ R^{d+1}.
  The extra column carries the denominator's gradient, the same trick as [V∣1] in the forward.
- ∂ℒ/∂k_ij = ⟨ĝ_i, v̂_j⟩ for j ≤ i.
- dΦ_Q,i = ∑_{j≤i} ⟨ĝ_i, v̂_j⟩ Φ_K,j = Ĥ_i ĝ_i, where Ĥ_i = [B_i ∣ A_i] is exactly the forward prefix state.
- dΦ_K,j = ∑_{i≥j} ⟨ĝ_i, v̂_j⟩ Φ_Q,i = Z_j v̂_j, where Z_j = ∑_{i≥j} Φ_Q,i ⊗ ĝ_i ∈ R^{S×(d+1)} is a suffix state (summed from the end).
- dv_j = ∑_{i≥j} k_ij g_i / D_i = Z_j[:, :d]ᵀ Φ_K,j.

Chunk form for a sub-chunk (E = tril(Ĝ V̂ᵀ), so E_ij = ⟨ĝ_i, v̂_j⟩ for j ≤ i; G = tril(Φ_Q Φ_Kᵀ) recomputed):

- dΦ_Q = Ĝ Ĥ_prevᵀ + E Φ_K (prefix state before the sub-chunk).
- dΦ_K = V̂ Z_nextᵀ + Eᵀ Φ_Q (suffix state after the sub-chunk).
- dV = Φ_K Z_next[:, :d] + Gᵀ (Ĝ[:, :d]).

Every term is a small matmul of the same shapes as the forward, so the v2b machinery carries over.

Through φ (product form, verified in the non-causal design): ∂φ_r/∂u_t = β·φ_r·(s_r,t - (2p_t - 1)), where s_r,t = ±1 is the corner sign and 2p_t - 1 is its expected value.
Per table this costs O(R·P): du_t = β·(∑_{r: s=+1} dφ_r φ_r - ∑_{r: s=-1} dφ_r φ_r - (2p_t - 1)·∑_r dφ_r φ_r).
Then u = tanh(y) gives dy = (1 - u²)·du, and y = W x gives dx = Wᵀ dy.
dβ = ∑ over all tokens, both sides and all tables of ∑_r dφ_r · φ_r · ∑_t u_t (s_r,t - (2p_t - 1)): one scalar.
No dW: W is a fixed buffer.

### 11.2 Kernels

| kernel | mirrors | work |
|---|---|---|
| B1 | K1 | per tile: Z tile totals ∑ Φ_Q,i ⊗ ĝ_i (reads Q, dO, O, D) |
| B2 | K2 | exclusive scan over tiles from the end (suffix): Z_next per tile |
| B3a | K3, sub-chunks in forward order | dΦ_Q → dQ, using the forward's saved tile prefixes Ĥ |
| B3b | K3, sub-chunks in reverse order | dΦ_K → dK and dV, using Z_next and a running in-tile suffix state |
| B4 | non-causal K2 (tree reduce) | dβ from per-CTA partials, fixed tree |

Why two B3 kernels and not one: dΦ_Q needs prefix states (walk forward), dΦ_K and dV need suffix states (walk backward).
One backward walk could recover prefixes by subtracting (B ← B - Φ_Kᵀ V, which the authors' unused backward kernel does), but subtracting from large running sums loses precision, and it is unnecessary because the tile prefixes are saved.

### 11.3 Saved versus recomputed

- Saved from the forward: D (fp32 [N][T], 64 MiB at 2M × 8), O (the bf16 output, which autograd keeps anyway), and the workspace after K2 (tile prefixes, 130 / 258 MiB).
- Recomputed: Φ_Q, Φ_K, G, all in-tile running states.
- Nothing O(T·S) is stored.
- If memory is tight, the workspace can be dropped and rebuilt by re-running K1 + K2 in the backward (one extra read of K and V, ~0.2 ms at config A on A100); keeping it costs 65 bytes per token (1.5% of the total).

### 11.4 Determinism

- Every dQ, dK, dV row is owned by exactly one CTA (the one whose tile holds that row), so no atomics.
- The tile scans run in a fixed order; dβ uses a fixed tree.
- Result: bitwise reproducible gradients, which the non-causal design requires for training.

## 12. Test and benchmark plan

### 12.1 References (under src/)

- alg1_causal_reference: fp64, softmax-over-corners φ from BatchedACE's buffers, β as a parameter, dense masked T×T for T ≤ 4096.
- alg1_causal_chunked: the section 3.1 form in any dtype, runs on GPU in fp64 for long T (matches the dense form to 2.5e-16).
- perbucket_causal_reference: the bridge to race_baseline (≤ 1e-12 in fp64).
- The CPU checks behind this document that live in the repository (Appendix B) are in src/causal_v2/reference.py and src/causal_v2/tests/.

### 12.2 Correctness matrix (kernel against alg1_causal_reference)

| axis | values |
|---|---|
| config | A, B, and one small config (d = 64, P = 2, L = 2, S = 8, padded to 16 feature columns with zero features, since wmma needs S and d to be multiples of 16) |
| T | 1, 2, 15, 16, 17, 63, 64, 65, 127, 129, T_blk - 1, T_blk, T_blk + 1, 3·T_blk + 17, 10,000 |
| T_blk (forced through a test hook) | C, 2C, 2048 |
| C | 32, 64 |
| N | 1, 3, 8 |
| β | 1/√d, 0.5, 1, 2 |
| tolerance | section 2.4: v2a max ≤ 1.1·F, RMS ≤ 1.05·F_rms; v2b max ≤ 2.5·F, RMS ≤ 2·F_rms |

### 12.3 Structural tests

| test | catches |
|---|---|
| two runs, torch.equal | any hidden nondeterminism (a stray atomic, a race) |
| constant V = c (bf16-representable): v2a and v2b with PRECISE_CARRY must return exactly c after rounding; v2b without it within 1 ulp | Num/Den inconsistency, wrong A |
| v2a outputs for different T_blk and C agree within 1 bf16 ulp | scan and carry bugs that happen to cancel at one setting |
| fill every other stream with NaN; the tested stream must stay finite and correct | cross-stream indexing, reads past T |
| K2 alone against torch.cumsum in fp64 on a random workspace | scan offsets, exclusive vs inclusive |
| causal final state equals the non-causal K1+K2 totals for the same K, V | cross-check of the two code paths |
| causal O at i = T-1 equals the non-causal O at i = T-1 (the last query sees every key) | cross-check of the query side |
| compute-sanitizer memcheck, racecheck, synccheck, initcheck at small T | out-of-bounds, shared-memory races from the buffer aliasing, barrier misuse |
| config B, N = 8, T = 2^21 (2^31 elements): last stream's last 4096 outputs against the fp64 chunked reference of that stream | 32-bit offset overflow |
| -Xptxas -v on every instantiation: registers, spills (must be 0), shared memory | register pressure regressions |

### 12.4 Benchmarks

- Shapes: config A first (the layer that hit 131,072), N = 8, T = 2^16 ... 2^21, then doubling to the ceiling (A100: 2^23 = 8.4M; H100: 2^24 = 16.8M); config B the same ramp.
- Timing: CUDA events, 5 warm-up runs, median of 20; K1, K2, K3 timed separately (Nsight Systems).
- Report per T:

| column | definition |
|---|---|
| ms | total, and K1 / K2 / K3 |
| tokens/s | N·T / time |
| GB/s (model) | (N·T·12d + state traffic) / time |
| % HBM peak | GB/s (model) / peak (864 / 1555 / 3350) |
| GB/s (minimum bytes) | N·T·8d / time, the single-pass view |
| peak memory | torch.cuda.max_memory_allocated |
| error | max error against the fp64 chunked reference on one sampled stream, as a multiple of F |
| same-GPU multiplier | largest T that ran / 131,072 on A100-40GB (the measured reference wall); on H100 measure the reference's wall directly instead of using the 262,144 extrapolation |

- Baselines in the same table: the reference (until it runs out of memory), the Phase 2 chunked PyTorch forward, v1 (up to 262,144 only, since it takes seconds), v2a, v2b.
- Nsight Compute per kernel: dram__throughput.avg.pct_of_peak_sustained_elapsed, sm__pipe_tensor_op_hmma_cycles_active, achieved occupancy, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld and _st, registers, spills, top warp stall reasons.
- Success criterion: v2b config A at ≥ 80% of HBM peak on A100 and H100 for T ≥ 2^20 (the model expects ~85%).
  Any reported performance number comes only from these measurements.

## 13. CUDA concepts the implementation uses

Execution model

- grid, block, warp, lane; 2-D grids (tile × stream) and a third dimension for DV_SPLIT
- SM residency, waves and the partial last wave
- occupancy: limits from threads, registers, shared memory and CTAs per SM; __launch_bounds__
- warp divergence; keeping barriers in uniform control flow
- kernel boundaries as the grid-wide synchronization between K1, K2 and K3 (same stream, launch order)

Memory

- global memory coalescing, 32-byte sectors and 128-byte lines
- vectorized 16-byte loads and stores (uint4), bf16x2 (__nv_bfloat162) words, alignment rules
- 64-bit index arithmetic for tensors above 2^31 elements
- L2 cache reuse (DV_SPLIT neighbors), __restrict__ and const pointers
- static versus dynamic shared memory (extern __shared__), the 48 KiB default, the cudaFuncAttributeMaxDynamicSharedMemorySize opt-in, the shared-memory carveout attribute, per-architecture limits (sm_80, sm_89, sm_90), the 1 KiB per-CTA reservation
- bank conflicts and padding (odd word strides for CUDA-core access, +8-element strides for wmma)
- reusing (aliasing) shared-memory regions across phases of a loop
- registers, register pressure and spills (-Xptxas -v)
- asynchronous copies (cp.async, __pipeline_memcpy_async or cuda::memcpy_async) and double buffering (v2b-2)
- workspace tensors allocated through the PyTorch caching allocator

Synchronization

- __syncthreads placement and cost; __syncwarp
- warp-collective semantics of the wmma calls
- no atomics: deterministic reductions by fixed ownership and fixed order

Warp-level primitives

- __shfl_xor_sync and __shfl_down_sync reductions, member masks

Numerics

- bf16 format, conversions (__float2bfloat16_rn, __bfloat162float), round-to-nearest
- fp32 accumulation; growth of rounding error with the length of a sequential sum
- hi/lo splitting of an fp32 value into two bf16 values
- TF32 as an alternative input precision (considered, not used)
- accurate versus approximate functions (tanhf, expf versus tanh.approx.f32, __expf), why no --use_fast_math, FMA contraction
- run-to-run versus cross-architecture reproducibility

Tensor cores

- the wmma API: fragment<matrix_a / matrix_b / accumulator, 16, 16, 16, __nv_bfloat16>, row_major and col_major, load_matrix_sync, mma_sync, store_matrix_sync, fill_fragment
- ldm and 32-byte pointer alignment rules
- opaque fragment layouts and the per-warp scratch round trip
- keeping accumulator fragments in registers across loop iterations
- getting a transpose for free by reading a buffer as col_major
- mma.sync versus Hopper's wgmma, and why wmma reaches about 2/3 of the H100 peak
- raw mma.sync PTX with documented layouts (optional v2b-3)

Algorithms

- tiling and register micro-tiling (4×4 outer products)
- causal (triangular) masking and skipping tiles above the diagonal
- balancing triangular work across warps (pairing row tile ti with 3 - ti)
- reduce-then-scan; exclusive versus inclusive scan
- sequential, Hillis-Steele, Blelloch and decoupled look-back scans, and their determinism
- chunked linear attention; the two-level (tile / sub-chunk) state granularity
- recompute versus store
- roofline model, arithmetic intensity, ridge points, Little's law for bytes in flight

Host side and tooling

- C++ templates for compile-time configuration (d, P, L, C, NWARPS, PRECISE_CARRY, EMIT_OUTPUT, DV_SPLIT)
- torch.utils.cpp_extension (load_inline), TORCH_CHECK, C10_CUDA_KERNEL_LAUNCH_CHECK, at::cuda::getCurrentCUDAStream
- cudaGetDeviceProperties for SM count and shared-memory limits
- -gencode for sm_80, sm_89, sm_90
- CUDA events for timing; Nsight Systems and Nsight Compute; compute-sanitizer (memcheck, racecheck, synccheck, initcheck)

## 14. Decisions

1. Ground truth: Algorithm 1 made causal, with global normalization (section 2.3).
   The per-bucket form stays a possible later v2c for the authors' checkpoints.
2. Workspace layout: split ws_B [..][S][d] and ws_A [..][S] for float4 and wmma alignment, instead of [..][R][d+1] (section 4).
   If the non-causal K1 keeps [R][d+1], K2 reads that layout through its stride.
3. β is a run-time argument in all kernels, causal and non-causal, because the paper trains it.
4. Config B on the L40S dev GPU runs only in reduced forms (v2a with DV_SPLIT = 2; v2b with C = 32 and no PRECISE_CARRY); its full form is validated on A100 and H100 only.
5. Decomposition: C = 64 sub-chunks inside state tiles of T_blk = 2048 (config A) or 4096 (config B) tokens (sections 3.3 and 9.5).
6. Φ is recomputed in K3, never stored in HBM (section 6.4).
7. Build order: v2a (fp32 CUDA cores) first, then v2b (wmma, bf16 in, fp32 accumulate) (section 6.5).
8. v2b precision: FlashAttention-class bf16 rounding of Φ and G is accepted; PRECISE_CARRY is on by default for config A and measured for config B before its default is fixed (section 6.3).
9. The transcendental cost (~40 instructions per tanh + sigmoid pair) is an estimate.
   If A100 turns out CUDA-core-bound, tanh.approx.f32 is a flagged, measured experiment, not a default (section 7.1).
10. H100: wmma (mma.sync) is enough because the kernel stays memory-bound; there is no Hopper-specific (wgmma/TMA) path.
11. The v1 timing estimate (~4.6 s at 2M) is unverified until v1 runs on a GPU.
12. v2a K1 reuses the non-causal per-table K1 rather than a new per-(stream, tile) all-table K1 (section 4).

## Appendix A. Formulas and GPU figures

Per token per stream (d, P, L, R = 2^P, S = L·R, f(C) = (n(n+1)/2)/n² with n = C/16):

- bytes = 12d + 16·S·(d+1)/T_blk
- v2a FLOPs = 6LPd + 6LR + 6S(d+1) + 2CS + 2C(d+1) + d, plus ~80·3LP FLOP-eq for tanh and sigmoid
- v2b tensor FLOPs = 12·LPpad·d + 8Sd + f(C)·(2CS + 2Cd) with PRECISE_CARRY (6Sd instead of 8Sd without)
- v2b CUDA-core FLOP-eq = 6LR + 240LP + 4S + 2C + d
- time = N·T × (per-token amount) / (rate); N·T = 16,777,216 at the benchmark shape
- workspace = N·(T/T_blk)·S·(d+1)·4 bytes; state traffic = 4 × workspace
- capacity: tokens = (memory - 1.5 GiB) / (4·N·d·2 + N·S·(d+1)·4/T_blk)

| GPU | arch | SMs | HBM GB/s | fp32 TFLOP/s | bf16 tensor TFLOP/s used | shared memory per SM / per CTA | threads per SM | memory |
|---|---|---|---|---|---|---|---|---|
| L40S | sm_89 | 142 | 864 | 91.6 | 362 (dense) | 100 / 99 KiB | 1,536 | 48 GB |
| A100-SXM4-40GB | sm_80 | 108 | 1,555 | 19.5 | 312 (dense) | 164 / 163 KiB | 2,048 | 40 GB |
| H100-SXM5-80GB | sm_90 | 132 | 3,350 | 66.9 | ~660 via mma.sync (989 wgmma peak) | 228 / 227 KiB | 2,048 | 80 GB |

The H100 mma.sync figure is an assumption; the others are datasheet values.

## Appendix B. CPU experiments behind the numbers

All ran on CPU, each in a few seconds and well under 1 GB.
The fp64 equivalence checks reproduce from src/causal_v2/reference.py and the tests in src/causal_v2/tests/test_reference.py, which assert them at a tolerance (≤ 1e-12 in fp64) rather than printing the exact values below.
The shared-memory, sub-chunk and tile-length choices are checked by src/causal_v2/tests/test_kernel_layout.py.
The β-gap tables, the v2a / v2b bf16 simulation, the error attribution and the long-sequence drift model are not in the repository; their numbers are recorded here only.

| experiment | result used in | in the repository |
|---|---|---|
| Bernoulli per-bucket reference vs race_baseline, fp64 | 3.6e-15 (sections 1, 2.4) | test_bridge_perbucket_reference_matches_race_baseline (race_causal_perbucket_reference vs BatchedACE) |
| chunked vs dense Algorithm 1, fp64, configs A and B, tile 256 and 320 | 2.5e-16, 3.9e-16 (3.1) | test_prefix_reference_matches_dense_masked_form, test_chunked_emulation_matches_reference, test_chunked_emulation_tile_and_chunk_sizes (smaller shapes) |
| chunked vs cumsum per-bucket, fp64 | 1.8e-15, 2.2e-15 (2.2) | no |
| Algorithm 1 vs race_baseline / L by β and by position | 2.3 | no |
| v2a / v2b arithmetic simulation (bf16 rounding points as in section 6.3), β ∈ {1/√d, 1, 2} | 7.4 | no (test_fp32_emulation_is_within_fp32_accuracy covers the fp32 data flow at a 1e-5 bound) |
| error attribution: Φ, G, carry, W rounding applied one at a time | 7.4, 6.3 | no |
| fp32 prefix drift at T = 2^21, reduced width, numpy float32 sequential sums | 7.5 | no |
| roofline, workspace, capacity and shared-memory tables | 3, 8, 9, 10 | Appendix A formulas; test_kernel_layout.py checks the v2a shared-memory totals, C and default T_blk |

## What the implementation changed

`src/causal_v2/` implements the v2a build.
It follows this design with these differences, each for a concrete reason (from the header of `src/causal_v2/kernels/race_causal_fwd.cu` and from `src/causal_v2/README.md`):

- **Den is computed in the G step** instead of a separate step (section 6.2).
  The 16 threads that hold one row of G add their G entries and a strided share of Φ_Q · A, then a 16-lane shuffle reduction finishes the row.
  This drops one barrier (5 per sub-chunk instead of 6) and the 64-thread serial loop.
- **C is 64 only when the Num micro-tile stays at 16 accumulators** (C · d ≤ 4096) and the L = 4 footprint fits the smallest opt-in shared-memory limit of the targets (99 KiB on sm_89); otherwise C = 32.
  So config A (d = 64, P = 4) uses C = 64 and config B uses C = 32, as section 6.2 planned for v2a; d = 64 with P = 5 also uses C = 32.
- **The state update walks the S rows in passes** of 64 (d = 64) or 32 (d = 128) rows, so its register tile stays at 16 accumulators for any L.
  A is updated by threads 0..S-1 in the same phase, instead of by the 16 threads with tx = 0.
- **K3 uses `__launch_bounds__(256, 3)`** (at most 80 registers per thread) instead of (256, 2).
  A 64-register cap would add 4-32 bytes of spills in the C = 64 instantiations and would not add a resident CTA, because at the benchmarked shapes shared memory already limits K3 to 1 CTA per SM on L40S and 2 on A100 and H100.
- **The workspace keeps the non-causal layout**, fp32 [tiles, B·H, L, R·(d+1)], where in each slice B[r][c] is at r·d + c and A[r] at R·d + r, instead of the split ws_B and ws_A of section 4.
  The size is the same: at T = 2²¹, B·H = 8, config A with T_blk = 2048 it is 1024 · 8 · 4 · 1040 floats = 130 MiB.
- **K1 is not the non-causal K1 binary unchanged** (section 4).
  It calls the non-causal `stage_tokens` and `accumulate_stage` from `src/noncausal/kernels/race_internal.cuh`, and only the outer tile loop is local, because the non-causal kernel fixes its tile at 2048 tokens while the causal tile length is chosen at run time.
- **K2 runs one thread per float of a tile's slice for all streams at once**, instead of a grid with y over streams (section 5).
  Each thread walks the tiles in index order with 8 loads in flight, so the scan is still bitwise reproducible.
- **T_blk is chosen by a rule rather than the clamp formula of section 3.3**: the smallest power of two ≥ C whose state traffic 16·S·(d+1)/T_blk bytes per token is at most 5% of the 12·d bytes of Q/K/V/O traffic (2048 for config A, 4096 for config B), shrunk for short sequences to about 4 waves of K3 CTAs, never below C.
- **K1 and K3 compute Φ_K with different fp32 operation orders** (a warp butterfly in K1, one thread's serial dot product in K3).
  That is harmless: each contribution enters A and B with the same Φ value, so Num and Den stay consistent and a constant V still comes back exactly.
- **DV_SPLIT is not implemented.**
  d = 128, P = 5 with L ≥ 3 does not fit L40S shared memory, so the binding refuses it there with a message.
- **The references have different names and shapes than section 12.1.**
  `src/causal_v2/reference.py` has `race_causal_reference` (the prefix-sum definition via a blocked cumsum, any T), `race_causal_dense_reference` (the masked T×T form, tiny T only), `race_causal_perbucket_reference` (the bridge to race_baseline) and `race_causal_chunked_emulation` (the tile and sub-chunk decomposition, returning the prefix states).
  They take the planes as W [L, P, d] and compute φ with the direct length-R softmax, so a kernel test against them still checks the Bernoulli factorization.
- **The GPU tolerance adds tiny absolute allowances** to the section 2.4 bounds (max ≤ 1.1·F, RMS ≤ 1.05·F_rms); they only matter when F is 0, for example at T = 1 or with a constant V.
  The derivation is in `src/causal_v2/tests/causal_numerics.py`.
- **Only v2a is built.**
  v2b (section 6.3) is not started, and the backward (section 11) is not implemented, so the causal binding is forward only.
