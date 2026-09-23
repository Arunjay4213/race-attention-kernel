# Causal RACE Attention forward, chunk-parallel v2a and v2b (CUDA)

Causal forward pass of RACE Attention (arXiv 2510.04008, Algorithm 1 made causal, global normalization) as three bf16 CUDA kernels that are parallel over the sequence, with an fp64-capable PyTorch reference, a CPU model of the kernel decomposition, and tests.
The design is `docs/causal_v2_design.md`; this directory implements both of its builds: v2a (fp32 CUDA cores, the default) and v2b (bf16 tensor cores through `nvcuda::wmma`, fp32 accumulation).

Status: validated on an NVIDIA A10G (sm_86, 99 KB opt-in shared memory) with torch 2.10 and CUDA 12.8.
All tests in `tests/` pass there for both variants (1681 passed, 38 skipped: the 18 d = 128, P = 5, L = 4 cases of each variant need 131-132 KiB of shared memory, and the two 2³¹-element tests are opt-in), and compute-sanitizer memcheck, racecheck, synccheck and initcheck report no errors on `infra/sanitize_cases.py`.
The kernels compile with nvcc 12.8 for sm_80, sm_86, sm_89 and sm_90 with no spills.
v2a measurements are in `benchmarks/a10g_first_run.md` and `benchmarks/a100_first_run.md`; the v2b measurements are in the v2b section below.

## Math

For each table l = 1..L with fixed planes W_l ∈ R^{P×d} and a scalar β (a run-time argument):

- uₜ = tanh(w_{l,t} · x) and φ_l(x)[r] = ∏ₜ σ(±2βuₜ), the product form of softmax_r(β ∑ₜ c_{r,t} uₜ) over the R = 2^P corners.
- Φ(x) ∈ R^S, S = L·R, concatenates the L tables (feature s = l·R + r).
- Aᵢ = ∑_{j≤i} Φ(kⱼ) and Bᵢ = ∑_{j≤i} Φ(kⱼ) ⊗ vⱼ.
- Oᵢ = (Φ(qᵢ) · Bᵢ) / (Φ(qᵢ) · Aᵢ) = ∑_{j≤i} kᵢⱼ vⱼ / ∑_{j≤i} kᵢⱼ with kᵢⱼ = ⟨Φ(qᵢ), Φ(kⱼ)⟩.

The paper's 1/L factors cancel.
Corner r has +1 on plane t iff bit (P−1−t) of r is set, the `itertools.product` order of the repo.
If Den underflows to exactly 0 (only possible for very large β), the output row is 0.
This is not the repo's per-bucket normalization (`src/race_baseline.py`); `reference.py` has that form too, as the bridge that proves both use the same planes, corner order and β.

## Layout

| path | what |
|---|---|
| `reference.py` | `race_causal_reference` (prefix-sum definition, blocked cumsum, any T), `race_causal_dense_reference` (masked T×T, tiny T), `race_causal_perbucket_reference` (the repo's form, the bridge), `race_causal_chunked_emulation` (the kernels' tile and sub-chunk decomposition, returns the prefix states) |
| `kernels/race_causal_fwd.h` | plain C++ launcher API (no torch), tensor and workspace layouts, the `Variant` switch |
| `kernels/race_causal_fwd.cu` | the v2a kernels (templated on d ∈ {64, 128} and P ∈ {1..5}, L at run time), the shared K2 scan and the public launchers for both variants |
| `kernels/race_causal_fwd_tc.cu` | the v2b kernels (K1 and K3 on tensor cores), templated on d, P and L |
| `kernels/race_causal_internal.cuh` | declarations shared by the two `.cu` files (shape dispatch, shared-memory opt-in, the v2b entry points, `RACE_CAUSAL_PRECISE_CARRY`) |
| `kernels/torch_binding.cpp` | `forward`, `forward_debug`, and shape queries, each with `tensor_cores=False/True` |
| `build.py` | JIT build via `torch.utils.cpp_extension.load`, `load_extension()` |
| `tests/test_reference.py` | CPU tests of the references and the emulation |
| `tests/test_kernel_layout.py` | CPU checks of both variants' index math, launch shapes and shared-memory layouts |
| `tests/test_forward_cuda.py` | GPU tests of both variants (skipped without CUDA) |
| `tests/causal_numerics.py` | GPU input generation and the derived tolerances of both variants |
| `bench/bench_causal.py` | timing against the HBM roofline and the PyTorch chunked forward, `--variant v2a` or `v2b` |

The kernels include `src/noncausal/kernels/race_common.cuh` and `race_internal.cuh` by relative path; nothing under `src/noncausal/` is modified.

Inputs: q, k, v are [B, H, T, d] bf16 contiguous CUDA tensors with identical shapes and 16-byte aligned data, W is [L, P, d] fp32, β is a one-element tensor (CPU or CUDA).
Limits: d ∈ {64, 128}, P ∈ {1..5}, L ∈ {1..4}, B·H ≤ 65535, T ≤ 2³¹ − 4098, and the output pass must fit the GPU's shared memory (see below).
Forward only: the binding is not an autograd function.

## Binding

```python
from build import load_extension
race = load_extension()
o = race.forward(q, k, v, W, beta)                       # v2a, tile length chosen automatically
o = race.forward(q, k, v, W, beta, tensor_cores=True)    # v2b
o, prefix_A, prefix_B, final_A, final_B, T_blk = race.forward_debug(q, k, v, W, beta, tile_tokens=0)
race.sub_chunk_tokens(d, P)                              # C (each query takes tensor_cores=...)
race.select_tile_tokens(BH, T, d, P, L)                  # the automatic T_blk on this GPU
race.output_pass_smem_bytes(d, P, L), race.fits_on_device(d, P, L)
race.precise_carry()                                     # whether v2b was built with the hi/lo carry
```

The variant sets C and the automatic tile length, so pass the same `tensor_cores` flag to the queries and to the forward.

`prefix_A` is [B, H, tiles, L, R] and `prefix_B` is [B, H, tiles, L, R, d]: the exclusive prefix state at each tile start (the workspace after the scan).
`final_A` and `final_B` are the state after all T tokens.
`tile_tokens` forces T_blk (a positive multiple of C); the tests use it to exercise the scan with many or few tiles.

## Kernels (v2a)

1. `tile_sums_kernel` (K1): grid (tiles·L, B·H), 256 threads.
   It is the non-causal bucket build: it calls `stage_tokens` and `accumulate_stage` from `race_internal.cuh`, and only the outer tile loop is local, because the non-causal kernel fixes the tile at `kBuildTileTokens` = 2048 while the causal tile length is chosen at run time.
2. `tile_scan_kernel` (K2): one thread per float of a tile's slice (all streams at once), each walking the tiles in index order and writing exclusive prefixes in place, 8 loads in flight per thread.
   The order is fixed, so the scan is bitwise reproducible.
3. `output_pass_kernel` (K3): grid (tiles, B·H), 256 threads, one CTA per tile.
   It loads the tile's prefix into shared memory and walks the tile's sub-chunks of C tokens in order.
   Per sub-chunk: stage Q, K, V (16-byte loads); Φ_Q and Φ_K (one thread per (side, table, token), rows past the end get Φ_K = 0); G = tril(Φ_Q Φ_Kᵀ) with Den = Φ_Q·A + rowsum(G) folded in by a half-warp shuffle; Num = Φ_Q B + G V, times 1/Den, stored as bf16; then A += colsum(Φ_K) and B += Φ_Kᵀ V.
   The products use register micro-tiles of at most 16 accumulators per thread on a 16×16 thread grid, with 5 barriers per sub-chunk and no atomics.

Tile length T_blk (both variants): the smallest power of two ≥ C whose state traffic 16·S·(d+1)/T_blk bytes per token is at most 5% of the 12·d bytes of Q/K/V/O traffic (2048 for config A, 4096 for config B), shrunk for short sequences to about 4 waves of K3 CTAs, never below C.
Sub-chunk C (v2a): 64 when 64·d ≤ 4096 and the L = 4 footprint fits 99 KiB (the sm_89 limit), else 32; so C = 64 for d = 64 with P ≤ 4 and C = 32 otherwise. v2b uses C = 32 throughout (see the v2b section).

## Workspace and shared memory

Workspace: fp32 [tiles, B·H, L, R·(d+1)], the non-causal layout: in each slice B[r][c] is at r·d + c and A[r] at R·d + r.
Example: T = 2²¹, B·H = 8, config A (d = 64, P = 4, L = 4, T_blk = 2048): 1024 · 8 · 4 · 1040 floats = 130 MiB.

v2a K3 dynamic shared memory (KiB) by (d, P) for L = 1..4; the opt-in limits are 99 (L40S), 163 (A100) and 227 (H100):

| d | P | C | L = 1 | L = 2 | L = 3 | L = 4 |
|---|---|---|---|---|---|---|
| 64 | 1 | 64 | 27.3 | 29.0 | 30.8 | 32.5 |
| 64 | 4 | 64 | 38.6 | 51.6 | 64.7 | 77.75 |
| 64 | 5 | 32 | 30.1 | 47.5 | 64.9 | 82.25 |
| 128 | 4 | 32 | 38.8 | 52.9 | 66.9 | 81.0 |
| 128 | 5 | 32 | 51.4 | 78.0 | 104.6 | 131.25 |

d = 128, P = 5 with L ≥ 3 does not fit L40S; the binding refuses it there with a message and the GPU tests skip it.
The plan's DV_SPLIT fallback for that case is not implemented.

## v2a registers (nvcc 12.8, -O3, -Xptxas -v)

| kernel | sm_80 | sm_89 | sm_90 | spills |
|---|---|---|---|---|
| `tile_sums_kernel<d, P>` | 31-61 | 38-64 | 31-63 | 0 |
| `tile_scan_kernel` | 47 | 47 | 47 | 0 |
| `output_pass_kernel<64, 1..4, 64>` | 80 | 80 | 80 | 0 |
| `output_pass_kernel<64, 5, 32>` | 76 | 76 | 80 | 0 |
| `output_pass_kernel<128, 1..5, 32>` | 71-80 | 72-80 | 72-80 | 0 |

K3 uses `__launch_bounds__(256, 3)` (at most 80 registers).
A 64-register cap would add 4-32 bytes of spills in the C = 64 instantiations and would not add a resident CTA, because at the benchmarked shapes shared memory already limits K3 to 1 CTA per SM on L40S and 2 on A100 and H100.

## Build, test, benchmark

From this directory, with the project venv:

```bash
python build.py                                                        # JIT build, prints the ptxas report
python -m pytest -q tests/test_reference.py tests/test_kernel_layout.py   # CPU, a few seconds
python -m pytest -q tests/test_forward_cuda.py                         # GPU
RACE_CAUSAL_HUGE=1 python -m pytest -q tests/test_forward_cuda.py -k 2_to_the_31   # 2^31 elements, ~25 GB
python bench/bench_causal.py                                           # v2a, T = 2^14 .. 2^21, B·H = 8
python bench/bench_causal.py --variant v2b                             # v2b, same table
```

Run each test directory on its own: `src/noncausal` also has a `reference.py` and a `tests/test_reference.py`.

Compile-only check without a GPU or torch (run from `kernels/`, with `src/noncausal/kernels` present at its relative path):

```bash
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c race_causal_fwd.cu
```

## Tolerances

The GPU tests compare the bf16 output with the fp64 reference on the same bf16-rounded inputs.
Let F be the largest and F_rms the RMS of |bf16(O_ref) − O_ref|, the rounding error a perfect kernel cannot avoid.
v2a must reach max error ≤ 1.1·F and RMS error ≤ 1.05·F_rms, and v2b max error ≤ 2.5·F and RMS error ≤ 2·F_rms (plan section 2.4), plus tiny absolute allowances that only matter when F is 0.
v2b against v2a must agree within the sum of the two bounds, (2.5 + 1.1)·F and (2 + 1.05)·F_rms: both are compared with the same reference, and two correct outputs that round a nearby value in opposite directions already differ by up to 3·F.
The derivations are in `tests/causal_numerics.py`.
Over the reference test's grid (d, P, L, T, B·H and three β) on the A10G the worst cases were 1.00·F and 1.00·F_rms for v2a, 1.69·F (d = 64, P = 2, L = 4, T = 2047, β = 1) and 1.42·F_rms (d = 128, P = 5, L = 2, T = 20000, β = 1) for v2b, and 2.33·F and 1.53·F_rms for v2b against v2a.
Structural tests, for both variants: constant V must come back exactly (v2b with the hi/lo carry), O₀ = v₀ at T = 1, rows up to i are bitwise unchanged when later tokens change, runs are bitwise identical, runs with different T_blk agree within one bf16 ulp, NaN in other streams does not leak, and the debug prefix states match the emulation (for v2b within the bf16 rounding of Φ_K, 2⁻⁸ per term, against an emulation with the kernel's effective planes W_hi + W_lo).

## v2b: tensor cores

v2a computes every product on fp32 CUDA cores, so its error is limited by the bf16 output rounding alone, and a wrong mask, boundary, scan offset or tail shows up as a clear failure.
On the A10G it is compute-bound at P = 4, L = 4 (its output pass was MIO-throttled at 16.7% occupancy), which is what v2b addresses.
v2b replaces K1 and K3 with `nvcuda::wmma` kernels (bf16 16×16×16 fragments, fp32 accumulation) and keeps K2, the workspace layout and the tile logic.
The kernel file's header comment (`kernels/race_causal_fwd_tc.cu`) documents the shared-memory layout, the warp mapping and the reasons for each choice; the summary:

- **On tensor cores:** the projection U = X Wᵀ (W as bf16 hi + lo), G = Φ_Q Φ_Kᵀ over the 16×16 tiles on or below the diagonal, the intra term (G_hi + G_lo) V, the carry Φ_Q (B_hi + B_lo) (`RACE_CAUSAL_PRECISE_CARRY`, on by default), and the state update B += Φ_Kᵀ V, whose fp32 master stays in accumulator fragments for the whole tile.
- **On CUDA cores:** tanh, the sigmoid pairs and the corner products (a product tree, bitwise equal to v2a's order), the causal mask and the hi/lo splits, Den = Φ_Q·A + rowsum(G_hi + G_lo), A += colsum(Φ_K), and the division.
- **Precision:** Φ_Q and Φ_K are rounded to bf16 (unit roundoff 2⁻⁸); everything else carries about 16 significant bits or more.
  Den sums exactly the weights the tensor cores apply to V, so a constant V comes back exactly.
  G is split into hi + lo, which the design did not plan: with G rounded to bf16 alone, the first rows of a sequence (queries that see few keys) reached 2.74·F against the 2.5·F bound (details in `tests/causal_numerics.py`).
- **Launch shape:** C = 32 for every (d, P).
  The output pass runs 2 CTAs of 8 warps per SM for d = 64 with P ≤ 4 (48.8 KiB each at L = 4) and 1 CTA of 16 warps otherwise; the tile sums run 8 warps (16 for d = 128, P = 5).
  Two CTAs per SM overlap one CTA's CUDA-core phases with the other's tensor-core phases, which measured faster than one larger CTA.
  L is a template parameter of the v2b kernels (80 instantiations per architecture), so strides and fragment loops are compile-time constants.
- **Per sub-chunk:** 6 barriers in the output pass (staged rows, U, Φ, G, Num, update) and 4 in K1; the next sub-chunk's Q, K, V are loaded into registers during the current one.

v2b K3 shared memory (KiB, dynamic) by (d, P) for L = 1..4, and resident CTAs per SM on a 100 KiB SM (A10G, L40S) at L = 4:

| d | P | C | warps | CTAs/SM | L = 1 | L = 2 | L = 3 | L = 4 |
|---|---|---|---|---|---|---|---|---|
| 64 | 1-2 | 32 | 8 | 2 | 36.6 | 36.6 | 36.6 | 36.6 |
| 64 | 3 | 32 | 8 | 2 | 36.6 | 36.6 | 40.6 | 40.6 |
| 64 | 4 | 32 | 8 | 2 | 36.6 | 40.6 | 44.7 | 48.8 |
| 64 | 5 | 32 | 16 | 1 | 48.6 | 56.8 | 68.9 | 85.5 |
| 128 | 1-2 | 32 | 16 | 1 | 66.6 | 66.6 | 66.6 | 66.6 |
| 128 | 3 | 32 | 16 | 1 | 66.6 | 66.6 | 74.6 | 74.6 |
| 128 | 4 | 32 | 16 | 1 | 66.6 | 74.6 | 82.7 | 90.8 |
| 128 | 5 | 32 | 16 | A100 / H100 only | 74.6 | 90.8 | 106.9 | 131.5 |

v2b K1 needs 21.6-30.5 KiB (d = 64) and 33.6-50.5 KiB (d = 128).
d = 128, P = 5 with L ≥ 3 does not fit a 99 KiB GPU in either variant.

v2b registers (nvcc 12.8, -O3; both kernels are capped at 128 per thread, the budget of 512 resident threads):

| kernel | sm_80 | sm_86 | sm_89 | sm_90 | spills |
|---|---|---|---|---|---|
| `output_pass_tc_kernel`, d = 64, P ≤ 4 (8 warps) | 104-128 | 106-128 | 106-128 | 102-126 | 0 |
| `output_pass_tc_kernel`, d = 64, P = 5 and d = 128 (16 warps) | 110-128 | 112-128 | 112-128 | 110-128 | 0 |
| `tile_sums_tc_kernel`, d = 64 | 92-128 | 92-128 | 92-128 | 90-128 | 0 |
| `tile_sums_tc_kernel`, d = 128 | 126-128 | 126-128 | 126-128 | 126-128 | 0 |

`RACE_CAUSAL_PRECISE_CARRY=0 python build.py` builds v2b with the single bf16 carry (as the separate extension `race_causal_v2_single_carry`); the default is the hi/lo carry.
That build also compiles with no spills on the four architectures, and its v2b tests pass on the A10G (574 passed, 1 skipped: the opt-in 2³¹-element test).
Without B_lo, d = 128, P = 5, L = 4 needs 97.5 KiB and fits a 99 KiB GPU; the tile-length agreement check is not applied to that build, for the reason in `tests/causal_numerics.py`.

### Measured on an A10G

A10G (sm_86, 600 GB/s), torch 2.10, CUDA 12.8; `bench/bench_causal.py --skip-baseline`, B·H = 8, β = 1/√d, median of 20 runs after 5 warmups, each variant timed while no other process used the GPU.
GB/s and % of peak use the model bytes 12·d + 16·S·(d+1)/T_blk per token with each variant's own T_blk, so the two variants' bytes differ slightly where their tile lengths differ.

| d, P, L | T | T_blk v2a / v2b | v2a ms | v2b ms | speedup | v2b Mtok/s | v2b GB/s | v2a % peak | v2b % peak | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 64, 4, 4 | 2¹⁴ | 256 / 128 | 1.231 | 0.462 | 2.66× | 283.8 | 365.6 | 18.2% | 60.9% | 0.08 |
| 64, 4, 4 | 2¹⁵ | 512 / 256 | 2.365 | 0.722 | 3.28× | 362.9 | 373.0 | 16.6% | 62.2% | 0.14 |
| 64, 4, 4 | 2¹⁶ | 1024 / 512 | 4.637 | 1.243 | 3.73× | 421.9 | 378.9 | 15.7% | 63.1% | 0.27 |
| 64, 4, 4 | 2¹⁷ | 2048 / 1024 | 9.168 | 2.263 | 4.05× | 463.3 | 386.0 | 15.3% | 64.3% | 0.52 |
| 64, 4, 4 | 2¹⁸ | 2048 / 2048 | 17.361 | 4.305 | 4.03× | 487.2 | 390.0 | 16.1% | 65.0% | 1.02 |
| 64, 4, 4 | 2¹⁹ | 2048 / 2048 | 34.735 | 8.560 | 4.06× | 490.0 | 392.3 | 16.1% | 65.4% | 2.03 |
| 64, 4, 4 | 2²⁰ | 2048 / 2048 | 70.380 | 16.984 | 4.14× | 493.9 | 395.4 | 15.9% | 65.9% | 4.06 |
| 64, 4, 4 | 2²¹ | 2048 / 2048 | 140.938 | 33.891 | 4.16× | 495.0 | 396.3 | 15.9% | 66.0% | 8.13 |
| 128, 4, 4 | 2¹⁴ | 256 / 256 | 2.038 | 0.842 | 2.42× | 155.7 | 319.5 | 22.0% | 53.3% | 0.14 |
| 128, 4, 4 | 2¹⁵ | 512 / 512 | 3.918 | 1.450 | 2.70× | 180.7 | 324.2 | 20.0% | 54.0% | 0.27 |
| 128, 4, 4 | 2¹⁶ | 1024 / 1024 | 7.601 | 2.674 | 2.84× | 196.1 | 326.4 | 19.1% | 54.4% | 0.52 |
| 128, 4, 4 | 2¹⁷ | 2048 / 2048 | 15.038 | 5.004 | 3.01× | 209.5 | 335.4 | 18.6% | 55.9% | 1.02 |
| 128, 4, 4 | 2¹⁸ | 2048 / 2048 | 28.430 | 9.212 | 3.09× | 227.7 | 364.4 | 19.7% | 60.7% | 2.03 |
| 128, 4, 4 | 2¹⁹ | 2048 / 2048 | 57.250 | 18.234 | 3.14× | 230.0 | 368.2 | 19.5% | 61.4% | 4.06 |
| 128, 4, 4 | 2²⁰ | 2048 / 2048 | 114.921 | 36.314 | 3.16× | 231.0 | 369.7 | 19.5% | 61.6% | 8.13 |
| 128, 4, 4 | 2²¹ | 2048 / 2048 | 228.787 | 71.867 | 3.18× | 233.4 | 373.6 | 19.6% | 62.3% | 16.25 |

Peak memory is the same for both variants (the v2b columns are shown; v2a matches within 0.01 GiB).
v2b picks shorter tiles than v2a at the shortest T because its output pass fits 2 CTAs per SM.

Nsight Compute (`--set full`), d = 64, P = 4, L = 4, T = 2¹⁸, B·H = 8 (clocks locked by ncu):

| kernel | ms | DRAM % | SM % | tensor pipe % | occupancy achieved / theoretical | registers | CTAs / SM | top stalls |
|---|---|---|---|---|---|---|---|---|
| v2a K1 `tile_sums_kernel` | 6.32 | 18.2 | 86.0 | 0 | 96.2 / 100 | 40 | 6 | short scoreboard 29%, not selected 24% |
| v2b K1 `tile_sums_tc_kernel` | 1.15 | 89.6 | 54.7 | 16.4 | 32.3 / 33.3 | 110 | 2 | barrier 39%, wait 15% |
| K2 `tile_scan_kernel` (shared) | 0.10 | 65-69 | 4.1 | 0 | 26-27 / 83.3 | 47 | | long scoreboard 95% |
| v2a K3 `output_pass_kernel` | 10.94 | 18.7 | 85.6 | 0 | 16.7 / 16.7 | 80 | 1 | MIO throttle 31%, selected 21% |
| v2b K3 `output_pass_tc_kernel` | 3.04 | 67.0 | 61.4 | 21.1 | 32.3 / 33.3 | 128 | 2 | barrier 19%, wait 18%, math pipe throttle 15%, short scoreboard 12% |

The tensor-pipe column is `sm__pipe_tensor_op_hmma_cycles_active` as a share of active cycles.
DRAM traffic per kernel: v2b K1 618 MB (v2a 687 MB), v2b K3 1.22 GB (v2a 1.23 GB), against the 1.07 GB of Q, K, V and O plus the 34 MB workspace K3 must move.
For d = 128, P = 4, L = 4 the v2b K1 takes 2.22 ms at 92.6% of DRAM peak and K3 6.77 ms at 60.5%.

What limits v2b now: K1 is at 90% of DRAM peak, so it is done; K3 moves the right bytes but reaches 67% of peak because each CTA's phases are separated by 6 barriers per 32 tokens and the tensor pipe is busy only 21% of the time.
The wmma API costs K3 instructions that raw `mma.sync` would not: fragment loads from shared memory compile to generic 32-bit loads (not `ldmatrix`), row-major B and column-major A operands add `movmatrix` transposes, and every element-wise step is a store_matrix_sync and reload through scratch (53 M shared-memory bank conflicts in K3 against 17 M in v2a).
The next steps are `ldmatrix` + `mma.sync` with documented fragment layouts (the design's v2b-3), which removes the scratch round trips of the G mask, the division and the hi/lo splits, and measuring C = 64 with 2 CTAs per SM on the 164 KiB and 228 KiB SMs of A100 and H100.

## Known untested

- v2a and v2b have run on an A10G (sm_86), an L40S (sm_89) and an A100 (sm_80); sm_90 is compiled but not run.
  See `benchmarks/a100_tensor_cores.md` and `benchmarks/l40s_run.md`.
- v2b with the default hi/lo carry for d = 128, P = 5 with L ≥ 3 needs more than 99 KiB of shared memory and has only been compiled; its output pass without the register prefetch (`TcTraits::prefetch_rows`) has run on the A10G only in the single-carry build.
- The v2b launch shapes were chosen from A10G measurements for the smallest per-SM shared memory of the targets; with 164 KiB (A100) or 228 KiB (H100) per SM, C = 64 with 2 CTAs per SM might be faster and has not been measured.
- compute-sanitizer has run on the default build only, not on the single-carry build.
- The 2³¹-element offset test needs about 25 GB and is opt-in; it has not been run (the A10G has 24 GB).
- β ≳ 30 gives non-finite outputs for some early tokens: Den can fall below 2⁻¹²⁸, where 1 / Den overflows to inf.
