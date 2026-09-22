# Causal RACE Attention forward, chunk-parallel v2a (CUDA)

Causal forward pass of RACE Attention (arXiv 2510.04008, Algorithm 1 made causal, global normalization) as three bf16 CUDA kernels that are parallel over the sequence, with an fp64-capable PyTorch reference, a CPU model of the kernel decomposition, and tests.
The design is `docs/causal_v2_design.md`; this directory implements its v2a build (fp32 CUDA cores).

Status: the kernels and the torch binding compile cleanly (nvcc 12.8 for sm_80, sm_89 and sm_90 with no warnings and no spills; the full extension also builds with torch 2.14 and nvcc 13), but they have **never run on a GPU**.
The references, the chunked emulation and the index-coverage tests pass on CPU.
`tests/test_forward_cuda.py` is the first thing to run on a GPU.

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
| `kernels/race_causal_fwd.h` | plain C++ launcher API (no torch), tensor and workspace layouts |
| `kernels/race_causal_fwd.cu` | the three kernels and launchers, templated on d ∈ {64, 128} and P ∈ {1..5}, L at run time |
| `kernels/torch_binding.cpp` | `forward`, `forward_debug`, and shape queries |
| `build.py` | JIT build via `torch.utils.cpp_extension.load`, `load_extension()` |
| `tests/test_reference.py` | CPU tests of the references and the emulation |
| `tests/test_kernel_layout.py` | CPU checks of the kernels' index math and shared-memory layout |
| `tests/test_forward_cuda.py` | GPU tests (skipped without CUDA) |
| `tests/causal_numerics.py` | GPU input generation and the derived tolerances |
| `bench/bench_causal.py` | timing against the HBM roofline and the PyTorch chunked forward |

The kernels include `src/noncausal/kernels/race_common.cuh` and `race_internal.cuh` by relative path; nothing under `src/noncausal/` is modified.

Inputs: q, k, v are [B, H, T, d] bf16 contiguous CUDA tensors with identical shapes and 16-byte aligned data, W is [L, P, d] fp32, β is a one-element tensor (CPU or CUDA).
Limits: d ∈ {64, 128}, P ∈ {1..5}, L ∈ {1..4}, B·H ≤ 65535, T ≤ 2³¹ − 4097, and the output pass must fit the GPU's shared memory (see below).
Forward only: the binding is not an autograd function.

## Binding

```python
from build import load_extension
race = load_extension()
o = race.forward(q, k, v, W, beta)                       # tile length chosen automatically
o, prefix_A, prefix_B, final_A, final_B, T_blk = race.forward_debug(q, k, v, W, beta, tile_tokens=0)
race.sub_chunk_tokens(d, P)                              # C
race.select_tile_tokens(BH, T, d, P, L)                  # the automatic T_blk on this GPU
race.output_pass_smem_bytes(d, P, L), race.fits_on_device(d, P, L)
```

`prefix_A` is [B, H, tiles, L, R] and `prefix_B` is [B, H, tiles, L, R, d]: the exclusive prefix state at each tile start (the workspace after the scan).
`final_A` and `final_B` are the state after all T tokens.
`tile_tokens` forces T_blk (a positive multiple of C); the tests use it to exercise the scan with many or few tiles.

## Kernels

1. `tile_sums_kernel` (K1): grid (tiles·L, B·H), 256 threads.
   It is the non-causal bucket build: it calls `stage_tokens` and `accumulate_stage` from `race_internal.cuh`, and only the outer tile loop is local, because the non-causal kernel fixes the tile at `kBuildTileTokens` = 2048 while the causal tile length is chosen at run time.
2. `tile_scan_kernel` (K2): one thread per float of a tile's slice (all streams at once), each walking the tiles in index order and writing exclusive prefixes in place, 8 loads in flight per thread.
   The order is fixed, so the scan is bitwise reproducible.
3. `output_pass_kernel` (K3): grid (tiles, B·H), 256 threads, one CTA per tile.
   It loads the tile's prefix into shared memory and walks the tile's sub-chunks of C tokens in order.
   Per sub-chunk: stage Q, K, V (16-byte loads); Φ_Q and Φ_K (one thread per (side, table, token), rows past the end get Φ_K = 0); G = tril(Φ_Q Φ_Kᵀ) with Den = Φ_Q·A + rowsum(G) folded in by a half-warp shuffle; Num = Φ_Q B + G V, times 1/Den, stored as bf16; then A += colsum(Φ_K) and B += Φ_Kᵀ V.
   The products use register micro-tiles of at most 16 accumulators per thread on a 16×16 thread grid, with 5 barriers per sub-chunk and no atomics.

Tile length T_blk: the smallest power of two ≥ C whose state traffic 16·S·(d+1)/T_blk bytes per token is at most 5% of the 12·d bytes of Q/K/V/O traffic (2048 for config A, 4096 for config B), shrunk for short sequences to about 4 waves of K3 CTAs, never below C.
Sub-chunk C: 64 when 64·d ≤ 4096 and the L = 4 footprint fits 99 KiB (the sm_89 limit), else 32; so C = 64 for d = 64 with P ≤ 4 and C = 32 otherwise.

## Workspace and shared memory

Workspace: fp32 [tiles, B·H, L, R·(d+1)], the non-causal layout: in each slice B[r][c] is at r·d + c and A[r] at R·d + r.
Example: T = 2²¹, B·H = 8, config A (d = 64, P = 4, L = 4, T_blk = 2048): 1024 · 8 · 4 · 1040 floats = 130 MiB.

K3 dynamic shared memory (KiB) by (d, P) for L = 1..4; the opt-in limits are 99 (L40S), 163 (A100) and 227 (H100):

| d | P | C | L = 1 | L = 2 | L = 3 | L = 4 |
|---|---|---|---|---|---|---|
| 64 | 1 | 64 | 27.3 | 29.0 | 30.8 | 32.5 |
| 64 | 4 | 64 | 38.6 | 51.6 | 64.7 | 77.75 |
| 64 | 5 | 32 | 30.1 | 47.5 | 64.9 | 82.25 |
| 128 | 4 | 32 | 38.8 | 52.9 | 66.9 | 81.0 |
| 128 | 5 | 32 | 51.4 | 78.0 | 104.6 | 131.25 |

d = 128, P = 5 with L ≥ 3 does not fit L40S; the binding refuses it there with a message and the GPU tests skip it.
The plan's DV_SPLIT fallback for that case is not implemented.

## Registers (nvcc 12.8, -O3, -Xptxas -v)

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
python bench/bench_causal.py                                           # T = 2^14 .. 2^21, B·H = 8
```

Run each test directory on its own: `src/noncausal` also has a `reference.py` and a `tests/test_reference.py`.

Compile-only check without a GPU or torch (run from `kernels/`, with `src/noncausal/kernels` present at its relative path):

```bash
nvcc -std=c++17 -O3 -arch=sm_80 -Xptxas -v -c race_causal_fwd.cu
```

## Tolerances

The GPU tests compare the bf16 output with the fp64 reference on the same bf16-rounded inputs.
Let F be the largest and F_rms the RMS of |bf16(O_ref) − O_ref|, the rounding error a perfect kernel cannot avoid.
v2a must reach max error ≤ 1.1·F and RMS error ≤ 1.05·F_rms (plan section 2.4), plus tiny absolute allowances that only matter when F is 0.
The derivation is in `tests/causal_numerics.py`.
Structural tests: constant V must come back exactly, O₀ = v₀ at T = 1, rows up to i are bitwise unchanged when later tokens change, runs are bitwise identical, runs with different T_blk agree within one bf16 ulp, NaN in other streams does not leak, and the debug prefix states match the emulation.

## v2a now, v2b next

v2a computes every product on fp32 CUDA cores, so its error is limited by the bf16 output rounding alone, and a wrong mask, boundary, scan offset or tail shows up as a clear failure.
It is memory-bound on L40S but compute-bound on A100 and H100 (plan section 9.4).
v2b (planned, not started) moves the five products to tensor cores with wmma (bf16 in, fp32 accumulate), splits W into bf16 hi and lo, keeps B in accumulator fragments, and adds PRECISE_CARRY; its test is "v2b agrees with v2a within bf16 tolerance" on top of the reference test.
The workspace, K1, K2 and the tile logic carry over unchanged.

## Known untested

- Nothing here has run on a GPU: correctness, the shared-memory opt-in, the carveout hint and the occupancy query are all unverified.
- The performance of v2a is unmeasured; the plan's predictions are for v2b.
- The 2³¹-element offset test needs about 25 GB and is opt-in.
- compute-sanitizer (memcheck, racecheck, synccheck, initcheck) has not been run; it should be, at small T, before trusting the aliasing of G over the staged Q and K rows.
