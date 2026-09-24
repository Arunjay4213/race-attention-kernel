# First GPU validation on NVIDIA A10G

Date: 2026-09-22.
Hardware: one NVIDIA A10G (sm_86, 24 GB, 99 KB opt-in shared memory per block, 600 GB/s peak HBM) on a SageMaker ml.g5.xlarge.
Software: torch 2.10.0+cu128, nvcc 12.8, Python 3.10, compute-sanitizer and ncu from CUDA 13.2.
All numbers below are from this run.
The raw logs are kept outside the repository.

## Result per suite

| suite | result |
|---|---|
| build (both extensions, sm_80/86/89/90) | pass, zero spills on every kernel and architecture |
| compute-sanitizer memcheck / racecheck / synccheck | pass: 0 errors, 0 hazards, 0 errors (7 cases, forward and backward) |
| non-causal test_forward_cuda.py | 593 passed on the first run |
| non-causal test_every_template_instantiation_backward | failed on the first run (bug 1), passes after the fix |
| non-causal test_backward_cuda.py | 311 passed after the fix |
| non-causal tests/ (CPU and GPU) | 1268 passed |
| causal test_debug_prefix_states_match_emulation | 15 passed on the first run, so K1 and K2 are correct |
| causal tests/ | first run: 20 failed, 735 passed, 19 skipped (all 20 in one assertion, item 2); after the fix: 755 passed, 19 skipped |

18 of the 19 causal skips are d = 128, P = 5, L = 4, which needs 134400 bytes of shared memory while this GPU allows 101376.
The 19th is the opt-in 2³¹-element test.
The causal RMS bound at T = 20000 passed in every configuration.

## What failed and how it was fixed

### 1. Backward launch refused at d = 64, P = 5, L = 4 (real bug, fixed)
race.backward failed with cudaErrorInvalidValue for exactly one of the 40 (d, P, L) shapes.
query_grad_kernel<64,5> at L = 4 needs 12288 floats = 49152 bytes of dynamic shared memory, exactly the 48 KB default.
opt_in_dynamic_smem only opted in above 48 KB.
The kernel also holds 32 bytes of static shared memory (warp_beta_s), and the 48 KB default limit covers static plus dynamic together.
So the launch asked for 49184 bytes against a 49152-byte limit, and the driver refused it.
The fix adds the static bytes to the comparison, which also covers key_grad_kernel (same helper, same static array).
After the fix all 40 shapes run.
The forward query_kernel and the causal K3 have no static shared memory, so they were not affected.

### 2. Causal "tile lengths agree within one bf16 ulp" check (the tolerance was wrong, fixed with numbers)
The check required |O₁ − O₂| ≤ max(|O₁|,|O₂|)·2⁻⁷ with no absolute term.
Every single run passed the reference tolerance; only this cross-run check failed, and only for T ≥ 2047.
Over the test grid, 511 of about 1.9×10⁷ elements were more than one ulp apart.
All of them had |O| between 10⁻¹¹ and 10⁻⁶, where the weighted sum nearly cancels to 0.
The largest absolute difference among them was 1.15×10⁻⁸·max|v|.
At |O| ≈ 10⁻⁹ one bf16 ulp is about 4×10⁻¹², so fp32 noise from a different summation order counts as up to 234 ulps.
The check is now one ulp plus the suite's existing fp32 allowance a = 10⁻⁵·max(1,β)·max|v|.
That allowance is 870× the worst measured difference and far below the 2⁻⁷|O| that a scan or carry error causes at typical |O|.
The derivation is written in causal_numerics.py.

### 3. Other fixes along the way
infra/sanitize_cases.py only ran forward calls, and its largest causal case is refused on a 99 KB GPU.
It now also runs forward_train plus backward, and has two more cases: (1, 2049, 128, 4, 4), the largest causal footprint that fits, and (1, 100, 64, 5, 4), the C = 32 path at d = 64.
The synccheck run on these cases was the first to hit bug 1.
The bench scripts had no A10G entry in their peak bandwidth table, so "% peak" printed n/a.
The runbook's peak_memory step always printed 0.0, because it ran in a fresh Python process; removed.

## Look-first items
Non-causal forward: the stage loop and the per-warp φ buffer (__syncwarp only) are clean under racecheck and synccheck; tail lengths N = 1, 16, 17, 127, 128, 129, 2047, 2048, 2049, 10000 pass; the 65-tile 3-pass reduce passes; the misaligned view is rejected.
Non-causal backward: corner_reduce_scatter, the __syncwarp placement, the weighted staging, reciprocal() and the error-message match strings all pass; no ulp-level offset in dq or dk.
Causal v2: the gram-over-staged-Q/K aliasing gives 0 racecheck hazards; update_state at L = 3 passes, as do the Den shuffle, prepare_output_pass, the 99 KB refusal, automatic tile selection and the 16-byte alignment rejection.
Den in (0, 2⁻¹²⁸), report only: at B·H = 4, T = 4096, d = 64, P = 5, L = 4 the non-causal forward and backward stay finite up to β = 100; causal v2 gives 64 non-finite outputs at β = 30, 192 at β = 45 and 60, and 64 at β = 100 (manual/large_beta_probe.log). Not fixed.

## Registers on sm_86 (ptxas, nvcc 12.8)
Every kernel on all four architectures reports 0 bytes of spill.
bucket_build<64,*> 37-48; bucket_build<128,*> 37-64; query_kernel 39-43; query_grad 36-64; key_grad 39-64; tree_reduce 38; beta_grad_reduce 40; tile_sums 39-64; tile_scan 47; output_pass<64,1..4,64> 80; <64,5,32> 76; <128,*,32> 72-80.

## Benchmarks (% of 600 GB/s)

Non-causal forward (d = 128, B·H = 4). Peak memory measured separately with q, k, v and dO allocated.

| P,L | N | ms | GB/s | % peak | peak GiB |
|---|---|---|---|---|---|
| 2,2 | 2¹⁶ | 0.806 | 333.1 | 55.5% | 0.313 |
| 2,2 | 2¹⁸ | 2.549 | 421.2 | 70.2% | 1.252 |
| 2,2 | 2²⁰ | 9.655 | 444.9 | 74.1% | 5.008 |
| 4,4 | 2¹⁶ | 3.071 | 87.4 | 14.6% | 0.316 |
| 4,4 | 2¹⁸ | 10.344 | 103.8 | 17.3% | 1.266 |
| 4,4 | 2²⁰ | 40.287 | 106.6 | 17.8% | 5.063 |

Non-causal backward / forward + backward (d = 128, B·H = 4)

| P,L | N | bwd ms | GB/s | % | fwd+bwd ms | GB/s | % | peak GiB |
|---|---|---|---|---|---|---|---|---|
| 2,2 | 2¹⁸ | 6.121 | 307.0 | 51.2% | 8.666 | 340.7 | 56.8% | 2.010 |
| 2,2 | 2²⁰ | 23.294 | 322.7 | 53.8% | 33.013 | 357.8 | 59.6% | 8.039 |
| 4,4 | 2¹⁸ | 24.079 | 78.0 | 13.0% | 34.536 | 85.5 | 14.2% | 2.024 |
| 4,4 | 2²⁰ | 94.324 | 79.7 | 13.3% | 135.067 | 87.4 | 14.6% | 8.094 |

Causal v2 (B·H = 8, β = 1/√d). The baseline is race_chunked in fp32.

| d,P,L | T | ms | model GB/s | % peak | min GB/s | peak GiB | baseline ms | speedup |
|---|---|---|---|---|---|---|---|---|
| 64,4,4 | 2¹⁴ | 1.232 | 109.3 | 18.2% | 54.5 | 0.07 | 45.33 | 36.8× |
| 64,4,4 | 2¹⁸ | 17.347 | 96.8 | 16.1% | 61.9 | 1.02 | 722.99 | 41.7× |
| 64,4,4 | 2²⁰ | 69.964 | 96.0 | 16.0% | 61.4 | 4.07 | 2888.71 | 41.3× |
| 64,4,4 | 2²¹ | 139.909 | 96.0 | 16.0% | 61.4 | 8.13 | OOM | |
| 128,4,4 | 2¹⁸ | 28.450 | 118.0 | 19.7% | 75.5 | 2.04 | 1319.53 | 46.4× |
| 128,4,4 | 2²¹ | 228.048 | 117.7 | 19.6% | 75.3 | 16.26 | OOM | |

## Causal v1 vs v2 (d = 64, P = 4, L = 4, B·H = 8)
v1 matches BatchedACE within 2⁻⁸·max|O| at T = 1, 17, 512 and 2048 (worst 2.80×10⁻² against a bound of 4.99×10⁻², at T = 512).
v1 computes the per-bucket form and v2 computes Algorithm 1, so they are not the same function numerically.

| T | v1 ms | v1 GB/s | v1 peak GiB | v2 ms | v2 min GB/s | v2 peak GiB | v2 speedup |
|---|---|---|---|---|---|---|---|
| 2¹⁴ | 101.33 | 0.66 | 0.070 | 1.232 | 54.5 | 0.07 | 82.2× |
| 2¹⁶ | 405.41 | 0.66 | 0.258 | 4.633 | 57.9 | 0.27 | 87.5× |
| 2¹⁸ | 1620.20 | 0.66 | 1.008 | 17.347 | 61.9 | 1.02 | 93.4× |

v1 stays at 1.29 Mtok/s at every length: it has only 8 CTAs (one per stream), and each walks its sequence one token at a time.

## ncu findings (--set full)
Shapes: non-causal B·H = 4, N = 2¹⁸, d = 128, P = 4, L = 4 (forward and backward). Causal B·H = 8, T = 2¹⁸, d = 64, P = 4, L = 4.
Kernel times sum to within 1% of the bench times.

| kernel | µs | DRAM % | SM % | occupancy achieved/theoretical | regs | bank conflicts | top stall |
|---|---|---|---|---|---|---|---|
| bucket_build<128,4,fwd> | 4504 | 24.5 | 70.9 | 63.6/66.7 | 63 | 1.06M (0.31%) | short scoreboard |
| tree_reduce (3 launches) | 41.5+7.3+3.3 | 89.0/55.9/15.0 | ~15 | 98/41/25 | 38 | 0 | long scoreboard |
| query_kernel<128,4> | 5795 | 17.9 | 51.7 | 31.7/33.3 | 43 | 13.5M (2.8%) | short scoreboard |
| query_grad<128,4> | 9920 | 15.9 | 71.3 | 31.9/33.3 | 64 | 0.74M (0.09%) | short scoreboard |
| bucket_build<128,4,weighted> | 4401 | 26.4 | 74.2 | 63.7/66.7 | 64 | 1.24M (0.36%) | short scoreboard |
| key_grad<128,4> | 9515 | 21.8 | 73.0 | 32.0/33.3 | 64 | 1.41M (0.18%) | short scoreboard |
| tile_sums<64,4> (K1) | 6353 | 18.0 | 85.9 | 96.2/100 | 40 | 2.28M (0.49%) | short scoreboard |
| tile_scan (K2) | 98 | 68.1 | 4.0 | 26.8/83.3 | 47 | 0 | long scoreboard |
| output_pass<64,4,64> (K3) | 10936 | 18.7 | 85.6 | 16.7/16.7 | 80 | 16.9M (2.5%) | MIO throttle |

Compute-bound, not memory-bound: at P = 4, L = 4 every main kernel runs at 52-86% SM throughput but only 16-26% DRAM.
The hashing work (dot products, tanh, exp, corner products) and the R·d dot products cost more than 600 GB/s of I/O can feed.
This is why P = 4, L = 4 reaches 13-18% of peak while P = 2, L = 2 reaches 74%.
K3 fits one CTA per SM (79.6 KB of shared memory) and is limited by MIO throttle (shared-memory instruction queue full); its bank-conflict rate is small (2.5%), so the cost is the volume of shared-memory traffic.
The planned v2b tensor-core version targets exactly this.
K1 is 37% of the causal forward time, mostly hashing keys; K3 hashes the same keys again.
Tree reduce and scan are memory-bound as intended and take under 1% of the time.

## Code changes

| file | change | reason |
|---|---|---|
| src/noncausal/kernels/race_bwd.cu | the opt-in check now counts the static warp_beta_s bytes | bug 1 |
| src/causal_v2/tests/test_forward_cuda.py | the agreement check adds the fp32 allowance | item 2 |
| src/causal_v2/tests/causal_numerics.py | documents the numbers and the derivation | required for a tolerance change |
| infra/sanitize_cases.py | adds the backward and two causal cases that fit 99 KB | sanitizer coverage on this GPU |
| src/noncausal/bench/bench_forward.py, src/causal_v2/bench/bench_causal.py | A10G at 600 GB/s | % of peak printed n/a |
| infra/gpu_session.sh | removed the peak_memory step | it always printed 0.0 |
| src/bench_causal_v1.py (new) | builds, checks and times v1 | the "before" number |
| src/noncausal/README.md, src/causal_v2/README.md | status sections updated | they still said "never run on a GPU" |

## Not resolved or not run
Causal v2 gives non-finite outputs at β ≥ 30 (Den underflow); report only.
compute-sanitizer initcheck was not run.
The 2³¹-element test was not run: it needs about 25 GB and the A10G has 24 GB.
Causal d = 128, P = 5, L ≥ 3 cannot run on this GPU because of shared memory.
The runbook's profile stage does not work on this box: ncu needs sudo (ERR_NVGPUCTRPERM), and nsys is absent.
