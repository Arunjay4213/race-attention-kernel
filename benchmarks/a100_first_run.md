# First run on an A100-SXM4-40GB

Date: 2026-09-23.
Hardware: one NVIDIA A100-SXM4-40GB (sm_80, 1555 GB/s peak HBM, 163 KB opt-in shared memory per block) on a SageMaker ml.p4d.24xlarge training job; the runbook pins one of the eight GPUs.
Software: the PyTorch 2.8.0+cu129 Deep Learning Container, nvcc 12.9, Python 3.12.
Every number below comes from that job (416 billed seconds).
This is the same GPU model as the Phase 1 baseline, where the reference implementation ran out of memory at T = 131072.

## Correctness

| suite | result |
|---|---|
| build, sm_80/86/89/90 | pass, 0 bytes spilled in all 292 kernel images |
| compute-sanitizer memcheck / racecheck / synccheck | 0 errors, 0 hazards, 0 errors |
| non-causal tests (CPU and GPU) | 1268 passed |
| causal v2 tests | 773 passed, 1 skipped (the opt-in 2³¹-element test) |

The 163 KB shared-memory limit lets the largest causal configuration (d = 128, P = 5, L = 4, 134400 bytes) run here; it is skipped on the 99 KB GPUs.

## Causal v2 (B·H = 8, β = 1/√d, median of 20)

The baseline is the chunked PyTorch forward in fp32 (median of 3).

| d,P,L | T | ms | Mtok/s | % HBM peak | peak GiB | baseline ms | baseline GiB | speedup |
|---|---|---|---|---|---|---|---|---|
| 64,4,4 | 2¹⁴ | 0.833 | 157.4 | 13.0% | 0.08 | 28.60 | 2.30 | 34.4× |
| 64,4,4 | 2¹⁷ | 6.086 | 172.3 | 9.2% | 0.52 | 226.01 | 4.16 | 37.1× |
| 64,4,4 | 2²⁰ | 46.064 | 182.1 | 9.4% | 4.07 | 1823.19 | 19.04 | 39.6× |
| 64,4,4 | 2²¹ | 91.867 | 182.6 | 9.4% | 8.13 | 3649.03 | 36.04 | 39.7× |
| 128,4,4 | 2¹⁷ | 8.930 | 117.4 | 12.6% | 1.04 | 323.90 | 8.29 | 36.3× |
| 128,4,4 | 2²⁰ | 67.682 | 123.9 | 12.8% | 8.13 | 2595.68 | 38.04 | 38.4× |
| 128,4,4 | 2²¹ | 135.166 | 124.1 | 12.8% | 16.26 | OOM | | |

At T = 2²¹ = 2097152 and d = 64 the kernel needs 8.13 GiB, 16× past the reference implementation's 131072-token wall on this GPU, and runs in 92 ms.
Peak memory is 4 KB per token at d = 64: q, k, v and the output in bf16, plus a workspace that is a few percent of that.

## Causal v1 prototype (d = 64, P = 4, L = 4, B·H = 8)

| T | ms | Mtok/s | % HBM peak |
|---|---|---|---|
| 2¹⁴ | 121.97 | 1.07 | 0.04% |
| 2¹⁶ | 486.50 | 1.08 | 0.04% |
| 2¹⁸ | 1946.15 | 1.08 | 0.04% |

v1 matches BatchedACE at T = 17, 512 and 2048 within 2⁻⁸·max|O|.
It launches 8 blocks for 108 SMs and walks each stream one token at a time, so its throughput is flat at 1.08 Mtok/s; v2 is about 170× faster on the same GPU.

## Non-causal forward (d = 128, B·H = 4)

| P,L | N | ms | GB/s | % HBM peak |
|---|---|---|---|---|
| 2,2 | 2¹⁸ | 2.056 | 522.2 | 33.6% |
| 2,2 | 2²⁰ | 7.233 | 593.8 | 38.2% |
| 4,4 | 2¹⁸ | 8.449 | 127.1 | 8.2% |
| 4,4 | 2²⁰ | 31.207 | 137.6 | 8.9% |

## Non-causal backward and forward + backward (d = 128, B·H = 4)

| P,L | N | bwd ms | bwd GB/s | fwd+bwd ms | fwd+bwd GB/s | % HBM peak |
|---|---|---|---|---|---|---|
| 2,2 | 2²⁰ | 17.525 | 428.9 | 25.033 | 471.8 | 30.3% |
| 4,4 | 2²⁰ | 69.630 | 107.9 | 100.792 | 117.2 | 7.5% |

## What the numbers say

The kernels are compute-bound on fp32 cores at every configuration on this GPU, more so than on the A10G because the A100 has 2.6× the bandwidth for a similar fp32 rate.
P = 2, L = 2 reaches 38% of HBM peak in the forward; P = 4, L = 4 reaches 9%.
The ncu profiles on the A10G show where the time goes: the hash projections, tanh and sigmoid, the corner products, and the R × d dot products in the bucket build and query pass, all on CUDA cores.
Those three heavy stages are small matrix products over a tile of tokens; they are what the tensor-core versions move onto wmma.
The 2M-token memory result does not depend on that: it is set by the 4 KB per token of I/O and holds for the fp32-core kernels as they are.
