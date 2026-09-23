# Tensor-core kernels on an L40S

Date: 2026-09-23.
Hardware: one NVIDIA L40S (sm_89, 48 GB, 864 GB/s peak HBM, 99 KB opt-in shared memory per block) on a SageMaker ml.g6e.xlarge training job.
Software: the PyTorch 2.8.0+cu129 Deep Learning Container, nvcc 12.9.
Every number comes from that job (611 billed seconds).

## Correctness

| suite | result |
|---|---|
| non-causal tests (fp32 and tensor-core paths, forward and backward) | 2170 passed |
| causal v2 tests (v2a and v2b) | 1681 passed, 38 skipped (the d = 128, P = 5, L = 4 shapes need more shared memory than this GPU allows, plus the two opt-in 2³¹-element tests) |

## Non-causal forward (d = 128, B·H = 4, N = 2²⁰, median of 20)

| P,L | fp32 ms | fp32 % peak | tc ms | tc % peak |
|---|---|---|---|---|
| 2,2 | 6.387 | 77.8% | 6.172 | 80.5% |
| 4,4 | 19.047 | 26.1% | 6.456 | 77.0% |

## Causal forward (B·H = 8, β = 1/√d, T = 2²¹, median of 20)

| d,P,L | v2a ms | v2a % peak | v2b ms | v2b Mtok/s | v2b % peak (model bytes) | peak GiB |
|---|---|---|---|---|---|---|
| 64,4,4 | 65.083 | 23.9% | 18.565 | 903.7 | 83.7% | 8.13 |
| 128,4,4 | 107.496 | 28.9% | 37.046 | 452.9 | 83.9% | 16.25 |

The chunked PyTorch baseline at d = 64 takes 4089 ms and 36.0 GiB, so v2b is 220× faster here.

## What the numbers say

On this GPU the tensor-core kernels are memory-bound at every configuration: 77-84% of HBM peak.
The L40S has 56% of the A100's bandwidth but a similar per-SM compute rate, so the CUDA-core phases and barriers that limit the A100 run to 38-52% are hidden behind memory traffic here.
The causal 2M-token forward is faster on the L40S (18.6 ms) than on the A100 (22.9 ms) for the same reason, and that is the clearest sign of what an mma.sync rewrite with fewer barriers would buy on the A100 and H100.
