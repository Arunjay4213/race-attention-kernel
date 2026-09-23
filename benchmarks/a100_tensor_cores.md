# Tensor-core kernels on an A100-SXM4-40GB

Date: 2026-09-23.
Hardware: one NVIDIA A100-SXM4-40GB (sm_80, 1555 GB/s peak HBM) on a SageMaker ml.p4d.24xlarge training job, one of the eight GPUs.
Software: the PyTorch 2.8.0+cu129 Deep Learning Container, nvcc 12.9.
Every number comes from that job (686 billed seconds).
The fp32-core numbers are from the same job and match the earlier run in `a100_first_run.md` within 1%.

## Correctness

| suite | result |
|---|---|
| build, sm_80/86/89/90, both variants of both kernel families | pass, 0 bytes spilled in all 932 kernel images |
| compute-sanitizer memcheck / racecheck / synccheck | 0 errors, 0 hazards, 0 errors |
| non-causal tests (fp32 and tensor-core paths, forward and backward) | 2170 passed |
| causal v2 tests (v2a and v2b) | 1717 passed, 2 skipped (the opt-in 2³¹-element tests) |

## Non-causal forward (d = 128, B·H = 4, median of 20)

| P,L | N | fp32 ms | fp32 GB/s | fp32 % peak | tc ms | tc GB/s | tc % peak | speedup |
|---|---|---|---|---|---|---|---|---|
| 2,2 | 2¹⁷ | 1.107 | 485.0 | 31.2% | 0.566 | 948.1 | 61.0% | 1.96× |
| 2,2 | 2¹⁸ | 2.057 | 521.9 | 33.6% | 0.980 | 1095.7 | 70.5% | 2.10× |
| 2,2 | 2¹⁹ | 3.722 | 576.9 | 37.1% | 1.778 | 1208.0 | 77.7% | 2.09× |
| 2,2 | 2²⁰ | 7.214 | 595.4 | 38.3% | 3.476 | 1235.6 | 79.5% | 2.08× |
| 4,4 | 2¹⁷ | 4.455 | 120.5 | 7.7% | 0.872 | 615.4 | 39.6% | 5.11× |
| 4,4 | 2¹⁸ | 8.456 | 127.0 | 8.2% | 1.470 | 730.2 | 47.0% | 5.75× |
| 4,4 | 2¹⁹ | 15.891 | 135.1 | 8.7% | 2.774 | 774.1 | 49.8% | 5.73× |
| 4,4 | 2²⁰ | 31.227 | 137.5 | 8.8% | 5.324 | 806.8 | 51.9% | 5.87× |

Bytes moved count q, k, v read once and the output written once in bf16.
At P = 2, L = 2 the tensor-core forward runs at 80% of HBM peak.
At P = 4, L = 4 it runs at 52%: on the A100 the remaining CUDA-core work per token (tanh, sigmoids, the corner products) and the barriers between the stages of each 32-token batch still cost more than the bytes, though 5.9× less than before.

## Causal forward (B·H = 8, β = 1/√d, median of 20)

v2a is the fp32-core kernel, v2b the tensor-core kernel.
Peak memory is identical for the two: 8.13 GiB at T = 2²¹ and d = 64, 16.25 GiB at d = 128.

| d,P,L | T | v2a ms | v2b ms | speedup | v2b Mtok/s | v2b GB/s (min bytes) | v2b GB/s (model bytes) |
|---|---|---|---|---|---|---|---|
| 64,4,4 | 2¹⁴ | 0.833 | 0.302 | 2.76× | 434.6 | 222.5 | 559.8 |
| 64,4,4 | 2¹⁷ | 6.086 | 1.593 | 3.82× | 658.1 | 336.9 | 548.2 |
| 64,4,4 | 2²⁰ | 46.064 | 11.570 | 3.98× | 725.0 | 371.2 | 580.4 |
| 64,4,4 | 2²¹ | 91.959 | 22.938 | 4.01× | 731.4 | 374.5 | 585.5 |
| 128,4,4 | 2¹⁷ | 8.930 | 3.087 | 2.89× | 339.6 | 347.8 | 543.6 |
| 128,4,4 | 2²⁰ | 67.682 | 22.772 | 2.97× | 368.4 | 377.2 | 589.6 |
| 128,4,4 | 2²¹ | 135.109 | 45.400 | 2.98× | 369.5 | 378.4 | 591.5 |

"Min bytes" counts q, k, v and the output once.
"Model bytes" adds the second read of k and v that the chunk-parallel design needs (the tile-sum pass and the output pass both read them) and the tile states; it is the traffic the kernel actually issues.
At T = 2²¹ and d = 64 the causal forward runs at 38% of HBM peak by the model count, 4.0× faster than v2a.

## Against the baselines at T = 2²¹, d = 64, P = 4, L = 4

| implementation | ms | Mtok/s | peak GiB |
|---|---|---|---|
| reference (Phase 1, cumsum path) | out of memory at T = 131072 | | |
| chunked PyTorch forward, fp32 | 3655.9 | 4.6 | 36.0 |
| v1 prototype, one block per stream | flat at 1.08 Mtok/s at every T | 1.08 | |
| v2a, fp32 cores | 92.0 | 182 | 8.13 |
| v2b, tensor cores | 22.9 | 731 | 8.13 |

## What limits v2b

The A10G profile of the v2b output kernel (in `src/causal_v2/README.md`) shows 67% of that GPU's DRAM peak with the top stalls at barriers and at the tensor pipe, and the wmma fragment loads compiling to plain 32-bit loads rather than ldmatrix.
The A100 has 2.6× the bandwidth of the A10G with a similar shared-memory and barrier cost per sub-chunk, so the same kernel lands at 38% here.
Closing that gap means raw mma.sync with ldmatrix and fewer barriers per sub-chunk, which is the next step if a higher number is needed.
The tensor-core rounding of the hash weights (bf16 for Φ) is a change to the function, so v2a and the fp32 non-causal path remain the defaults until a training comparison shows it is harmless.
