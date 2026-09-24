# Run on an H100 80GB HBM3

Date: 2026-09-24.
Hardware: one NVIDIA H100 80GB HBM3 (sm_90, 3350 GB/s peak HBM, 227 KB opt-in shared memory per block) on an EC2 p5.4xlarge spot instance.
Software: PyTorch 2.12.1+cu130 with the CUDA 13.0 toolkit from the Deep Learning AMI, Python 3.13.
All numbers below are from that instance.
This is the first run of the sm_90 code path; every earlier run was on sm_80, sm_86 or sm_89.

## Correctness

| suite | result |
|---|---|
| build, sm_80/86/89/90, both variants of both kernel families | pass, 0 bytes spilled in all 932 kernel images |
| compute-sanitizer memcheck / racecheck / synccheck / initcheck | 0 errors, 0 hazards, 0 errors, 0 errors |
| non-causal tests (fp32 and tensor-core paths, forward and backward) | 2170 passed |
| causal v2 tests (v2a and v2b) | 1717 passed, 2 skipped (the opt-in 2³¹-element tests) |

The 227 KB shared-memory limit runs every configuration, including the causal d = 128, P = 5, L = 4 shapes that the 99 KB GPUs skip.

## Non-causal forward (d = 128, B·H = 4, median of 20)

| P,L | N | fp32 ms | fp32 GB/s | fp32 % peak | tc ms | tc GB/s | tc % peak | speedup |
|---|---|---|---|---|---|---|---|---|
| 2,2 | 2¹⁷ | 0.694 | 773.7 | 23.1% | 0.312 | 1722.8 | 51.4% | 2.22× |
| 2,2 | 2¹⁸ | 1.275 | 842.3 | 25.1% | 0.540 | 1987.9 | 59.3% | 2.36× |
| 2,2 | 2¹⁹ | 2.384 | 900.6 | 26.9% | 1.037 | 2070.4 | 61.8% | 2.30× |
| 2,2 | 2²⁰ | 4.552 | 943.5 | 28.2% | 1.991 | 2157.4 | 64.4% | 2.29× |
| 4,4 | 2¹⁷ | 2.494 | 215.3 | 6.4% | 0.456 | 1177.8 | 35.2% | 5.47× |
| 4,4 | 2¹⁸ | 4.812 | 223.1 | 6.7% | 0.884 | 1215.3 | 36.3% | 5.44× |
| 4,4 | 2¹⁹ | 9.401 | 228.4 | 6.8% | 1.733 | 1239.0 | 37.0% | 5.42× |
| 4,4 | 2²⁰ | 18.163 | 236.5 | 7.1% | 3.449 | 1245.2 | 37.2% | 5.27× |

## Non-causal backward and forward + backward (fp32 path, d = 128, B·H = 4)

| P,L | N | bwd ms | bwd GB/s | fwd+bwd ms | fwd+bwd GB/s | % HBM peak |
|---|---|---|---|---|---|---|
| 2,2 | 2²⁰ | 11.352 | 662.1 | 15.882 | 743.7 | 22.2% |
| 4,4 | 2²⁰ | 40.401 | 186.0 | 58.529 | 201.8 | 6.0% |

## Causal forward (B·H = 8, β = 1/√d, median of 20)

| d,P,L | T | v2a ms | v2b ms | speedup | v2b Mtok/s | v2b GB/s (model bytes) | v2b % peak | peak GiB |
|---|---|---|---|---|---|---|---|---|
| 64,4,4 | 2¹⁴ | 0.585 | 0.300 | 1.95× | 437.2 | 790.4 | 23.6% | 0.09 |
| 64,4,4 | 2¹⁷ | 3.636 | 1.089 | 3.34× | 962.5 | 864.4 | 25.8% | 0.53 |
| 64,4,4 | 2²⁰ | 27.861 | 7.520 | 3.70× | 1115.5 | 892.9 | 26.7% | 4.06 |
| 64,4,4 | 2²¹ | 55.006 | 14.726 | 3.74× | 1139.3 | 912.0 | 27.2% | 8.13 |
| 128,4,4 | 2¹⁷ | 5.597 | 1.961 | 2.85× | 534.7 | 890.3 | 26.6% | 1.03 |
| 128,4,4 | 2²⁰ | 42.921 | 14.811 | 2.90× | 566.4 | 906.5 | 27.1% | 8.13 |
| 128,4,4 | 2²¹ | 84.672 | 29.190 | 2.90× | 574.8 | 919.9 | 27.5% | 16.25 |

The chunked PyTorch baseline (fp32) at T = 2²¹ takes 2431.6 ms and 36.1 GiB at d = 64 and 3286.8 ms and 72.1 GiB at d = 128 (the H100's 80 GB lets the d = 128 baseline run, which was out of memory on the A100).
So v2b is 165× the baseline at d = 64 and 113× at d = 128.

## Causal v1 prototype (d = 64, P = 4, L = 4, B·H = 8)

| T | ms | Mtok/s | % HBM peak |
|---|---|---|---|
| 2¹⁴ | 94.69 | 1.38 | 0.02% |
| 2¹⁶ | 378.63 | 1.38 | 0.02% |
| 2¹⁸ | 1512.93 | 1.39 | 0.02% |

## Same job across the GPUs it has run on (2M-token causal forward, d = 64, P = 4, L = 4, 8 streams)

| implementation | H100 80GB | A100 40GB | L40S |
|---|---|---|---|
| v2b, tensor cores | 14.7 ms | 22.9 ms | 18.6 ms |
| v2a, fp32 cores | 55.0 ms | 92.0 ms | 65.1 ms |
| chunked PyTorch forward, fp32 | 2432 ms | 3656 ms | 4089 ms |
| v1 prototype | about 1.5 s | about 1.9 s | |
| reference implementation | | out of memory at 131072 | |

## Profile

ncu on the tensor-core variants at N = T = 2¹⁸ (metrics: duration, DRAM throughput and SM throughput as % of peak sustained, tensor-pipe active cycles, achieved occupancy, registers).

| kernel | µs | DRAM % | SM % | tensor pipe % | occupancy % | regs |
|---|---|---|---|---|---|---|
| non-causal tc bucket build (d = 128, P = 2, L = 2) | 211 | 76.5 | 42.9 | 11.2 | 23.9 | 87 |
| non-causal tree reduce (3 launches) | 4.1 + 3.4 + 3.4 | 15.7 / 2.4 / 0.4 | | 0 | | 32 |
| causal v2b K1 tile sums (d = 64, P = 4, L = 4, C = 32) | 556 | 30.6 | 37.3 | 8.3 | 24.2 | 110 |
| causal v2b K2 tile scan | 91 | 15.0 | 3.8 | 0 | 13.3 | 47 |
| causal v2b K3 output pass (d = 64, P = 4, L = 4, C = 32) | 1586 | 20.7 | 45.8 | 10.0 | 24.4 | 126 |

The non-causal tensor-core build is memory-bound here at the small configuration (76% of DRAM peak), as on the A100.
The causal output kernel is neither memory-bound nor compute-bound on the H100: 21% of DRAM, 46% SM throughput, 10% of tensor-pipe cycles, at 24% occupancy.
That combination is a latency-bound kernel: one or two CTAs per SM, six barriers per 32-token sub-chunk, and wmma fragment loads that compile to ordinary loads, so the SM waits more than it works.
The same kernel reaches 67% of DRAM on the A10G because that GPU has 5.6× less bandwidth to fill per SM.

## What the numbers say

The H100 is 1.56× the A100 on the tensor-core causal kernel and 1.67× on the fp32 one, against a 2.15× bandwidth ratio.
The share of HBM peak the kernels reach drops accordingly: v2b sits at 27% here against 38% on the A100 and 84% on the L40S, and the non-causal tensor-core forward at 64% (P = 2, L = 2) and 37% (P = 4, L = 4) against 80% and 52%.
The kernels move bytes faster than on any other GPU but the H100 has more bandwidth than their wmma fragments, barrier-separated phases and CUDA-core hashing can feed.
The design was tuned on a 99 KB shared-memory GPU with C = 32; the 227 KB here would allow C = 64 with two CTAs per SM, which has not been measured.
Closing the gap on Hopper means raw mma.sync with ldmatrix, fewer barriers per sub-chunk, and the larger sub-chunk; the memory result (2M tokens in 8.13 GiB) is unaffected.
