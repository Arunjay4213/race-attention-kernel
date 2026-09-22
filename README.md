# RACE Attention Kernel

Fusing RACE attention's causal forward pass into a single CUDA kernel, so that a single GPU can handle a longer context.

RACE Attention ([ICLR 2026](https://openreview.net/forum?id=RR8Lh8RHgA), [arXiv:2510.04008](https://arxiv.org/abs/2510.04008)) is a strictly linear-time attention mechanism.
Linear time is not the same as small memory, though.
The reference implementation's causal path allocates a five-dimensional prefix-sum tensor that grows by about 128 KB per token, and on a 40 GB A100 that runs out of memory at 131072 tokens.
The tensor is written once and read once, which means it does not need to exist at all.

This repository holds the measurement and analysis work behind a fused kernel that removes it.

## Findings

I traced the reference implementation, [sahiljoshi515/RACE_Attention](https://github.com/sahiljoshi515/RACE_Attention), at commit `b24a3d0`, to find out what the existing kernels cover before writing a new one.
The full walkthrough with citations is in [`analysis/hot-path.md`](analysis/hot-path.md).
The summary:

**The shipped CUDA kernels never run.**
`kernels/gpu/forward_kernel.cu` and `kernels/gpu/backward_kernels.cu` are real, reasonable CUDA, but nothing compiles or calls them.
There is no CUDA build step anywhere in the repository (the only `setup.py`, at `kernels/cpu/setup.py`, builds C++ for the CPU).
Neither `.cu` file declares a `PYBIND11_MODULE` or `TORCH_LIBRARY` block, so there is no symbol for Python to import.
And no file in the repository, notebooks and README included, mentions `forward_kernel`, `race_fused_fwd`, or `kernels/gpu` at all.

**The CPU extension is dead too.**
`kernels/cpu/race_ext.py:23` and `:31` pass `sources=[""]` to `torch.utils.cpp_extension.load`, so the C++ never builds either.
Its call site in `misc/race.py:114-133` is commented out regardless.

**The real path is PyTorch `cumsum` plus `bmm`.**
Every model runs `BatchedACE.forward` in `misc/race.py:63-142`, reached from `misc/lm.py:16` via `RACEBlock` and `RACEAttention`.
The expensive lines are `misc/race.py:108-110`:

```python
A_pref = probsK.cumsum(dim=1)                                                 # [N, T, L, R]
B_pref = (probsK.unsqueeze(-1) * V2.unsqueeze(2).unsqueeze(3)).cumsum(dim=1)  # [N, T, L, R, d_k]
E_pref = B_pref.div(A_pref.unsqueeze(-1).add(1e-6))                           # [N, T, L, R, d_k]
```

`B_pref` is five-dimensional, `[N, T, L, R, d_k]`.
The broadcast multiply materializes the whole thing before `cumsum` even starts, `E_pref` makes a second copy, and the `.contiguous()` at `misc/race.py:137` makes a third.

**Nothing ever needs the whole table.**
The only consumer is the lookup at `misc/race.py:135-138`, which contracts one timestep of `E_pref` against one timestep of `probsQ` and discards the rest:

```python
out2 = torch.bmm(
    probsQ.view(N*T, 1, S),                      # [N*T, 1, S]
    E_pref.contiguous().view(N*T, S, dk)         # [N*T, S, d_k]
).view(N, T, dk)
```

So the `T` axis of that tensor is pure overhead.
A kernel that carries the running sums `A` and `B` in registers and emits each output before advancing can drop it entirely.
The unused `race_fused_fwd_cuda` at `kernels/gpu/forward_kernel.cu:19-62` already sketches exactly this, which is a useful starting point.

One caveat worth stating: this is specific to the causal path.
The non-causal variants used for classification and vision reduce over the full sequence in one step (`scaling/benchmark_time.py:155-162`) and are already memory-light.

## Baseline results

Measured on a single NVIDIA A100-SXM4-40GB (42.4 GB reported, compute capability 8.0) with torch 2.11.0+cu128.
Configuration `M=1, B=1, H=8, d_k=64, K=4, L=4`, giving `N=8` streams and `S=64` buckets.
Forward pass only.
The notebook with these outputs is [`benchmarks/phase1_baseline.ipynb`](benchmarks/phase1_baseline.ipynb).

At this configuration one timestep of `B_pref` costs `N * S * d_k * 4` bytes, so that one tensor grows by **128 KB per token**.
That figure is arithmetic on the config, not a measurement; the measured number is below.

```
        T |    fwd ms |  peak GB |  M tok/s
--------------------------------------------
      512 |      1.06 |     0.15 |     3.88
     1024 |      1.69 |     0.30 |     4.86
     2048 |      3.03 |     0.59 |     5.41
     4096 |      6.59 |     1.17 |     4.98
     8192 |     12.57 |     2.33 |     5.21
    16384 |     24.97 |     4.66 |     5.25
    32768 |     49.96 |     9.30 |     5.25
    65536 |     99.88 |    18.60 |     5.25
   131072 |  OUT OF MEMORY  <-- this is the baseline's wall
```

Three things to take from this.

Peak memory is almost perfectly linear in `T`, doubling with every doubling of the sequence.
Dividing peak by `T` gives 277 KB per token at T=4096 and still 277 KB per token at T=65536, so the constant is genuinely flat across a 16x range and there is no quadratic term.
RACE's linear-time claim holds; the problem is the size of that constant.

Worth noting that 277 KB is about 2.17x the 128 KB that a single `B_pref` costs, and the ratio holds to three significant figures at every length measured.
`B_pref` alone does not explain the peak, but a small fixed number of same-shape tensors alive at once does, which is what the code does: the broadcast product, the `cumsum` result, `E_pref`, and the `.contiguous()` copy.
This is consistent with the trace, though attributing the exact 2.17 to specific allocations would need a memory profile that has not been run yet.

Throughput plateaus at about **5.25 M tokens/s** from T=16384 onward and does not improve with more work in flight.
A compute-bound kernel would usually keep climbing as the sequence gets long enough to saturate the GPU.
Flattening this early is the signature of a memory-bound loop, which is consistent with a hot path whose main activity is writing and re-reading a large scratch tensor.

The wall is at **T=131072**.
The last sequence length that fits is 65536, at 18.60 GB peak.
Doubling to 131072 would need roughly 37 GB for the prefix tensors alone on a 40 GB card, and it fails.

These numbers are the "before" picture.
Every later performance claim in this repository will be measured against them on the same hardware.

## Roadmap

**Phase 1 - baseline measurement.**
Done.
Reproduce the reference causal forward faithfully, sweep sequence length, and record time, peak memory, and throughput up to the OOM point.
That is the table above.

**Phase 2 - memory-light PyTorch reference.**
Done.
`src/race_chunked.py` carries the running `A` and `B` state across chunks and never builds `B_pref`.
`src/test_equivalence.py` checks it against a faithful copy of the reference (`src/race_baseline.py`): max difference about 1e-7 in fp32, 2e-15 in fp64, and bit-identical when the chunk is the whole sequence.
`src/bench_cpu_memory.py` shows the reference growing by about 267 KB per token in peak RSS while the chunked version stays flat.

**Phase 3 - fused CUDA kernel.**
Written, not yet run on a GPU.
There are two causal kernels.
`src/race_fused_fwd.cu` is the direct prototype: one block per stream, bucket state in shared memory, bf16 in and out, fp32 accumulation, no atomics.
It removes the prefix tensor but walks each stream sequentially, so at the benchmarked config it keeps 8 of an A100's 108 SMs busy.
`src/causal_v2/` is the parallel version: tiles are summed by the non-causal bucket build, a per-stream scan turns tile sums into prefix states, and an output kernel per tile handles the causal part in 64-token sub-chunks.
The design with the arithmetic behind the tile and sub-chunk sizes is in `docs/causal_v2_design.md`.

**Non-causal forward and backward.**
Written, not yet run on a GPU.
`src/noncausal/` implements the paper's Algorithm 1 as three kernels (per-tile bucket build, fixed-order tree reduce, query pass) plus a five-launch backward with an `autograd.Function` wrapper.
The corner probabilities use the exact product form φᵣ = ∏ₜ [pₜ or 1 − pₜ] with pₜ = σ(2βuₜ), so there is no length-R softmax anywhere.
Every reduction is a fixed-shape tree, which makes forward and backward bitwise reproducible run to run.
The backward derivation is in `src/noncausal/BACKWARD_DERIVATION.md` and matches fp64 autograd to about 1e-14.
The design is in `docs/noncausal_design.md`.

## Status

Everything under `src/` compiles for sm_80, sm_89 and sm_90 with no register spills, and the CPU test suites pass: fp64 references, emulations of each kernel's exact summation order, and the derivation checks.
What has not happened yet is a GPU run.
The GPU tests are written and skip without CUDA; `infra/gpu_session.sh` runs the whole sequence (build, compute-sanitizer, tests, benchmarks, ncu) and collects the logs.
Until that run exists, every performance statement in this repository is the Phase 1 baseline above, and the kernels' correctness rests on the CPU checks.
Numbers will replace this paragraph when they are measured.

## Credit

The RACE Attention method and reference implementation are by Sahil Joshi, Agniva Chowdhury, Amar Kanakamedala, Ekam Singh, Evan Tu, and Anshumali Shrivastava.

- Paper: [RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](https://openreview.net/forum?id=RR8Lh8RHgA), ICLR 2026
- Code: [github.com/sahiljoshi515/RACE_Attention](https://github.com/sahiljoshi515/RACE_Attention)

All baselines here were measured against that implementation, and the benchmark cell reproduces its causal `BatchedACE.forward` unchanged so the comparison stays fair.
This repository is independent work and is not affiliated with or endorsed by the RACE Attention authors.

## Layout

```
analysis/hot-path.md              trace of the reference implementation, with file:line citations
benchmarks/phase1_baseline.ipynb  Phase 1 sweep, with real A100 outputs preserved
docs/                             design documents for the non-causal kernels and the causal v2 kernel
src/race_baseline.py              faithful copy of the reference causal forward
src/race_chunked.py               Phase 2 chunked forward, plus its tests and memory benchmark
src/race_fused_fwd.cu             Phase 3 prototype: one fused causal kernel, one block per stream
src/regime_analysis.py            FLOPs and bytes per token, roofline placement
src/verify_factorization.py       numerical check of the Bernoulli product form
src/noncausal/                    non-causal forward and backward: reference, kernels, binding, tests, benchmarks
src/causal_v2/                    chunk-parallel causal forward: reference, kernels, binding, tests, benchmark
infra/                            GPU session runbook and sanitizer cases
```

## License

MIT.
See [LICENSE](LICENSE).
