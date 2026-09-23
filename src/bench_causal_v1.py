"""Timing of the v1 fused causal RACE forward (src/race_fused_fwd.cu), the "before" number for causal v2.

v1 runs one CTA per stream and walks the T tokens of that stream in order
with the running state in shared memory, so its parallelism is the number of
streams, not the sequence length. It computes the repo's per-bucket
normalization (src/race_baseline.py, BatchedACE) with beta fixed at 1/sqrt(d),
and is compiled only for d = 64, P = 4, L <= 4.

Correctness: at small T the bf16 output is compared with BatchedACE in fp32 on
the same bf16-rounded inputs. The kernel rounds its output to bf16 once, so
the check allows 2^-8 relative to the output scale plus fp32 noise.

Timing: the benchmarked causal v2 configuration, B * H = 8, d = 64, P = 4,
L = 4, T in {2^14, 2^16, 2^18}; 2 warmup runs, then the median of 5 timed runs
with CUDA events. GB/s uses the minimum traffic of 8 d bytes per token per
stream (read Q, K, V and write O once in bf16), which is also v1's actual
traffic. Peak memory is torch.cuda.max_memory_allocated over the timed runs,
including the inputs.

Usage: python src/bench_causal_v1.py [--log2 14 16 18]
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from race_baseline import BatchedACE  # noqa: E402

HEAD_DIM, NUM_PLANES, NUM_TABLES, STREAMS = 64, 4, 4, 8
WARMUP_RUNS = 2
TIMED_RUNS = 5
CHECK_SEQ_LENS = (1, 17, 512, 2048)
# First substring match wins, so "L40S" must come before "L4".
PEAK_BANDWIDTH_GBPS = {"A10G": 600.0, "L40S": 864.0, "L4": 300.0, "A100": 1555.0, "H100": 3350.0}


def build_v1():
    # load_inline prepends torch/types.h to CUDA sources; the full extension
    # header is only needed by the generated pybind11 module.
    cuda_source = (HERE / "race_fused_fwd.cu").read_text().replace("#include <torch/extension.h>\n", "")
    declaration = ("torch::Tensor race_fused_fwd(torch::Tensor K, torch::Tensor Q, torch::Tensor V, "
                   "torch::Tensor planes, int64_t L);")
    return load_inline(name="race_fused_v1", cpp_sources=declaration, cuda_sources=cuda_source,
                       functions=["race_fused_fwd"], extra_cuda_cflags=["-O3"], verbose=False)


def median_ms(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def make_inputs(seq_len: int, gen: torch.Generator):
    """[N, T, d] bf16 streams for v1, as K, Q, V."""
    return [torch.randn(STREAMS, seq_len, HEAD_DIM, device="cuda", generator=gen, dtype=torch.bfloat16)
            for _ in range(3)]


def check_against_baseline(v1, ace: BatchedACE, planes_rows: torch.Tensor, gen: torch.Generator) -> bool:
    """Returns False if any length fails, so a broken kernel never reports timings as if valid."""
    all_ok = True
    for seq_len in CHECK_SEQ_LENS:
        keys, queries, values = make_inputs(seq_len, gen)
        # BatchedACE takes [M, B, T, H, d]; with M = B = 1 the stream index is H.
        to_repo = lambda x: x.float().permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            ref = ace(to_repo(keys), to_repo(values), to_repo(queries))  # [1, 1, T, N, d]
        ref = ref[0, 0].permute(1, 0, 2)
        out = v1.race_fused_fwd(keys, queries, values, planes_rows, NUM_TABLES).float()
        scale = ref.abs().max().item()
        error = (out - ref).abs().max().item()
        bound = 2.0**-8 * scale + 1e-5 * values.float().abs().max().item()
        status = "PASS" if error <= bound else "FAIL"
        all_ok = all_ok and error <= bound
        print(f"check T={seq_len:>5}: max|v1 - BatchedACE| = {error:.3e}, bound {bound:.3e} "
              f"(2^-8 of max|O| = {scale:.3f}) {status}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log2", type=int, nargs="+", default=[14, 16, 18])
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name()
    peak = next((bw for key, bw in PEAK_BANDWIDTH_GBPS.items() if key in device_name), None)
    print(f"device: {device_name}, peak HBM: {f'{peak:.0f} GB/s' if peak else 'unknown'}")
    v1 = build_v1()

    torch.manual_seed(0)
    ace = BatchedACE(HEAD_DIM, NUM_PLANES, NUM_TABLES, 1, device="cuda")
    planes_rows = ace.planes_T.T.contiguous().float()  # [L * P, d], v1's plane layout
    gen = torch.Generator(device="cuda").manual_seed(0)
    if not check_against_baseline(v1, ace, planes_rows, gen):
        sys.exit("v1 does not match BatchedACE; not timing a wrong kernel")

    print(f"B*H={STREAMS}, d={HEAD_DIM}, P={NUM_PLANES}, L={NUM_TABLES}; median of {TIMED_RUNS} after {WARMUP_RUNS} warmups")
    header = f"{'T':>8} {'ms':>10} {'Mtok/s':>8} {'GB/s':>7} {'% peak':>7} {'peak GiB':>8}"
    print(header)
    print("-" * len(header))
    for log2_t in args.log2:
        seq_len = 1 << log2_t
        keys, queries, values = make_inputs(seq_len, gen)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ms = median_ms(lambda: v1.race_fused_fwd(keys, queries, values, planes_rows, NUM_TABLES),
                       WARMUP_RUNS, TIMED_RUNS)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        gbps = STREAMS * seq_len * 8 * HEAD_DIM / (ms * 1e-3) / 1e9
        percent = f"{100 * gbps / peak:6.2f}%" if peak else "    n/a"
        print(f"{seq_len:>8} {ms:>10.2f} {STREAMS * seq_len / ms / 1e3:>8.2f} {gbps:>7.2f} {percent:>7} {peak_gib:>8.3f}")
        del keys, queries, values
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
