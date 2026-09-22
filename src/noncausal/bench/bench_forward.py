"""Timing of the non-causal RACE forward kernels against the HBM roofline.

For each configuration: 5 warmup runs, then 20 timed runs with CUDA events;
the median is reported. Bytes moved counts the unavoidable traffic only:
Q, K, V read once and O written once, all bf16 (4 * BH * N * d * 2 bytes).
The workspace round trip (O(N / 2048 * R * d) floats) and repeated K/V
reads served from L2 are excluded, so the % of peak is a lower bound on how
close the kernels are to the memory floor.

Usage: python bench/bench_forward.py [--bh 4] [--min-log2 14] [--max-log2 20]
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from build import load_extension  # noqa: E402

WARMUP_RUNS = 5
TIMED_RUNS = 20
HEAD_DIM = 128
CONFIGS = [(2, 2), (4, 4)]  # (P, L)

# Peak HBM bandwidth in GB/s by device-name substring, first match wins
# (more specific names first, e.g. "L40S" before "L4").
PEAK_BANDWIDTH_GBPS = [
    ("H100 NVL", 3900.0),
    ("H100 PCIe", 2000.0),
    ("H100", 3350.0),
    ("H200", 4800.0),
    ("A100-SXM4-80GB", 2039.0),
    ("A100 80GB PCIe", 1935.0),
    ("A100", 1555.0),
    ("L40S", 864.0),
    ("L40", 864.0),
    ("L4", 300.0),
]


def peak_bandwidth_gbps(device_name: str) -> float | None:
    for key, bandwidth in PEAK_BANDWIDTH_GBPS:
        if key in device_name:
            return bandwidth
    return None


def median_ms(fn, *args, warmup: int = WARMUP_RUNS, runs: int = TIMED_RUNS) -> float:
    """Median wall time of fn(*args) on the current stream, in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bh", type=int, default=4, help="batch * heads")
    parser.add_argument("--min-log2", type=int, default=14)
    parser.add_argument("--max-log2", type=int, default=20)
    args = parser.parse_args()

    race = load_extension(verbose=False)
    device_name = torch.cuda.get_device_name()
    peak = peak_bandwidth_gbps(device_name)
    print(f"device: {device_name}, peak HBM: {f'{peak:.0f} GB/s' if peak else 'unknown'}")
    print(f"d={HEAD_DIM}, B*H={args.bh}, median of {TIMED_RUNS} runs after {WARMUP_RUNS} warmups")
    header = f"{'P':>2} {'L':>2} {'N':>9} {'ms':>9} {'MB moved':>10} {'GB/s':>8} {'% peak':>7}"
    print(header)
    print("-" * len(header))

    gen = torch.Generator(device="cuda").manual_seed(0)
    for num_planes, num_tables in CONFIGS:
        W = torch.randn(num_tables, num_planes, HEAD_DIM, device="cuda", generator=gen)
        beta = torch.tensor(1.0, device="cuda")
        for log2_n in range(args.min_log2, args.max_log2 + 1):
            seq_len = 1 << log2_n
            shape = (1, args.bh, seq_len, HEAD_DIM)
            q, k, v = (
                torch.randn(shape, device="cuda", generator=gen, dtype=torch.bfloat16)
                for _ in range(3)
            )
            ms = median_ms(race.forward, q, k, v, W, beta)
            bytes_moved = 4 * args.bh * seq_len * HEAD_DIM * 2
            gbps = bytes_moved / (ms * 1e-3) / 1e9
            percent = f"{100 * gbps / peak:6.1f}%" if peak else "    n/a"
            print(
                f"{num_planes:>2} {num_tables:>2} {seq_len:>9} {ms:>9.3f} "
                f"{bytes_moved / 1e6:>10.1f} {gbps:>8.1f} {percent:>7}"
            )
            del q, k, v
        print()


if __name__ == "__main__":
    main()
