"""Timing of the causal RACE v2a forward against the HBM roofline and the PyTorch chunked forward.

For each configuration and sequence length T (B * H = 8 streams):
  - v2a: 5 warmup runs, then 20 timed runs with CUDA events; the median.
  - baseline: src/race_chunked.py chunked_forward (fp32, 4096-token chunks),
    1 warmup and 3 timed runs because it takes seconds at long T. It computes
    the repo's per-bucket normalization, not Algorithm 1, so it is a timing
    baseline only; its outputs are not compared.

Bytes per token per stream (plan section 9):
  - model   = 12 d (K1 reads K and V; K3 reads Q, K, V and writes O; all bf16)
              + 16 S (d + 1) / T_blk (the tile state: K1 write, K2 read and
              write, K3 read, fp32)
  - minimum = 8 d (read Q, K, V once and write O once), the single-pass view
GB/s and % of peak use the model bytes; "min GB/s" uses the minimum bytes.
Peak memory is torch.cuda.max_memory_allocated over the timed runs,
including the inputs, so it states what a T-token forward needs.

Usage: python bench/bench_causal.py [--min-log2 14] [--max-log2 21]
       [--baseline-max-log2 21] [--skip-baseline]
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # src/causal_v2
sys.path.insert(1, str(HERE.parents[2]))  # src
from build import load_extension  # noqa: E402
from race_baseline import BatchedACE  # noqa: E402
from race_chunked import chunked_forward  # noqa: E402

WARMUP_RUNS = 5
TIMED_RUNS = 20
BASELINE_WARMUP_RUNS = 1
BASELINE_TIMED_RUNS = 3
BASELINE_CHUNK = 4096
STREAMS = 8  # batch 1, 8 heads
CONFIGS = [(64, 4, 4), (128, 4, 4)]  # (d, P, L); the first is the benchmarked layer

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


def median_ms(fn, warmup: int, runs: int) -> float:
    """Median wall time of fn() on the current stream, in milliseconds."""
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


def model_bytes(seq_len: int, head_dim: int, num_planes: int, num_tables: int, tile_tokens: int) -> int:
    num_tiles = -(-seq_len // tile_tokens)
    features = num_tables << num_planes
    token_bytes = STREAMS * seq_len * 12 * head_dim
    state_bytes = STREAMS * num_tiles * 16 * features * (head_dim + 1)
    return token_bytes + state_bytes


def time_v2a(race, q, k, v, W, beta) -> tuple[float, float]:
    """(median ms, peak GiB) of the v2a forward."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = median_ms(lambda: race.forward(q, k, v, W, beta), WARMUP_RUNS, TIMED_RUNS)
    return ms, torch.cuda.max_memory_allocated() / 2**30


def time_baseline(q, k, v, W) -> tuple[float, float] | None:
    """(median ms, peak GiB) of the PyTorch chunked forward in fp32, or None on OOM."""
    num_tables, num_planes, head_dim = W.shape
    ace = BatchedACE(head_dim, num_planes, num_tables, 1, device="cuda")
    ace.planes_T = W.reshape(num_tables * num_planes, head_dim).T.contiguous()

    def to_repo(x: torch.Tensor) -> torch.Tensor:  # [B, H, T, d] bf16 -> [1, B, T, H, d] fp32
        return x.float().permute(0, 2, 1, 3).unsqueeze(0).contiguous()

    try:
        keys, values, queries = to_repo(k), to_repo(v), to_repo(q)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ms = median_ms(
            lambda: chunked_forward(ace, keys, values, queries, chunk=BASELINE_CHUNK),
            BASELINE_WARMUP_RUNS, BASELINE_TIMED_RUNS,
        )
        return ms, torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-log2", type=int, default=14)
    parser.add_argument("--max-log2", type=int, default=21)
    parser.add_argument("--baseline-max-log2", type=int, default=21)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    race = load_extension(verbose=False)
    device_name = torch.cuda.get_device_name()
    peak = peak_bandwidth_gbps(device_name)
    print(f"device: {device_name}, peak HBM: {f'{peak:.0f} GB/s' if peak else 'unknown'}")
    print(f"B*H={STREAMS}; v2a median of {TIMED_RUNS} after {WARMUP_RUNS} warmups; "
          f"baseline (race_chunked, fp32) median of {BASELINE_TIMED_RUNS} after {BASELINE_WARMUP_RUNS}")
    header = (f"{'d':>3} {'P':>2} {'L':>2} {'T':>8} {'T_blk':>5} {'ms':>9} {'Mtok/s':>8} {'GB/s':>7} "
              f"{'% peak':>7} {'min GB/s':>8} {'peak GiB':>8} {'base ms':>9} {'base GiB':>8} {'speedup':>8}")
    print(header)
    print("-" * len(header))

    gen = torch.Generator(device="cuda").manual_seed(0)
    for head_dim, num_planes, num_tables in CONFIGS:
        if not race.fits_on_device(head_dim, num_planes, num_tables):
            print(f"{head_dim:>3} {num_planes:>2} {num_tables:>2}  does not fit this GPU's shared memory")
            continue
        W = torch.randn(num_tables, num_planes, head_dim, device="cuda", generator=gen)
        beta = torch.tensor(head_dim**-0.5, device="cuda")  # the repo's frozen beta
        for log2_t in range(args.min_log2, args.max_log2 + 1):
            seq_len = 1 << log2_t
            shape = (1, STREAMS, seq_len, head_dim)
            q, k, v = (torch.randn(shape, device="cuda", generator=gen, dtype=torch.bfloat16)
                       for _ in range(3))
            tile_tokens = race.select_tile_tokens(STREAMS, seq_len, head_dim, num_planes, num_tables)
            ms, peak_gib = time_v2a(race, q, k, v, W, beta)
            bytes_model = model_bytes(seq_len, head_dim, num_planes, num_tables, tile_tokens)
            gbps = bytes_model / (ms * 1e-3) / 1e9
            min_gbps = STREAMS * seq_len * 8 * head_dim / (ms * 1e-3) / 1e9
            percent = f"{100 * gbps / peak:6.1f}%" if peak else "    n/a"
            baseline = "skipped"
            if not args.skip_baseline and log2_t <= args.baseline_max_log2:
                result = time_baseline(q, k, v, W)
                baseline = "OOM" if result is None else result
            if isinstance(baseline, tuple):
                base_ms, base_gib = baseline
                base_cols = f"{base_ms:>9.2f} {base_gib:>8.2f} {base_ms / ms:>7.1f}x"
            else:
                base_cols = f"{baseline:>9} {'':>8} {'':>8}"
            print(f"{head_dim:>3} {num_planes:>2} {num_tables:>2} {seq_len:>8} {tile_tokens:>5} {ms:>9.3f} "
                  f"{STREAMS * seq_len / ms / 1e3:>8.1f} {gbps:>7.1f} {percent:>7} {min_gbps:>8.1f} "
                  f"{peak_gib:>8.2f} {base_cols}")
            del q, k, v
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
