"""Timing of the non-causal RACE backward kernels against the HBM roofline.

For each configuration, the median over 20 timed runs (after 5 warmups, CUDA
events) of two things:
  - backward alone: race.backward on saved bucket sums, the five backward
    launches (query grad, weighted bucket build, tree reduce, key grad,
    beta reduce),
  - forward + backward: race.forward_train followed by race.backward, one
    training step of the op.
Bytes moved counts the unavoidable traffic only, all bf16:
  backward: q, k, v, dO read once, dq, dk, dv written once (7 * BH * N * d * 2)
  forward + backward: that plus q, k, v read and O written (11 * BH * N * d * 2)
The second read of q and dO by the weighted bucket build, the workspace round
trips (O(N / 2048 * R * d) floats) and the 8-byte per-token weights are
excluded, so % of peak is a lower bound on how close the kernels are to the
memory floor.

Usage: python bench/bench_backward.py [--bh 4] [--min-log2 14] [--max-log2 20]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench_forward import (  # noqa: E402
    CONFIGS,
    HEAD_DIM,
    TIMED_RUNS,
    WARMUP_RUNS,
    median_ms,
    peak_bandwidth_gbps,
)
from build import load_extension  # noqa: E402

BACKWARD_ROWS = 7  # q, k, v, dO read; dq, dk, dv written
TRAINING_STEP_ROWS = 11  # plus q, k, v read and O written by the forward


def training_step(race, grad_o, q, k, v, W, beta):
    """One forward + backward of the op, as autograd would run it."""
    _, bucket_totals = race.forward_train(q, k, v, W, beta)
    return race.backward(grad_o, q, k, v, W, beta, bucket_totals)


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
    header = (
        f"{'P':>2} {'L':>2} {'N':>9} {'bwd ms':>9} {'GB/s':>8} {'% peak':>7}"
        f" {'fwd+bwd ms':>11} {'GB/s':>8} {'% peak':>7}"
    )
    print(header)
    print("-" * len(header))

    def percent(gbps: float) -> str:
        return f"{100 * gbps / peak:6.1f}%" if peak else "    n/a"

    gen = torch.Generator(device="cuda").manual_seed(0)
    for num_planes, num_tables in CONFIGS:
        W = torch.randn(num_tables, num_planes, HEAD_DIM, device="cuda", generator=gen)
        beta = torch.tensor(1.0, device="cuda")
        for log2_n in range(args.min_log2, args.max_log2 + 1):
            seq_len = 1 << log2_n
            shape = (1, args.bh, seq_len, HEAD_DIM)
            q, k, v, grad_o = (
                torch.randn(shape, device="cuda", generator=gen, dtype=torch.bfloat16)
                for _ in range(4)
            )
            _, bucket_totals = race.forward_train(q, k, v, W, beta)
            backward_ms = median_ms(race.backward, grad_o, q, k, v, W, beta, bucket_totals)
            step_ms = median_ms(training_step, race, grad_o, q, k, v, W, beta)
            row_bytes = args.bh * seq_len * HEAD_DIM * 2
            backward_gbps = BACKWARD_ROWS * row_bytes / (backward_ms * 1e-3) / 1e9
            step_gbps = TRAINING_STEP_ROWS * row_bytes / (step_ms * 1e-3) / 1e9
            print(
                f"{num_planes:>2} {num_tables:>2} {seq_len:>9} {backward_ms:>9.3f} "
                f"{backward_gbps:>8.1f} {percent(backward_gbps):>7} {step_ms:>11.3f} "
                f"{step_gbps:>8.1f} {percent(step_gbps):>7}"
            )
            del q, k, v, grad_o, bucket_totals
        print()


if __name__ == "__main__":
    main()
