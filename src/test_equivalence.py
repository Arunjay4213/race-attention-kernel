"""Prove the chunked (memory-light) forward equals the reference forward.

Runs on CPU. Checks:
  1. fp32 max-abs diff at several (T, chunk) combos, including chunk sizes that
     do not divide T, chunk=1 (fully streaming) and chunk>=T (degenerate: must
     be bit-identical to the baseline).
  2. fp64: the diff should shrink to ~1e-12, showing the residual fp32 diff is
     accumulation-order rounding, not a math difference.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from race_baseline import BatchedACE
from race_chunked import chunked_forward

CFG = dict(M=1, B=1, H=8, d_k=64, K=4, L=4)  # same config as the phase-1 benchmark


def make_inputs(T, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (CFG["M"], CFG["B"], T, CFG["H"], CFG["d_k"])
    return [torch.randn(shape, generator=g, dtype=dtype) for _ in range(3)]


def run(dtype, cases):
    torch.manual_seed(1234)
    ace = BatchedACE(CFG["d_k"], CFG["K"], CFG["L"], CFG["M"]).to(dtype)
    failed = False
    for T, chunk, tol in cases:
        K, V, Q = make_inputs(T, dtype)
        with torch.no_grad():
            ref = ace(K, V, Q)
        got = chunked_forward(ace, K, V, Q, chunk=chunk)
        diff = (got - ref).abs().max().item()
        scale = ref.abs().max().item()
        ok = diff <= tol
        failed |= not ok
        print(
            f"{str(dtype):15s} T={T:5d} chunk={chunk:5d} "
            f"max|diff|={diff:.3e} (ref max {scale:.2f}, tol {tol:.0e}) "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
    return failed


def main():
    fp32_cases = [
        (512, 512, 1e-6),    # chunk == T: identical accumulation order
        (2048, 512, 1e-4),   # even split
        (2048, 300, 1e-4),   # chunk does not divide T
        (2048, 1, 1e-4),     # fully streaming, worst case for order effects
        (4096, 1024, 1e-4),
    ]
    fp64_cases = [
        (2048, 300, 1e-12),
        (2048, 1, 1e-12),
    ]
    failed = run(torch.float32, fp32_cases)
    failed |= run(torch.float64, fp64_cases)
    print()
    print("RESULT:", "FAIL" if failed else "ALL PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
