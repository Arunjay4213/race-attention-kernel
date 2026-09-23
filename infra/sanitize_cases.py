"""A handful of small forward calls for compute-sanitizer.

The full pytest suites are far too slow under memcheck/racecheck, so this runs
only the shapes that exercise the risky paths: tails that are not a multiple of
a tile or sub-chunk, the half-idle (d=64, P=1) instantiation, the largest
shared-memory configuration, and the multi-level reduce.
The non-causal extension runs forward_train and then backward (with v as the
output gradient) so the backward's per-warp scratch and __syncwarp ordering is
checked too, and then the same with the tensor-core forward
(tensor_cores=True), whose kernels stage every tile through shared memory
with cp.async and several barriers per stage. Each call is checked for
finiteness only; correctness is the test suites' job.

Usage: python infra/sanitize_cases.py [noncausal] [causal_v2]
With no arguments both extensions run; naming one runs only that one (and
builds only that one).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "src"


def load(package: str):
    """Imports src/<package>/build.py by path so the two build modules do not collide."""
    spec = importlib.util.spec_from_file_location(f"{package}_build", ROOT / package / "build.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / package))
    spec.loader.exec_module(module)
    return module.load_extension(verbose=False)


def inputs(bh: int, n: int, d: int, p: int, l: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    make = lambda: torch.randn(1, bh, n, d, device="cuda", generator=gen).to(torch.bfloat16)
    w = torch.randn(l, p, d, device="cuda", generator=gen)
    return make(), make(), make(), w


CASES = [
    # (B*H, N, d, P, L)
    (2, 2049, 128, 5, 4),   # tail past a full tile, largest smem configuration
    (1, 127, 64, 1, 1),     # half-idle build instantiation, N < tile
    (1, 17, 64, 2, 2),      # one token into a second staging batch
    (1, 65, 128, 4, 3),     # L = 3 (state size not a multiple of the pass size), sub-chunk tail
    (4, 20000, 64, 4, 4),   # multi-level reduce / several tiles per stream
    (1, 2049, 128, 4, 4),   # largest d = 128 causal footprint that fits a 99 KB opt-in GPU (C = 32), tile tail
    (1, 100, 64, 5, 4),     # largest causal footprint overall that fits 99 KB (84224 bytes, C = 32), sub-chunk tail
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("extensions", nargs="*", choices=["noncausal", "causal_v2"])
    selected = parser.parse_args().extensions or ["noncausal", "causal_v2"]
    noncausal = load("noncausal") if "noncausal" in selected else None
    causal = load("causal_v2") if "causal_v2" in selected else None
    beta = torch.tensor(1.0, device="cuda")
    for bh, n, d, p, l in CASES:
        q, k, v, w = inputs(bh, n, d, p, l, seed=n)
        if noncausal is not None:
            check_noncausal(noncausal, q, k, v, w, beta, (bh, n, d, p, l))
        if causal is not None:
            try:
                out = causal.forward(q, k, v, w, beta, 0)
            except RuntimeError as err:  # configuration refused on this GPU (shared memory)
                print(f"causal skipped {(bh, n, d, p, l)}: {str(err).splitlines()[0]}")
                continue
            torch.cuda.synchronize()
            assert torch.isfinite(out).all(), f"causal non-finite at {(bh, n, d, p, l)}"
        print(f"ok {(bh, n, d, p, l)}")


def check_noncausal(noncausal, q, k, v, w, beta, case) -> None:
    for tensor_cores in (False, True):
        label = "noncausal tc" if tensor_cores else "noncausal"
        out, bucket_totals = noncausal.forward_train(q, k, v, w, beta, tensor_cores=tensor_cores)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"{label} non-finite at {case}"
        grads = noncausal.backward(v, q, k, v, w, beta, bucket_totals)
        torch.cuda.synchronize()
        assert all(torch.isfinite(g).all() for g in grads), f"{label} backward non-finite at {case}"


if __name__ == "__main__":
    main()
