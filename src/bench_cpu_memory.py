"""Measure peak RSS of baseline vs chunked forward at growing T (CPU).

Peak RSS is process-wide and monotonic, so each measurement runs in a fresh
subprocess (`--mode baseline|chunked --T <int>`); the parent collects results.

Expectation: baseline peak grows linearly with T (B_pref + E_pref, ~256 KB per
token at this config); chunked peak stays flat at the QKV-plus-one-chunk level.
"""
import argparse
import resource
import subprocess
import sys

HERE = __file__.rsplit("/", 1)[0]
CFG = dict(M=1, B=1, H=8, d_k=64, K=4, L=4)


def child(mode, T):
    import torch

    sys.path.insert(0, HERE)
    from race_baseline import BatchedACE
    from race_chunked import chunked_forward

    torch.manual_seed(1234)
    ace = BatchedACE(CFG["d_k"], CFG["K"], CFG["L"], CFG["M"])
    g = torch.Generator().manual_seed(0)
    shape = (CFG["M"], CFG["B"], T, CFG["H"], CFG["d_k"])
    K, V, Q = [torch.randn(shape, generator=g) for _ in range(3)]
    with torch.no_grad():
        if mode == "baseline":
            ace(K, V, Q)
        else:
            chunked_forward(ace, K, V, Q, chunk=1024)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"{peak_gb:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "chunked"])
    ap.add_argument("--T", type=int)
    args = ap.parse_args()
    if args.mode:
        child(args.mode, args.T)
        return

    Ts = [2048, 4096, 8192, 16384]
    print(f"{'T':>7} | {'baseline peak GB':>16} | {'chunked peak GB':>15}")
    print("-" * 46)
    for T in Ts:
        peaks = {}
        for mode in ("baseline", "chunked"):
            r = subprocess.run(
                [sys.executable, __file__, "--mode", mode, "--T", str(T)],
                capture_output=True, text=True,
            )
            peaks[mode] = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "OOM/ERR"
        print(f"{T:>7} | {peaks['baseline']:>16} | {peaks['chunked']:>15}")


if __name__ == "__main__":
    main()
