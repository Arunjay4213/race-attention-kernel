"""Step 1: roofline regime analysis for the non-causal RACE forward kernel.

Conventions:
- "per token per head" counts one query row AND one key/value row (each position
  is both), M = 1 ensemble.
- 1 MAC = 2 FLOPs. tanh/sigmoid counted separately as SFU ops (they run on the
  special-function units, not the FMA pipes).
- Bytes assume Q, K, V read from HBM exactly once and O written once; the
  R-space intermediates (phi, A, B) never touch HBM. bf16 I/O.
"""
GPUS = {
    "A100-SXM-40GB (sm_80)": dict(bw=1.555e12, fp32=19.5e12, tc_bf16=312e12, sms=108),
    "H100-SXM-80GB (sm_90)": dict(bw=3.35e12, fp32=66.9e12, tc_bf16=989e12, sms=132),
}
BYTES_ELEM = 2  # bf16 I/O


def stages(P, L, d):
    R = 1 << P
    return {
        # both sides (query + key), L tables: L*P dots of length d each side
        "projection u=Wx (Q and K)": 4 * L * P * d,
        # factorized corner probs: product tree ~2R muls per table per side
        "corner probs (Bernoulli)": 2 * L * 2 * R,
        # key side: B[r,:] += phi_r * v  (R*d MACs), A[r] += phi_r
        "bucket aggregation A,B": 2 * L * R * d + L * R,
        # query side: Num += phi_r * B[r,:] (R*d MACs), Den (R MACs), final divide
        "query mixing Num,Den,O": 2 * L * R * d + 2 * L * R + d,
    }


def analyze(P, L, d):
    R = 1 << P
    fl = stages(P, L, d)
    total = sum(fl.values())
    sfu = 2 * L * P * 2  # tanh + sigmoid, both sides
    byt = 4 * d * BYTES_ELEM  # 3 reads + 1 write
    ai = total / byt
    print(f"\n=== P={P} L={L} M=1 d={d}  (R={R}) ===")
    for k, v in fl.items():
        print(f"  {k:28s} {v:7d} FLOPs/token/head")
    print(f"  {'TOTAL':28s} {total:7d} FLOPs/token/head  (+{sfu} SFU ops)")
    print(f"  bytes/token/head (bf16, QKV once + O once): {byt}")
    print(f"  arithmetic intensity: {ai:.1f} FLOP/byte")
    for name, g in GPUS.items():
        ridge_fp32 = g["fp32"] / g["bw"]
        ridge_tc = g["tc_bf16"] / g["bw"]
        bound = "COMPUTE-bound on fp32 CUDA cores" if ai > ridge_fp32 else "MEMORY-bound"
        print(f"  {name}: ridge fp32={ridge_fp32:.1f}, tensor-core bf16={ridge_tc:.0f}"
              f" FLOP/B -> {bound}"
              f"{' (memory-bound if GEMM stages use tensor cores)' if ai > ridge_fp32 and ai < ridge_tc else ''}")
    # wall-clock floor at N = 1M, 4 heads
    N, H = 1_000_000, 4
    tot_bytes = N * H * byt
    tot_flops = N * H * total
    print(f"  floors at N=1M, {H} heads ({tot_bytes/1e9:.2f} GB, {tot_flops/1e9:.0f} GFLOP):")
    for name, g in GPUS.items():
        t_mem = tot_bytes / g["bw"] * 1e3
        t_fp32 = tot_flops / g["fp32"] * 1e3
        t_tc = tot_flops / g["tc_bf16"] * 1e3
        print(f"    {name}: memory {t_mem:.2f} ms | fp32-core {t_fp32:.2f} ms | "
              f"tensor-core {t_tc:.2f} ms -> floor {max(t_mem, min(t_fp32, t_tc)):.2f} ms")


if __name__ == "__main__":
    analyze(2, 2, 128)
    analyze(4, 4, 128)
