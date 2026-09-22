"""Simulate src/race_fused_fwd.cu's math in Python and compare to the reference.

This cannot catch CUDA mechanics (sync placement, indexing arithmetic bugs),
but it does catch algorithm bugs: the corner-sign logit formula replacing the
protos_T matmul, softmax placement, inclusive-update ordering, and the
planes layout handed to the kernel ([L*K, dk] row-major).
"""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from race_baseline import BatchedACE

CFG = dict(M=1, B=1, H=8, d_k=64, K=4, L=4)


def simulate_kernel(ace, K2, Q2, V2):
    """Mirror the .cu control flow: per token, per lane, corner-sign logits."""
    N, T, dk = K2.shape
    L, KB, R = ace.L, ace.K, ace.R
    scale = math.sqrt(dk)
    planes = ace.planes_T.T.contiguous()  # [L*K, dk], what the harness passes
    signs = torch.tensor(
        [[1.0 if (r >> (KB - 1 - k)) & 1 else -1.0 for k in range(KB)] for r in range(R)]
    )  # [R, K] - the kernel's bit trick

    A = torch.zeros(N, L * R)
    B = torch.zeros(N, L * R, dk)
    out = torch.empty_like(V2)
    for t in range(T):
        th_k = torch.tanh(K2[:, t] @ planes.T).div(scale).view(N, L, KB)
        th_q = torch.tanh(Q2[:, t] @ planes.T).div(scale).view(N, L, KB)
        pk = torch.softmax(th_k @ signs.T, dim=-1).reshape(N, L * R)
        pq = torch.softmax(th_q @ signs.T, dim=-1).reshape(N, L * R)
        A += pk
        B += pk.unsqueeze(-1) * V2[:, t].unsqueeze(1)
        out[:, t] = (pq.unsqueeze(-1) * B / (A.unsqueeze(-1) + 1e-6)).sum(dim=1)
    return out


def main():
    torch.manual_seed(1234)
    ace = BatchedACE(CFG["d_k"], CFG["K"], CFG["L"], CFG["M"])
    T = 256
    g = torch.Generator().manual_seed(0)
    shape = (CFG["M"], CFG["B"], T, CFG["H"], CFG["d_k"])
    Khf, Vhf, Qhf = [torch.randn(shape, generator=g) for _ in range(3)]
    with torch.no_grad():
        ref = ace(Khf, Vhf, Qhf)

    def streams(Z):
        M, Bb, T, H, dk = Z.shape
        return Z.permute(0, 1, 3, 2, 4).contiguous().view(M * Bb * H, T, dk)

    got = simulate_kernel(ace, streams(Khf), streams(Qhf), streams(Vhf))
    diff = (got - streams(ref)).abs().max().item()
    print(f"kernel-math simulation vs reference: max|diff|={diff:.3e} ->",
          "PASS" if diff < 1e-4 else "FAIL")
    sys.exit(0 if diff < 1e-4 else 1)


if __name__ == "__main__":
    main()
