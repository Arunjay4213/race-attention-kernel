"""Phase 2: memory-light causal RACE forward (chunked scan), same math as the baseline.

Key idea: the baseline materializes the whole prefix history
    B_pref[N, T, L, R, d_k]
even though position t only ever needs the running totals *at* t.
A scan needs O(state) memory, not O(T * state).

We process the sequence in chunks of C tokens. Across chunks we carry only
    A_state: [N, L, R]        running sum of key bucket probs
    B_state: [N, L, R, d_k]   running sum of prob * value
and inside a chunk we do the same cumsum the baseline does, but on C tokens
instead of T. Peak extra memory is O(N * C * L * R * d_k), independent of T.

The per-position math is identical to the baseline (same accumulation order:
cumsum within a chunk on top of the carried totals equals the global cumsum
up to float rounding), so outputs match to ~1e-5 in fp32 and ~1e-12 in fp64.
"""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def chunked_forward(ace, Khf, Vhf, Qhf, chunk=4096):
    """Same signature/semantics as BatchedACE.forward, using ace's buffers.

    ace: a race_baseline.BatchedACE (we reuse its planes_T / protos_T so the
         comparison is exact, not seed-dependent).
    """
    M, B, T, H, dk = Khf.shape
    L, R = ace.L, ace.R
    S = L * R
    scale = math.sqrt(dk)
    N = M * B * H

    Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
    Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
    V2 = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

    A_state = torch.zeros(N, 1, L, R, dtype=Khf.dtype, device=Khf.device)
    B_state = torch.zeros(N, 1, L, R, dk, dtype=Khf.dtype, device=Khf.device)
    out = torch.empty(N, T, dk, dtype=Khf.dtype, device=Khf.device)

    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        C = t1 - t0
        projK = (Kh2[:, t0:t1] @ ace.planes_T).view(N, C, L, ace.K)
        projQ = (Qh2[:, t0:t1] @ ace.planes_T).view(N, C, L, ace.K)
        probsK = F.softmax(projK.tanh().div(scale) @ ace.protos_T, dim=-1)  # [N,C,L,R]
        probsQ = F.softmax(projQ.tanh().div(scale) @ ace.protos_T, dim=-1)

        Vc = V2[:, t0:t1]  # [N,C,dk]
        A_pref = A_state + probsK.cumsum(dim=1)  # [N,C,L,R]
        B_pref = B_state + (probsK.unsqueeze(-1) * Vc.unsqueeze(2).unsqueeze(3)).cumsum(dim=1)
        E_pref = B_pref.div(A_pref.unsqueeze(-1).add(1e-6))  # [N,C,L,R,dk]

        out[:, t0:t1] = torch.bmm(
            probsQ.reshape(N * C, 1, S), E_pref.reshape(N * C, S, dk)
        ).view(N, C, dk)

        A_state = A_pref[:, -1:].clone()
        B_state = B_pref[:, -1:].clone()

    return out.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)
