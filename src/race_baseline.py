"""Faithful copy of the reference causal RACE forward (repo/misc/race.py, class BatchedACE).

This is the path the released models actually run. It materializes two 5-D
tensors per call:
    B_pref: [N, T, L, R, d_k]   running sum of prob * value  (cumsum over T)
    E_pref: [N, T, L, R, d_k]   B_pref / (A_pref + eps)
which is what makes memory grow ~O(T * L * R * d_k) and OOM at long T.

Kept byte-for-byte identical in math so the optimized versions have a fair
reference. Only the unused non-shared-planes branch and the commented-out CPU
extension path are dropped.
"""
import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn


class BatchedACE(nn.Module):
    def __init__(self, d_k, K, L, M, device="cpu", share_planes=True):
        super().__init__()
        assert share_planes, "reference config uses shared planes"
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        planes = torch.randn(L, K, d_k, device=device)
        self.register_buffer("planes_T", planes.view(L * K, d_k).T)  # [d_k, L*K]
        corners = torch.tensor(
            list(itertools.product([-1.0, 1.0], repeat=K)), device=device
        )
        self.register_buffer("protos_T", corners.T)  # [K, R]

    def forward(self, Khf, Vhf, Qhf):
        M, B, T, H, dk = Khf.shape
        scale = math.sqrt(dk)
        N = M * B * H
        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
        V2 = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
        projK = (Kh2 @ self.planes_T).view(N, T, self.L, self.K)
        projQ = (Qh2 @ self.planes_T).view(N, T, self.L, self.K)
        probsK = F.softmax(projK.tanh().div(scale) @ self.protos_T, dim=-1)  # [N,T,L,R]
        probsQ = F.softmax(projQ.tanh().div(scale) @ self.protos_T, dim=-1)
        A_pref = probsK.cumsum(dim=1)  # [N,T,L,R]
        B_pref = (probsK.unsqueeze(-1) * V2.unsqueeze(2).unsqueeze(3)).cumsum(dim=1)
        E_pref = B_pref.div(A_pref.unsqueeze(-1).add(1e-6))  # [N,T,L,R,d_k]
        S = self.L * self.R
        out2 = torch.bmm(
            probsQ.view(N * T, 1, S), E_pref.contiguous().view(N * T, S, dk)
        ).view(N, T, dk)
        return out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)
