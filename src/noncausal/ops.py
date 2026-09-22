"""Differentiable non-causal RACE Attention on the CUDA kernels.

    from ops import race_attention
    o = race_attention(q, k, v, W, beta)   # q, k, v: [B, H, N, d] bf16 CUDA
    o.float().sum().backward()             # fills q.grad, k.grad, v.grad, beta.grad

W ([L, P, d] fp32) is a fixed random buffer and never receives a gradient.
beta is a one-element tensor (typically an nn.Parameter) or a Python float.
The math, the saved tensors and the gradient formulas are in
BACKWARD_DERIVATION.md; shapes and limits are those of the forward (README.md).
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from build import load_extension


class RaceAttentionFunction(torch.autograd.Function):
    """o = race(q, k, v, W, beta) with the hand-written backward kernels.

    Saved for backward: q, k, v, W, beta and the reduced bucket sums A, B
    (L * R * (d + 1) fp32 per head, the forward's workspace tile 0). The
    output O and the per-token denominators are not saved: the backward
    recomputes Den from A and never needs O (see BACKWARD_DERIVATION.md).
    """

    @staticmethod
    def forward(ctx, q, k, v, W, beta):
        out, bucket_totals = load_extension(verbose=False).forward_train(q, k, v, W, beta)
        ctx.save_for_backward(q, k, v, W, beta, bucket_totals)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        # once_differentiable: the backward is itself a kernel on saved
        # intermediates, not a composition of differentiable ops, so a second
        # derivative through it would be silently wrong. With the decorator,
        # double backward raises instead.
        needs_q, needs_k, needs_v, _, needs_beta = ctx.needs_input_grad
        if not (needs_q or needs_k or needs_v or needs_beta):
            return None, None, None, None, None
        q, k, v, W, beta, bucket_totals = ctx.saved_tensors
        # Autograd may hand over a non-contiguous or stride-0 gradient (for
        # example the expanded ones of o.sum()); the kernels need dense rows.
        grad_out = grad_out.to(q.dtype).contiguous()
        dq, dk, dv, dbeta = load_extension(verbose=False).backward(
            grad_out, q, k, v, W, beta, bucket_totals
        )
        # All four gradients come out of the same five launches (dq and the
        # per-token weights feed dk, dv and d beta), so unneeded ones are only
        # dropped here.
        return (
            dq if needs_q else None,
            dk if needs_k else None,
            dv if needs_v else None,
            None,  # W is fixed: no dW
            dbeta.to(device=beta.device, dtype=beta.dtype).reshape(beta.shape)
            if needs_beta
            else None,
        )


def race_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: torch.Tensor,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    """Non-causal RACE Attention (paper Algorithm 1), differentiable in q, k, v, beta.

    Args:
        q, k, v: [B, H, N, d] bf16 CUDA tensors, contiguous, identical shapes.
        W: [L, P, d] fp32 fixed random planes on the same device; must not
            require grad (there is no dW).
        beta: one-element floating tensor on any device, or a float.

    Returns:
        o: [B, H, N, d] bf16.
    """
    if W.requires_grad:
        raise ValueError("W is a fixed random buffer and has no gradient; pass W.detach()")
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(float(beta), dtype=torch.float32)
    return RaceAttentionFunction.apply(q, k, v, W, beta)
