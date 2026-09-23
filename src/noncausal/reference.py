"""Reference implementation of non-causal RACE Attention (paper Algorithm 1).

This is the ground truth the CUDA kernels are tested against. It favours
clarity over speed, runs on CPU or GPU, works in fp32 or fp64, and is
differentiable end to end.

Math (arXiv 2510.04008, Algorithm 1, global normalization):

    For each table l = 1..L with fixed random planes W_l in R^{P x d}
    (rows ~ N(0, I_d), a buffer, never trained) and a trainable scalar beta:

        u_l(x)      = tanh(W_l x)                              in (-1, 1)^P
        phi_l(x)[r] = softmax_r( beta * <u_l(x), c_r> )        r = 0..R-1, R = 2^P

    where c_r in {-1, +1}^P are the hypercube corners. Keys build per-bucket
    sums, queries mix them:

        A_l[r]   = sum_j phi_l(k_j)[r]
        B_l[r,:] = sum_j phi_l(k_j)[r] * v_j
        Num_i    = (1/L) sum_l sum_r phi_l(q_i)[r] * B_l[r,:]
        Den_i    = (1/L) sum_l sum_r phi_l(q_i)[r] * A_l[r]
        O_i      = Num_i / Den_i

Placement of beta and scaling, confirmed against the paper and the authors'
code (repo/scaling/benchmark_time.py, repo/misc/race.py):
    - The paper writes phi = softmax(beta * tanh(W x)^T c_r) with W rows drawn
      from N(0, I_d): beta multiplies the corner logits, after the tanh.
    - The repo computes softmax((x @ planes).tanh().div(scale) @ corners) with
      planes = torch.randn(L, P, d), so beta = 1 / scale. The repo's scale is
      sqrt(d) in misc/race.py (beta = 1/sqrt(d) frozen) and exp(logit_temp)
      clamped to [1e-2, 10] in the scaling benchmark (beta in [0.1, 100]).
    - No 1/sqrt(d) is applied to the projection itself, and q, k are not
      normalized before projection (the paper's unit-norm assumption is only
      used by the theory, the code never normalizes). Both conventions are
      followed here.
    - The same W is used for queries and keys.

The corner probabilities are computed by the exact Bernoulli product form

    phi_l(x)[r] = prod_t ( p_t if c_{r,t} = +1 else 1 - p_t ),  p_t = sigmoid(2 beta u_t),

which holds because the softmax partition function factorizes into
prod_t 2 cosh(beta u_t). 1 - p_t is evaluated as sigmoid(-2 beta u_t) to avoid
cancellation when p_t is close to 1.

Corner ordering: corner r has c_{r,t} = +1 iff bit (P-1-t) of r is set, i.e.
plane 0 is the most significant bit. This is the ordering of
itertools.product([-1, +1], repeat=P) used by the authors, and the ordering
the CUDA kernels use for the debug A/B outputs.

Den is a positive sum in exact arithmetic (every phi entry is > 0). It can
only be zero after floating-point underflow, which needs beta large enough
that sigmoid(-2 beta) underflows and the query's buckets hold no key mass.
Such a query has no information to mix, so its output is defined as 0.
NaN in Den is propagated rather than masked.
"""
from __future__ import annotations

import itertools

import torch

MAX_PLANES = 5


def corner_signs(num_planes: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Hypercube corners in the authors' ordering.

    Returns:
        [R, P] tensor of +-1, R = 2^P, row r is corner r (plane 0 is the MSB).
    """
    corners = list(itertools.product([-1.0, 1.0], repeat=num_planes))
    return torch.tensor(corners, dtype=dtype, device=device)


def hash_projections(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Soft projections u = tanh(W_l x) for every table.

    Args:
        x: [..., d] query or key rows.
        W: [L, P, d] random planes.

    Returns:
        [..., L, P] values in (-1, 1).
    """
    return torch.tanh(torch.einsum("...d,lpd->...lp", x, W))


def corner_probs_softmax(x: torch.Tensor, W: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Corner probabilities by the direct length-R softmax (the paper's definition).

    Args:
        x: [..., d] rows. W: [L, P, d]. beta: 0-dim tensor.

    Returns:
        [..., L, R] probabilities, each length-R row sums to 1.
    """
    u = hash_projections(x, W)
    signs = corner_signs(W.shape[1], u.dtype, u.device)
    return torch.softmax(beta * (u @ signs.T), dim=-1)


def corner_probs_bernoulli(x: torch.Tensor, W: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Corner probabilities by the Bernoulli product form (what the kernels compute).

    The product is built plane by plane, doubling the corner axis each time:
    after plane t the entry index holds the signs of planes 0..t with plane 0
    as the most significant bit, matching corner_signs.

    Args:
        x: [..., d] rows. W: [L, P, d]. beta: 0-dim tensor.

    Returns:
        [..., L, R] probabilities, each length-R row sums to 1.
    """
    u = hash_projections(x, W)
    prob_plus = torch.sigmoid(2.0 * beta * u)
    prob_minus = torch.sigmoid(-2.0 * beta * u)
    phi = torch.ones_like(u[..., :1])
    for t in range(u.shape[-1]):
        pair = torch.stack(
            [phi * prob_minus[..., t : t + 1], phi * prob_plus[..., t : t + 1]], dim=-1
        )
        phi = pair.flatten(-2)
    return phi


def bucket_sums(
    k: torch.Tensor, v: torch.Tensor, W: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Key-side bucket statistics A and B.

    Args:
        k, v: [B, H, N, d]. W: [L, P, d]. beta: 0-dim tensor.

    Returns:
        A: [B, H, L, R], sum over keys of phi.
        B: [B, H, L, R, d], sum over keys of phi times v.
    """
    phi_k = corner_probs_bernoulli(k, W, beta)
    mass = phi_k.sum(dim=2)
    weighted_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_k, v)
    return mass, weighted_values


def race_forward_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Non-causal RACE forward, Algorithm 1 with global normalization.

    Args:
        q, k, v: [B, H, N, d] (the key and query lengths may differ).
        W: [L, P, d] fixed random planes, same dtype as q.
        beta: 0-dim tensor, the corner-logit temperature.

    Returns:
        o: [B, H, N_q, d].
    """
    num_tables = W.shape[0]
    mass, weighted_values = bucket_sums(k, v, W, beta)
    phi_q = corner_probs_bernoulli(q, W, beta)
    numerator = torch.einsum("bhnlr,bhlrd->bhnd", phi_q, weighted_values) / num_tables
    denominator = torch.einsum("bhnlr,bhlr->bhn", phi_q, mass).unsqueeze(-1) / num_tables
    empty = denominator == 0
    safe_denominator = torch.where(empty, torch.ones_like(denominator), denominator)
    return torch.where(empty, torch.zeros_like(numerator), numerator / safe_denominator)


def soft_hash(
    x: torch.Tensor, W: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Everything the backward needs about one side's hashing.

    Args:
        x: [..., d] rows. W: [L, P, d]. beta: 0-dim tensor.

    Returns:
        phi: [..., L, R] corner probabilities (same values as corner_probs_bernoulli).
        u: [..., L, P] projections tanh(W x).
        prob_plus, prob_minus: [..., L, P] sigmoid(2 beta u) and sigmoid(-2 beta u).
    """
    u = hash_projections(x, W)
    prob_plus = torch.sigmoid(2.0 * beta * u)
    prob_minus = torch.sigmoid(-2.0 * beta * u)
    plus = corner_signs(W.shape[1], u.dtype, u.device) > 0  # [R, P]
    factors = torch.where(plus, prob_plus[..., None, :], prob_minus[..., None, :])
    return factors.prod(dim=-1), u, prob_plus, prob_minus


def hash_backward(
    grad_phi: torch.Tensor, phi: torch.Tensor, u: torch.Tensor,
    prob_plus: torch.Tensor, prob_minus: torch.Tensor, W: torch.Tensor, beta: torch.Tensor,
    grad_phi_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Pulls a gradient with respect to phi back to the rows x and to beta.

    With s_{r,t} = +-1 the sign of corner r on plane t,

        d phi[r] / d u_t  = beta * phi[r] * c_{r,t}
        d phi[r] / d beta = phi[r] * sum_t u_t * c_{r,t}

    where c_{r,t} = s_{r,t} - (2 p_t - 1), evaluated without cancellation as
    2 sigmoid(-2 beta u_t) for s = +1 and -2 sigmoid(2 beta u_t) for s = -1.
    See BACKWARD_DERIVATION.md for the derivation.

    Args:
        grad_phi: [..., L, R] gradient with respect to phi.
        phi, u, prob_plus, prob_minus: from soft_hash. W: [L, P, d].
        grad_phi_scale: optional [..., L, R] magnitude of the terms that
            grad_phi sums; if given, the matching scale of h is returned too.

    Returns:
        dx: [..., d] gradient with respect to the rows.
        h: [..., L, P] h_t = sum_r phi[r] grad_phi[r] c_{r,t}, so that the
            gradient with respect to u_t is beta h_t and this side's
            contribution to the beta gradient is sum u_t h_t.
        h_scale (only with grad_phi_scale): sum_r phi[r] grad_phi_scale[r] |c_{r,t}|.
    """
    plus = corner_signs(W.shape[1], u.dtype, u.device) > 0  # [R, P]
    corner_coef = torch.where(plus, 2.0 * prob_minus[..., None, :], -2.0 * prob_plus[..., None, :])
    weighted = phi * grad_phi  # [..., L, R]
    h = torch.einsum("...lr,...lrp->...lp", weighted, corner_coef)  # [..., L, P]
    dz = beta * h * (1.0 - u * u)
    dx = torch.einsum("...lp,lpd->...d", dz, W)
    if grad_phi_scale is None:
        return dx, h
    h_scale = torch.einsum("...lr,...lrp->...lp", phi * grad_phi_scale, corner_coef.abs())
    return dx, h, h_scale


def race_backward_reference(
    grad_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    W: torch.Tensor, beta: torch.Tensor, return_scales: bool = False,
    bucket_totals: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Vector-Jacobian product of race_forward_reference, decomposed like the kernels.

    Query side (per query i, with g_i = grad_o[i]):
        y_l[r]   = B_l[r, :] . g_i
        Den_i    = sum_l sum_r phi_q[r] A_l[r]
        gNum_i   = sum_l sum_r phi_q[r] y_l[r]          (= g_i . Num_i)
        inv_den  = 1 / Den_i,  dden = -gNum_i / Den_i^2 (= dL/dDen_i)
        dphi_q   = y * inv_den + A * dden
        dB_l     = sum_i phi_q[r] * g_i * inv_den,  dA_l = sum_i phi_q[r] * dden
    Key side (per key j):
        dv_j     = sum_l sum_r phi_k[r] dB_l[r, :]
        dphi_k   = dB_l[r, :] . v_j + dA_l[r]
    Both sides then go through hash_backward. The 1/L of the forward cancels in
    O and is omitted. Queries whose Den underflowed to 0 have output 0 and no
    gradient (inv_den = dden = 0), matching the forward's torch.where.

    Args:
        grad_o: [B, H, N_q, d]. q: [B, H, N_q, d]. k, v: [B, H, N, d].
        W: [L, P, d]. beta: 0-dim tensor.
        bucket_totals: optional (A [B, H, L, R], B [B, H, L, R, d]) used in
            place of the bucket sums of k and v, the way the backward kernels
            use whatever totals the forward saved. Without it the result is
            the exact VJP.

    Returns:
        dq, dk, dv with the shapes of q, k, v, and dbeta as a 0-dim tensor.
        With return_scales, also h and h_scale (see hash_backward) for the
        query side [B, H, N_q, L, P] and the key side [B, H, N, L, P], where
        h_scale takes the two terms of each dphi (y / Den and A dDen on the
        query side, dB . v and dA on the key side) in absolute value. The
        tests derive their tolerances from them (tests/numerics.py).
    """
    phi_q, u_q, plus_q, minus_q = soft_hash(q, W, beta)
    phi_k, u_k, plus_k, minus_k = soft_hash(k, W, beta)
    if bucket_totals is None:
        mass = phi_k.sum(dim=2)  # A: [B, H, L, R]
        weighted_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_k, v)  # B: [B, H, L, R, d]
    else:
        mass, weighted_values = bucket_totals

    y = torch.einsum("bhlrd,bhnd->bhnlr", weighted_values, grad_o)
    den = torch.einsum("bhnlr,bhlr->bhn", phi_q, mass)
    grad_num_dot = torch.einsum("bhnlr,bhnlr->bhn", phi_q, y)
    empty = den == 0
    inv_den = torch.where(empty, torch.zeros_like(den), 1.0 / torch.where(empty, 1.0, den))
    grad_den = -(grad_num_dot * inv_den) * inv_den
    grad_phi_q = y * inv_den[..., None, None] + mass[:, :, None] * grad_den[..., None, None]
    grad_weighted_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_q, grad_o * inv_den[..., None])
    grad_mass = torch.einsum("bhnlr,bhn->bhlr", phi_q, grad_den)

    dv = torch.einsum("bhnlr,bhlrd->bhnd", phi_k, grad_weighted_values)
    grad_phi_k = torch.einsum("bhlrd,bhnd->bhnlr", grad_weighted_values, v) + grad_mass[:, :, None]

    if not return_scales:
        dq, h_q = hash_backward(grad_phi_q, phi_q, u_q, plus_q, minus_q, W, beta)
        dk, h_k = hash_backward(grad_phi_k, phi_k, u_k, plus_k, minus_k, W, beta)
        return dq, dk, dv, (u_q * h_q).sum() + (u_k * h_k).sum()

    grad_phi_q_scale = (y * inv_den[..., None, None]).abs() + (
        mass[:, :, None] * grad_den[..., None, None]
    ).abs()
    grad_phi_k_scale = torch.einsum("bhlrd,bhnd->bhnlr", grad_weighted_values, v).abs() + (
        grad_mass[:, :, None].abs()
    )
    dq, h_q, h_q_scale = hash_backward(
        grad_phi_q, phi_q, u_q, plus_q, minus_q, W, beta, grad_phi_q_scale
    )
    dk, h_k, h_k_scale = hash_backward(
        grad_phi_k, phi_k, u_k, plus_k, minus_k, W, beta, grad_phi_k_scale
    )
    dbeta = (u_q * h_q).sum() + (u_k * h_k).sum()
    return dq, dk, dv, dbeta, h_q, h_k, h_q_scale, h_k_scale

