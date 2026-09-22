"""CPU checks of the backward derivation (BACKWARD_DERIVATION.md).

race_backward_reference evaluates the vector-Jacobian product in the same
decomposition the kernels use (y, Den, gNum, dB, dA, the corner coefficients
c_{r,t}); here it is compared, in fp64, with torch.autograd applied to
race_forward_reference, and checked against finite differences with
gradcheck. All tensors are tiny: the file runs in a few seconds.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from reference import (
    corner_probs_softmax,
    corner_signs,
    race_backward_reference,
    race_forward_reference,
)

PLANE_COUNTS = [1, 2, 4, 5]
TABLE_COUNTS = [1, 2, 4]
HEAD_DIMS = [64, 128]
BETAS = [1.0 / 128**0.5, 1.0, 4.0]
# fp64 evaluation of two algebraically equal expressions: the measured
# worst case is ~1e-14 of the tensor's largest entry (3e-13 for d beta against
# its sum of magnitudes), so 1e-11 only fails on a real mismatch.
FP64_TOL = 1e-11


def make_inputs(seed, batch, heads, seq_len, head_dim, num_tables, num_planes):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    q, k, v, grad_o = (torch.randn(shape, generator=gen, dtype=torch.float64) for _ in range(4))
    W = torch.randn(num_tables, num_planes, head_dim, generator=gen, dtype=torch.float64)
    return q, k, v, grad_o, W


def autograd_grads(grad_o, q, k, v, W, beta):
    inputs = [t.clone().requires_grad_() for t in (q, k, v, beta)]
    out = race_forward_reference(inputs[0], inputs[1], inputs[2], W, inputs[3])
    return torch.autograd.grad(out, inputs, grad_o)


def relative_errors(grad_o, q, k, v, W, beta) -> dict[str, float]:
    """max |decomposed - autograd| / scale per gradient; the scale is max |autograd|
    for dq, dk, dv and sum |u_t h_t| over all terms for d beta."""
    expected = autograd_grads(grad_o, q, k, v, W, beta)
    *got, h_q, h_k, _, _ = race_backward_reference(grad_o, q, k, v, W, beta, return_scales=True)
    errors = {}
    for name, a, b in zip(("dq", "dk", "dv"), got[:3], expected[:3], strict=True):
        errors[name] = ((a - b).abs().max() / b.abs().max()).item()
    u_q, u_k = (torch.tanh(torch.einsum("bhnd,lpd->bhnlp", x, W)) for x in (q, k))
    beta_scale = (u_q * h_q).abs().sum() + (u_k * h_k).abs().sum()
    errors["dbeta"] = ((got[3] - expected[3]).abs() / beta_scale).item()
    return errors


@pytest.mark.parametrize("beta_value", BETAS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_decomposition_matches_autograd(num_planes, num_tables, head_dim, beta_value):
    q, k, v, grad_o, W = make_inputs(
        hash((num_planes, num_tables, head_dim)) % 2**31, 2, 2, 23, head_dim, num_tables, num_planes
    )
    errors = relative_errors(grad_o, q, k, v, W, torch.tensor(beta_value, dtype=torch.float64))
    assert max(errors.values()) <= FP64_TOL, errors


def test_zero_beta_and_beta_symmetry():
    # beta = 0 makes phi uniform and O = mean(v), independent of q and k, so
    # dq and dk vanish exactly (every chain carries a factor beta). O is also
    # even in beta: phi(-beta)[r] = phi(beta)[complement of r], and the sum
    # over r of phi_q[r] phi_k[r] does not care which corner is which. So
    # d beta is odd in beta and 0 at beta = 0.
    q, k, v, grad_o, W = make_inputs(1, 1, 2, 19, 16, 2, 3)
    zero = torch.tensor(0.0, dtype=torch.float64)
    dq, dk, dv, dbeta = race_backward_reference(grad_o, q, k, v, W, zero)
    expected = autograd_grads(grad_o, q, k, v, W, zero)
    assert not dq.any() and not dk.any()
    assert expected[0].abs().max() <= 1e-15 and expected[1].abs().max() <= 1e-15
    torch.testing.assert_close(dv, expected[2], rtol=1e-12, atol=1e-14)
    assert dbeta.abs() <= 1e-14 and expected[3].abs() <= 1e-14

    plus = race_backward_reference(grad_o, q, k, v, W, torch.tensor(0.7, dtype=torch.float64))
    minus = race_backward_reference(grad_o, q, k, v, W, torch.tensor(-0.7, dtype=torch.float64))
    torch.testing.assert_close(minus[3], -plus[3], rtol=1e-12, atol=1e-15)
    for a, b in zip(plus[:3], minus[:3], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)
    assert plus[3].abs() > 1e-6


@pytest.mark.parametrize("num_planes", [1, 3, 5])
def test_phi_jacobian_is_beta_not_two_beta(num_planes):
    # d phi[r] / d u_t = beta phi[r] c_{r,t} with c = s - (2p - 1) = s - tanh(beta u),
    # checked against autograd of the softmax definition phi = softmax(beta u . s).
    gen = torch.Generator().manual_seed(num_planes)
    u = torch.tanh(torch.randn(6, num_planes, generator=gen, dtype=torch.float64))
    beta = 1.7
    signs = corner_signs(num_planes, torch.float64, "cpu")  # [R, P]
    jacobian = torch.autograd.functional.jacobian(
        lambda x: torch.softmax(beta * (x @ signs.T), dim=-1), u
    )  # [6, R, 6, P]
    jacobian = torch.einsum("irip->irp", jacobian)
    phi = torch.softmax(beta * (u @ signs.T), dim=-1)
    prob_plus = torch.sigmoid(2 * beta * u)
    coef = torch.where(signs > 0, 2 * torch.sigmoid(-2 * beta * u)[:, None], -2 * prob_plus[:, None])
    torch.testing.assert_close(coef, signs - (2 * prob_plus[:, None] - 1), rtol=0, atol=1e-15)
    torch.testing.assert_close(coef, signs - torch.tanh(beta * u)[:, None], rtol=0, atol=1e-15)
    torch.testing.assert_close(jacobian, beta * phi[..., None] * coef, rtol=1e-12, atol=1e-15)
    assert (jacobian - 2 * beta * phi[..., None] * coef).abs().max() > 1e-3


def test_phi_beta_derivative():
    # d phi[r] / d beta = phi[r] sum_t u_t c_{r,t}.
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(5, 8, generator=gen, dtype=torch.float64)
    W = torch.randn(2, 3, 8, generator=gen, dtype=torch.float64)
    beta = torch.tensor(0.9, dtype=torch.float64)
    derivative = torch.autograd.functional.jacobian(lambda b: corner_probs_softmax(x, W, b), beta)
    u = torch.tanh(torch.einsum("nd,lpd->nlp", x, W))
    signs = corner_signs(3, torch.float64, "cpu")
    coef = signs - torch.tanh(beta * u)[..., None, :]  # [N, L, R, P]
    phi = corner_probs_softmax(x, W, beta)
    torch.testing.assert_close(derivative, phi * (coef * u[..., None, :]).sum(-1), rtol=1e-12, atol=1e-15)


class ReferenceAttention(torch.autograd.Function):
    """race_forward_reference with race_backward_reference as its backward, so
    gradcheck compares the decomposition with finite differences directly."""

    @staticmethod
    def forward(ctx, q, k, v, W, beta):
        ctx.save_for_backward(q, k, v, W, beta)
        return race_forward_reference(q, k, v, W, beta)

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, W, beta = ctx.saved_tensors
        dq, dk, dv, dbeta = race_backward_reference(grad_o, q, k, v, W, beta)
        return dq, dk, dv, None, dbeta


@pytest.mark.parametrize("num_planes, num_tables", [(1, 1), (3, 2), (5, 2)])
def test_gradcheck_against_finite_differences(num_planes, num_tables):
    # d = 4 keeps |w . x| small enough that tanh is not saturated, so every
    # chain carries a finite-difference-visible gradient.
    q, k, v, _, W = make_inputs(num_planes, 1, 2, 5, 4, num_tables, num_planes)
    beta = torch.tensor(0.8, dtype=torch.float64)
    inputs = [t.clone().requires_grad_() for t in (q, k, v)] + [beta.clone().requires_grad_()]
    assert torch.autograd.gradcheck(
        lambda q_, k_, v_, b_: ReferenceAttention.apply(q_, k_, v_, W, b_), inputs
    )


def test_underflowed_denominator_row_has_zero_gradient():
    # Same construction as the forward test: in fp32 with a huge beta the
    # query's Den underflows to 0 and its output is defined as 0, so neither
    # it nor the keys receive gradient through it, and nothing is NaN.
    W = torch.eye(2, 4).unsqueeze(0)
    k = torch.zeros(1, 1, 5, 4)
    k[..., :2] = 5.0
    q = torch.cat([-k[:, :, :1], k[:, :, :1]], dim=2)  # row 0 underflows, row 1 does not
    v = torch.randn(1, 1, 5, 4, generator=torch.Generator().manual_seed(3))
    grad_o = torch.randn(1, 1, 2, 4, generator=torch.Generator().manual_seed(4))
    beta = torch.tensor(200.0)
    dq, dk, dv, dbeta = race_backward_reference(grad_o, q, k, v, W, beta)
    for grad in (dq, dk, dv, dbeta):
        assert torch.isfinite(grad).all()
    assert not dq[:, :, 0].any()
    # Only row 1 contributes, and it matches the reference restricted to it.
    dq1, dk1, dv1, _ = race_backward_reference(grad_o[:, :, 1:], q[:, :, 1:], k, v, W, beta)
    torch.testing.assert_close(dv, dv1)
    torch.testing.assert_close(dk, dk1)
    torch.testing.assert_close(dq[:, :, 1:], dq1)


def test_largest_relative_errors_over_the_grid():
    # One pass over the full grid of the derivation check, reporting the worst
    # error per gradient (visible with pytest -s); the per-case test above
    # already asserts each case.
    worst = dict.fromkeys(("dq", "dk", "dv", "dbeta"), 0.0)
    for num_planes, num_tables, head_dim, beta_value in itertools.product(
        PLANE_COUNTS, TABLE_COUNTS, HEAD_DIMS, BETAS
    ):
        q, k, v, grad_o, W = make_inputs(7, 1, 1, 9, head_dim, num_tables, num_planes)
        errors = relative_errors(grad_o, q, k, v, W, torch.tensor(beta_value, dtype=torch.float64))
        for name, error in errors.items():
            worst[name] = max(worst[name], error)
    print("worst relative errors:", {name: f"{error:.1e}" for name, error in worst.items()})
    assert max(worst.values()) <= FP64_TOL
