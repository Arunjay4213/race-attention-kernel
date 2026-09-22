"""CPU tests of the reference implementation (reference.py).

All tensors are tiny so the whole file runs in a few seconds.
"""
from __future__ import annotations

import pytest
import torch

from reference import (
    bucket_sums,
    corner_probs_bernoulli,
    corner_probs_softmax,
    corner_signs,
    race_forward_reference,
)

PLANE_COUNTS = [1, 2, 3, 4, 5]
BETAS = [0.0, 1.0 / 128**0.5, 1.0, 4.0]


def make_inputs(
    seed: int, batch: int, heads: int, seq_len: int, head_dim: int,
    num_tables: int, num_planes: int, dtype: torch.dtype,
):
    """Random q, k, v [B, H, N, d] and planes W [L, P, d]."""
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, generator=gen, dtype=dtype)
    k = torch.randn(shape, generator=gen, dtype=dtype)
    v = torch.randn(shape, generator=gen, dtype=dtype)
    W = torch.randn(num_tables, num_planes, head_dim, generator=gen, dtype=dtype)
    return q, k, v, W


@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("beta_value", BETAS)
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_bernoulli_matches_softmax(num_planes, beta_value, dtype, tol):
    x = torch.randn(2, 3, 50, 16, dtype=dtype, generator=torch.Generator().manual_seed(1))
    W = torch.randn(3, num_planes, 16, dtype=dtype, generator=torch.Generator().manual_seed(2))
    beta = torch.tensor(beta_value, dtype=dtype)
    direct = corner_probs_softmax(x, W, beta)
    product = corner_probs_bernoulli(x, W, beta)
    assert direct.shape == (2, 3, 50, 3, 1 << num_planes)
    torch.testing.assert_close(product, direct, rtol=0.0, atol=tol)


@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_corner_probs_are_distributions(num_planes):
    x = torch.randn(4, 7, 32, dtype=torch.float64)
    W = torch.randn(2, num_planes, 32, dtype=torch.float64)
    phi = corner_probs_bernoulli(x, W, torch.tensor(2.0, dtype=torch.float64))
    assert (phi > 0).all()
    torch.testing.assert_close(phi.sum(-1), torch.ones(4, 7, 2, dtype=torch.float64))


def test_corner_order_puts_plane_zero_in_the_msb():
    signs = corner_signs(3, torch.float64, "cpu")
    for r in range(8):
        for t in range(3):
            expected = 1.0 if (r >> (2 - t)) & 1 else -1.0
            assert signs[r, t].item() == expected


def test_large_positive_projection_selects_the_all_plus_corner():
    # A key aligned with every plane lands in corner R-1 (all +1) when beta is large.
    W = torch.eye(4, 8, dtype=torch.float64).unsqueeze(0)
    x = torch.zeros(8, dtype=torch.float64)
    x[:4] = 10.0
    phi = corner_probs_bernoulli(x, W, torch.tensor(20.0, dtype=torch.float64))
    assert phi.argmax().item() == 15
    assert phi[0, 15].item() > 1 - 1e-12


@pytest.mark.parametrize("num_planes", [1, 3, 5])
@pytest.mark.parametrize("num_tables", [1, 2, 4])
def test_permutation_invariance_over_keys(num_planes, num_tables):
    q, k, v, W = make_inputs(3, 2, 2, 37, 16, num_tables, num_planes, torch.float64)
    beta = torch.tensor(1.5, dtype=torch.float64)
    perm = torch.randperm(37, generator=torch.Generator().manual_seed(4))
    out = race_forward_reference(q, k, v, W, beta)
    out_permuted = race_forward_reference(q, k[:, :, perm], v[:, :, perm], W, beta)
    torch.testing.assert_close(out_permuted, out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("num_tables", [1, 4])
def test_zero_beta_gives_mean_of_values(num_planes, num_tables):
    # beta = 0 makes every phi uniform (2^-P), so all keys get equal weight.
    q, k, v, W = make_inputs(5, 1, 3, 29, 8, num_tables, num_planes, torch.float64)
    out = race_forward_reference(q, k, v, W, torch.tensor(0.0, dtype=torch.float64))
    expected = v.mean(dim=2, keepdim=True).expand_as(out)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("num_planes", [1, 2, 4, 5])
@pytest.mark.parametrize("num_tables", [1, 3])
def test_matches_quadratic_kernel_form(num_planes, num_tables):
    # Algorithm 1 is attention with the kernel s(q, k) = (1/L) sum_l <phi_l(q), phi_l(k)>.
    # Checking against the O(N^2) form verifies the bucket factorization independently.
    q, k, v, W = make_inputs(6, 2, 1, 23, 16, num_tables, num_planes, torch.float64)
    beta = torch.tensor(2.0, dtype=torch.float64)
    phi_q = corner_probs_softmax(q, W, beta)
    phi_k = corner_probs_softmax(k, W, beta)
    scores = torch.einsum("bhilr,bhjlr->bhij", phi_q, phi_k) / num_tables
    expected = scores @ v / scores.sum(-1, keepdim=True)
    out = race_forward_reference(q, k, v, W, beta)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_output_is_a_convex_combination_of_values():
    q, k, v, W = make_inputs(7, 1, 2, 64, 16, 2, 3, torch.float64)
    out = race_forward_reference(q, k, v, W, torch.tensor(3.0, dtype=torch.float64))
    assert (out <= v.amax(dim=2, keepdim=True) + 1e-12).all()
    assert (out >= v.amin(dim=2, keepdim=True) - 1e-12).all()


def test_bucket_sums_shapes_and_total_mass():
    q, k, v, W = make_inputs(8, 2, 3, 41, 16, 3, 4, torch.float64)
    mass, weighted_values = bucket_sums(k, v, W, torch.tensor(1.0, dtype=torch.float64))
    assert mass.shape == (2, 3, 3, 16)
    assert weighted_values.shape == (2, 3, 3, 16, 16)
    torch.testing.assert_close(mass.sum(-1), torch.full((2, 3, 3), 41.0, dtype=torch.float64))
    # Summing B over buckets undoes the soft assignment: sum_r B[r] = sum_j v_j.
    torch.testing.assert_close(
        weighted_values.sum(-2), v.sum(2, keepdim=True).expand(2, 3, 3, 16)
    )


def test_underflowed_denominator_gives_zero_not_nan():
    # With huge beta in fp32 every key sits in the all-plus corner and a query in
    # the all-minus corner has exactly zero overlap after underflow.
    W = torch.eye(2, 4).unsqueeze(0)
    k = torch.zeros(1, 1, 5, 4)
    k[..., :2] = 5.0
    q = -k[:, :, :1]
    v = torch.randn(1, 1, 5, 4)
    out = race_forward_reference(q, k, v, W, torch.tensor(200.0))
    assert torch.equal(out, torch.zeros_like(out))


def test_gradients_flow_to_inputs_and_beta():
    q, k, v, W = make_inputs(9, 1, 1, 6, 4, 2, 2, torch.float64)
    beta = torch.tensor(0.7, dtype=torch.float64)
    inputs = [t.clone().requires_grad_() for t in (q, k, v, beta)]
    assert torch.autograd.gradcheck(
        lambda q_, k_, v_, b_: race_forward_reference(q_, k_, v_, W, b_), inputs
    )
