"""Shared input generation and tolerances for the kernel tests.

Tolerance for the bf16 output O (kernel) against the fp64 reference evaluated
on the same bf16-rounded inputs:

    |O_kernel - O_ref| <= 2^-8 * |O_ref| + atol,   atol = 1e-5 * max(1, beta) * max_j |v_j|

Why:
  - The kernel rounds an fp32 value y32 to bf16 with round-to-nearest. bf16 has
    8 significant bits, so the unit roundoff is 2^-8 and
    |bf16(y32) - y32| <= 2^-8 |y32|. That is the rtol term, and it is tight:
    about half of all elements sit within a factor 2 of it.
  - y32 differs from the exact value by fp32 error. O is a convex combination
    of the v_j, so the natural scale of that error is max_j |v_j|, not |O|
    (|O| can be near 0 through cancellation). The dominant sources are the
    length-d fp32 dot products inside tanh (absolute error ~sqrt(d) * 2^-24 *
    |q| |w|), amplified by 2 beta P through the sigmoids into relative errors
    of phi, and the fp32 sums over up to 10^4 keys (serial per-tile sums of
    2048 terms plus the pairwise tree). A priori bounds put this at 1e-7 to
    1e-6 of max|v|; the CPU emulation of the kernels' exact summation order
    (test_kernel_emulation.py) measures 1.5e-8 at beta = 1 and 1.4e-7 at
    beta = 4. atol = 1e-5 * max(1, beta) * max|v| leaves a margin of more than
    100x over the measurement and grows with beta because the phi error does.
  - atol is kept that tight on purpose: outputs are averages of many values,
    so |O| is often ~0.05, where atol rather than rtol decides the check.
  - The emulation test asserts the fp32 error stays under 10% of atol, so a
    failure on the GPU at this tolerance points at a kernel bug, not at a
    tolerance that was too tight.

Tolerance for the debug A and B (fp32, no output rounding): the scale-aware
bound is 3e-5 relative to the natural magnitude of each entry, A[r] for A and
A[r] * max|v| for B[r, :], plus 1e-6 of the total mass N for rarely hit buckets
whose tiny A[r] carries a larger relative error. The emulation measures at most
1.4% of this bound. It is tight enough to catch a single masked tail token that
leaked into the sums (a relative change of about 1/N in A).

Gradients. dq, dk, dv (bf16) and d beta (fp32) are compared with the fp64
reference on the same bf16-rounded q, k, v and dO (race_backward_reference,
which test_backward_reference.py checks against autograd):

    |G_kernel - G_ref| <= 2^-8 * |G_ref| + 5e-5 * max(1, beta) * S_G

  - 2^-8 is again the final bf16 rounding.
  - S_G is the scale of the fp32 error, per head. It cannot be max |G_ref|:
    dq and dk carry the tanh derivative 1 - u^2, which fp32 evaluates with an
    absolute error near 2^-23, and z = w . x has standard deviation ~sqrt(d),
    so whole heads can be saturated with |G_ref| ~ 1e-12 (at small N every
    row can be). Their error scales with the gradient with respect to u,
    beta h_t, not with the gradient itself. And h_t = sum_r phi dphi c_{r,t}
    inherits the cancellation inside dphi (y / Den against A dDen: when
    a query's buckets all have mean close to O, these nearly cancel). So
      S_dq = head max of sum_{l,t} beta h_scale_{l,t} |w_{l,t}|
    with h_scale = sum_r phi (|y / Den| + |A dDen|) |c_{r,t}|, and the same
    for dk with |dB . v| + |dA|; S_dv = head max of |dv_ref| (no tanh, no
    cancellation inside); S_dbeta = sum of h_scale over every term.
    reference_backward computes them.
  - The CPU emulation of the backward's fp32 data flow
    (test_backward_emulation.py, which follows the kernels' decomposition,
    workspace layout and summation orders for dA, dB and d beta) uses at most
    7% of the absolute term for every gradient, over P in 1..5, L in 1..4,
    beta in {1/sqrt(128), 1/8, 1, 4} and N from 2 to 5000 (2049 in the test file); the test asserts
    10%, so a GPU failure points at a kernel bug. S_dq and S_dk are 1.1x to
    ~100x max |G_ref| (largest at small beta, where dphi is dominated by a
    part that is the same for every corner and cancels in h), but a single
    query token dropped from dA/dB still fails the dk/dv check
    (test_tolerance_sees_one_dropped_token).

The d beta bound uses 1e-6 instead of 5e-5: it is one fp32 scalar with no
bf16 rounding, and the emulation uses at most 7% of it.
"""
from __future__ import annotations

import torch

BF16_UNIT_ROUNDOFF = 2.0**-8
OUTPUT_ATOL_FACTOR = 1e-5
BUCKET_RTOL = 3e-5
BUCKET_MASS_ATOL = 1e-6


def make_bf16_inputs(
    seed: int, batch: int, heads: int, seq_len: int, head_dim: int,
    num_tables: int, num_planes: int, device: torch.device | str = "cpu",
):
    """Random bf16 q, k, v [B, H, N, d] and fp32 planes W [L, P, d] ~ N(0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    q, k, v = (torch.randn(shape, generator=gen).to(torch.bfloat16) for _ in range(3))
    W = torch.randn(num_tables, num_planes, head_dim, generator=gen)
    return q.to(device), k.to(device), v.to(device), W.to(device)


def make_bf16_grad_output(seed: int, like: torch.Tensor) -> torch.Tensor:
    """Random bf16 dO with the shape and device of `like`."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(like.shape, generator=gen).to(torch.bfloat16).to(like.device)


def output_atol(v: torch.Tensor, beta: float) -> torch.Tensor:
    """Per-head absolute tolerance, shape [B, H, 1, 1]."""
    value_scale = v.double().abs().amax(dim=(2, 3), keepdim=True)
    return OUTPUT_ATOL_FACTOR * max(1.0, beta) * value_scale


def assert_output_close(out: torch.Tensor, ref: torch.Tensor, v: torch.Tensor, beta: float) -> None:
    """Checks a bf16 kernel output against the fp64 reference, see module docstring."""
    error = (out.double() - ref).abs()
    bound = BF16_UNIT_ROUNDOFF * ref.abs() + output_atol(v, beta)
    worst = (error / bound).max().item()
    assert worst <= 1.0, f"output error exceeds tolerance by {worst:.3f}x"


def assert_buckets_close(
    mass: torch.Tensor, weighted_values: torch.Tensor,
    ref_mass: torch.Tensor, ref_weighted_values: torch.Tensor, v: torch.Tensor,
) -> None:
    """Checks A [B, H, L, R] and B [B, H, L, R, d] against the fp64 reference."""
    seq_len = v.shape[2]
    value_scale = v.double().abs().amax(dim=(2, 3), keepdim=True).unsqueeze(-1)
    mass_bound = BUCKET_RTOL * ref_mass + BUCKET_MASS_ATOL * seq_len
    mass_worst = ((mass.double() - ref_mass).abs() / mass_bound).max().item()
    assert mass_worst <= 1.0, f"A error exceeds tolerance by {mass_worst:.3f}x"
    values_bound = mass_bound.unsqueeze(-1) * value_scale
    values_worst = ((weighted_values.double() - ref_weighted_values).abs() / values_bound).max().item()
    assert values_worst <= 1.0, f"B error exceeds tolerance by {values_worst:.3f}x"


# Backward tolerances, derived in the "Gradients" part of the docstring.
GRAD_ATOL_FACTOR = 5e-5
BETA_GRAD_TOL_FACTOR = 1e-6


def reference_backward(grad_o, q, k, v, W, beta: float):
    """fp64 reference gradients and the error scales of the tolerances.

    Returns:
        (dq, dk, dv, dbeta) in fp64, and the scales (dq, dk, dv as [B, H, 1, 1],
        dbeta as a float) described in the module docstring.
    """
    from reference import race_backward_reference

    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    W64 = W.double()
    *grads, _, _, h_q_scale, h_k_scale = race_backward_reference(
        grad_o.double(), q.double(), k.double(), v.double(), W64, beta64, return_scales=True
    )

    def pre_tanh_scale(h_scale):
        # Head maximum of sum_{l,t} beta h_scale_{l,t} |w_{l,t}|: the scale of
        # the gradient with respect to u, pushed through W without the tanh
        # derivative.
        magnitude = torch.einsum("bhnlp,lpd->bhnd", abs(beta) * h_scale, W64.abs())
        return magnitude.amax(dim=(2, 3), keepdim=True)

    scales = (
        pre_tanh_scale(h_q_scale),
        pre_tanh_scale(h_k_scale),
        grads[2].abs().amax(dim=(2, 3), keepdim=True),
        (h_q_scale.sum() + h_k_scale.sum()).item(),
    )
    return grads, scales


def assert_grad_close(out: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor, beta: float,
                      name: str) -> None:
    """Checks a bf16 kernel gradient against the fp64 reference, scale from reference_backward."""
    error = (out.double() - ref).abs()
    bound = BF16_UNIT_ROUNDOFF * ref.abs() + GRAD_ATOL_FACTOR * max(1.0, beta) * scale
    worst = (error / bound).max().item()
    assert worst <= 1.0, f"{name} error exceeds tolerance by {worst:.3f}x"


def assert_beta_grad_close(got: torch.Tensor, ref: torch.Tensor, scale: float, beta: float) -> None:
    """Checks the fp32 d beta against the fp64 reference, scale from reference_backward."""
    error = abs(got.item() - ref.item())
    bound = BETA_GRAD_TOL_FACTOR * max(1.0, beta) * scale
    assert error <= bound, f"d beta error {error:.3e} exceeds {bound:.3e}"
