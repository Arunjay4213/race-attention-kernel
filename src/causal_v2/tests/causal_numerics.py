"""Input generation and tolerances for the causal v2a GPU tests.

Output tolerance (docs/causal_v2_design.md section 2.4). The kernel output O (bf16) is
compared with the fp64 reference O_ref evaluated on the same bf16-rounded
inputs. Let e_ref = |bf16(O_ref) - O_ref| per element, the rounding error that
even a perfect kernel cannot avoid, F = max(e_ref) and F_rms = rms(e_ref).
v2a must satisfy

    max|O - O_ref| <= 1.1 F + a,      rms|O - O_ref| <= 1.05 F_rms + a_rms.

Why this holds for a correct v2a and why it is tight:
  - v2a computes an fp32 value y whose error delta = |y - O_ref| is ~1e-7 of
    max|v| (fp32 dot products, sums of <= T_blk terms, one reciprocal; the
    plan's CPU simulation measures 1e-7 to 3.5e-7 relative before rounding).
    bf16(y) equals bf16(O_ref) unless O_ref lies within delta of a rounding
    midpoint; there both candidates are about half an ulp away, so the error
    of that element exceeds its own e_ref by at most 2 delta. Hence
    max|O - O_ref| <= F + 2 max(delta), and the RMS changes by a negligible
    amount because only a fraction ~delta/ulp of elements flip.
    The plan's simulation gives exactly 1.00 F and 1.00 F_rms.
  - The factors 1.1 and 1.05 therefore flag any systematic error larger than
    10% (max) or 5% (RMS) of bf16 output rounding, far below what a wrong
    mask, carry, scan offset or tail would cause. An off-by-one mask moves
    O_i by about |v_i - O_i| / (i + 1), which exceeds F (about 2^-7 at
    |O| in [2, 4)) for roughly the first hundred tokens of every sequence.
  - a = 1e-5 max(1, beta) max|v| is the non-causal suite's fp32 allowance
    (100x its measured fp32 error). It is about 1% of F or less here and
    only matters when F is 0: outputs that are exactly representable, such as
    T = 1 (O_0 = v_0) or a constant V, where a correct kernel also returns
    the exact value.
  - a_rms = 1e-7 max|v| plays the same role for the RMS bound; an allowance of
    a's size would be 10-70% of F_rms at long T, where |O| is small.

Bucket-state tolerance (debug prefix states, fp32 sums without output
rounding), as in the non-causal suite: |X - X_ref| <= 3e-5 * scale + 1e-6 * T
(times max|v| for B), where scale is the fp64 A entry. Sums of up to T_blk
tokens in fp32 plus a sequential scan over tiles stay far inside it.
"""
from __future__ import annotations

import torch

MAX_FACTOR = 1.1
RMS_FACTOR = 1.05
ATOL_FACTOR = 1e-5
RMS_ATOL_FACTOR = 1e-7
STATE_RTOL = 3e-5
STATE_MASS_ATOL = 1e-6


def make_bf16_inputs(
    seed: int, batch: int, heads: int, seq_len: int, head_dim: int,
    num_tables: int, num_planes: int, device: str = "cuda",
):
    """Random bf16 q, k, v [B, H, T, d] and fp32 planes W [L, P, d] ~ N(0, 1)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    q, k, v = (torch.randn(shape, generator=gen, device=device).to(torch.bfloat16) for _ in range(3))
    W = torch.randn(num_tables, num_planes, head_dim, generator=gen, device=device)
    return q, k, v, W


def assert_within_rounding_floor(out: torch.Tensor, ref: torch.Tensor, v: torch.Tensor, beta: float) -> None:
    """The v2a output tolerance of the module docstring."""
    rounding = (ref.to(torch.bfloat16).double() - ref).abs()
    floor_max = rounding.max().item()
    floor_rms = rounding.pow(2).mean().sqrt().item()
    value_scale = v.double().abs().max().item()
    error = (out.double() - ref).abs()
    error_max = error.max().item()
    error_rms = error.pow(2).mean().sqrt().item()
    max_bound = MAX_FACTOR * floor_max + ATOL_FACTOR * max(1.0, beta) * value_scale
    rms_bound = RMS_FACTOR * floor_rms + RMS_ATOL_FACTOR * value_scale
    assert error_max <= max_bound, (
        f"max error {error_max:.3e} > {max_bound:.3e} (F = {floor_max:.3e}, {error_max / max(floor_max, 1e-30):.3f} F)"
    )
    assert error_rms <= rms_bound, (
        f"rms error {error_rms:.3e} > {rms_bound:.3e} (F_rms = {floor_rms:.3e}, "
        f"{error_rms / max(floor_rms, 1e-30):.3f} F_rms)"
    )


def assert_states_close(mass, values, ref_mass, ref_values, v: torch.Tensor) -> None:
    """Checks A [..., L, R] and B [..., L, R, d] (fp32) against fp64 references.

    The leading dimensions of mass and values must be [B, H, ...]; v is
    [B, H, T, d] and sets the value scale per head.
    """
    seq_len = v.shape[2]
    extra = mass.dim() - 2
    value_scale = v.double().abs().amax(dim=(2, 3)).reshape(*v.shape[:2], *([1] * (extra + 1)))
    mass_bound = STATE_RTOL * ref_mass.abs() + STATE_MASS_ATOL * seq_len
    mass_worst = ((mass.double() - ref_mass).abs() / mass_bound).max().item()
    assert mass_worst <= 1.0, f"A error exceeds tolerance by {mass_worst:.3f}x"
    values_bound = mass_bound.unsqueeze(-1) * value_scale
    values_worst = ((values.double() - ref_values).abs() / values_bound).max().item()
    assert values_worst <= 1.0, f"B error exceeds tolerance by {values_worst:.3f}x"
