"""Input generation and tolerances for the causal v2 GPU tests (v2a and v2b).

Output tolerance (docs/causal_v2_design.md section 2.4). The kernel output O (bf16) is
compared with the fp64 reference O_ref evaluated on the same bf16-rounded
inputs. Let e_ref = |bf16(O_ref) - O_ref| per element, the rounding error that
even a perfect kernel cannot avoid, F = max(e_ref) and F_rms = rms(e_ref).
A variant must satisfy

    max|O - O_ref| <= k_max F + a,      rms|O - O_ref| <= k_rms F_rms + a_rms,

with (k_max, k_rms) = (1.1, 1.05) for v2a and (2.5, 2.0) for v2b.

v2a (fp32 CUDA cores).

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

v2b (bf16 tensor cores, fp32 accumulation).
  - v2b rounds Phi_Q and Phi_K to bf16 (unit roundoff u = 2^-8) before the
    products. W enters as bf16 hi + lo (W to 2^-16, which the design
    measures at 1.7e-5 relative in O), the carry as B_hi + B_lo, and the
    masked Gram matrix as G_hi + G_lo (G to 2^-16). Den sums exactly the
    weights that multiply V in Num, so O stays a convex combination of the
    values with slightly perturbed weights w_ij = k_ij / sum_j k_ij: the
    pre-rounding error is delta_i = sum_j (w'_ij - w_ij)(v_j - O_i).
  - There is no useful worst-case bound: each k_ij = <Phi_Q,i, Phi_K,j>
    carries a relative error of up to about 2u from the two operand
    roundings, so |delta_i| can reach ~4u max|v - O|, several F. The
    errors of different j have random signs and average out once a query
    sees many keys; they are largest in the first rows of a sequence.
  - The bounds 2.5 F and 2 F_rms are the design's, from its CPU simulation
    of the v2b arithmetic (1.00-1.82 F max and 1.27-1.53 F_rms for beta from
    1/sqrt(d) to 2, with G rounded to bf16 alone). They are regression
    bounds at typical (Gaussian) data that stay tight enough to catch
    structural errors: an off-by-one mask, a dropped carry term or a wrong
    scan offset gives errors of order |v_i - O_i| / (i + 1) or larger, far
    above 2.5 F in the first rows.
  - Rounding G to bf16 alone, as the design planned, does not meet them on
    this grid: on an A10G, d = 64, P = 4, L = 4, T = 2047, B * H = 1,
    beta = 1/8 gave 2.74 F (row 2, a query with 3 keys), and d = 128, P = 1,
    L = 1, T = 64, beta = 1/sqrt(128) gave 2.03 F_rms. An fp64 emulation of
    the kernel's rounding points reproduces both numbers; with G exact it
    gives 1.00 F and 1.34 F_rms for the same cases, hence the G split. With
    it, the worst cases over test_forward_matches_reference's grid on an
    A10G were 1.69 F and 1.42 F_rms.
  - The allowances a and a_rms are the v2a ones; with PRECISE_CARRY a
    constant V still comes back exactly (Num and Den carry the same weights
    to fp32 accuracy), so F = 0 cases need no more.

v2b against v2a (test_v2b_agrees_with_v2a). Both outputs are compared with
the same reference, so by the triangle inequality (and Minkowski's for the
RMS) |O_b - O_a| <= |O_b - O_ref| + |O_a - O_ref|, and the agreement bound
is the sum of the two variants' bounds: (2.5 + 1.1) F + 2 a in the max and
(2.0 + 1.05) F_rms + 2 a_rms in the RMS (worst measured over the reference
test's grid on an A10G: 2.33 F and 1.53 F_rms). The v2b bound alone is not enough:
on an A10G, d = 64, P = 4, L = 4, T = 5000, beta = 1 gave
max|O_b - O_a| = 3.01 F while v2b was within 2.01 F and v2a within 1.00 F
of the reference; one bf16 ulp in the top binade is already 2 F, so two
correct outputs that round a nearby value in opposite directions differ by
up to 3 F.

Agreement between tile lengths (test_forced_tile_lengths_agree). Runs with
T_blk = C, 2C and 2048 sum the same terms in different orders, so their fp32
values differ by fp32 noise and their bf16 outputs by at most one ulp, except
where O nearly cancels to 0: there the noise is absolute (a few 2^-24 of
max|v|) while the ulp scales with |O|. Measured on an A10G, over the
test's grid, 511 of 1.9e7 compared elements more than one ulp apart, all at
|O| between 1e-11 and 1e-6, with |O_1 - O_2| at most 1.15e-8 max|v|; so a
pure one-ulp bound cannot hold. The check is one ulp plus the same a as
above, which is still 870x above the worst measured difference and far below
the 2^-7 |O| an actual scan or carry error produces at typical |O|.

Bucket-state tolerance (debug prefix states, fp32 sums without output
rounding), as in the non-causal suite: |X - X_ref| <= 3e-5 * scale + 1e-6 * T
(times max|v| for B), where scale is the fp64 A entry. Sums of up to T_blk
tokens in fp32 plus a sequential scan over tiles stay far inside it.
v2b sums bf16-rounded Phi_K: |bf16(phi) - phi| <= 2^-8 phi for every term
(round to nearest, 8 significant bits), so |A' - A| <= 2^-8 A and
|B' - B| <= 2^-8 A max|v| in the worst case, and the relative part of the
v2b bound is 2^-8 + 3e-5. The v2b states are compared with an emulation
that uses the kernel's effective planes W_hi + W_lo (split_planes), so the
projection's hi/lo error, which depends on the data (a worst case of
2^-16 sum_i |x_i w_i| per plane before tanh), is not part of the
comparison; the output tests use the true W.
Agreement of the tile lengths holds for v2b by the same argument as for
v2a: the bf16 Phi values do not depend on the tile length (a token's row
inside its 16-row fragment is always its index mod 16), so only fp32 sums
change order, and the hi/lo carry keeps them to ~2^-16. It does not hold
for a v2b built with the single bf16 carry: that rounds the differently
ordered state sums to 2^-8, and on an A10G two cases of the test's grid
(d = 128, P = 4, L = 4, T = 129 and d = 128, P = 3, L = 2, T = 10000)
differed by more than one ulp, so the check is skipped for that build.
"""
from __future__ import annotations

import torch

# (max factor, RMS factor) per variant, see the module docstring.
OUTPUT_FACTORS = {"v2a": (1.1, 1.05), "v2b": (2.5, 2.0)}
ATOL_FACTOR = 1e-5
RMS_ATOL_FACTOR = 1e-7
STATE_RTOL = {"v2a": 3e-5, "v2b": 2.0**-8 + 3e-5}
STATE_MASS_ATOL = 1e-6


def split_planes(W: torch.Tensor) -> torch.Tensor:
    """W_hi + W_lo in fp64: the planes v2b effectively projects with.

    W_hi = bf16(W) and W_lo = bf16(W - W_hi), both round to nearest even, as
    in the kernels' prologue; W - W_hi is exact in fp32.
    """
    high = W.to(torch.bfloat16)
    low = (W - high.float()).to(torch.bfloat16)
    return high.double() + low.double()


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


def _rounding_floor(ref: torch.Tensor) -> tuple[float, float]:
    """(F, F_rms): the max and RMS of |bf16(O_ref) - O_ref|."""
    rounding = (ref.to(torch.bfloat16).double() - ref).abs()
    return rounding.max().item(), rounding.pow(2).mean().sqrt().item()


def _check_error(error: torch.Tensor, floor_max: float, floor_rms: float, max_factor: float,
                 rms_factor: float, atol: float, rms_atol: float, label: str) -> None:
    error_max = error.max().item()
    error_rms = error.pow(2).mean().sqrt().item()
    max_bound = max_factor * floor_max + atol
    rms_bound = rms_factor * floor_rms + rms_atol
    assert error_max <= max_bound, (
        f"{label}: max error {error_max:.3e} > {max_bound:.3e} "
        f"(F = {floor_max:.3e}, {error_max / max(floor_max, 1e-30):.3f} F)"
    )
    assert error_rms <= rms_bound, (
        f"{label}: rms error {error_rms:.3e} > {rms_bound:.3e} (F_rms = {floor_rms:.3e}, "
        f"{error_rms / max(floor_rms, 1e-30):.3f} F_rms)"
    )


def assert_within_rounding_floor(out: torch.Tensor, ref: torch.Tensor, v: torch.Tensor, beta: float,
                                 variant: str = "v2a") -> None:
    """The output tolerance of the module docstring for `variant` ("v2a" or "v2b")."""
    max_factor, rms_factor = OUTPUT_FACTORS[variant]
    floor_max, floor_rms = _rounding_floor(ref)
    value_scale = v.double().abs().max().item()
    _check_error((out.double() - ref).abs(), floor_max, floor_rms, max_factor, rms_factor,
                 ATOL_FACTOR * max(1.0, beta) * value_scale, RMS_ATOL_FACTOR * value_scale, variant)


def assert_variants_agree(out_v2b: torch.Tensor, out_v2a: torch.Tensor, ref: torch.Tensor,
                          v: torch.Tensor, beta: float) -> None:
    """v2b against v2a: the sum of the two variants' bounds (module docstring)."""
    (max_b, rms_b), (max_a, rms_a) = OUTPUT_FACTORS["v2b"], OUTPUT_FACTORS["v2a"]
    floor_max, floor_rms = _rounding_floor(ref)
    value_scale = v.double().abs().max().item()
    _check_error((out_v2b.double() - out_v2a.double()).abs(), floor_max, floor_rms, max_b + max_a,
                 rms_b + rms_a, 2 * ATOL_FACTOR * max(1.0, beta) * value_scale,
                 2 * RMS_ATOL_FACTOR * value_scale, "v2b vs v2a")


def assert_states_close(mass, values, ref_mass, ref_values, v: torch.Tensor, variant: str = "v2a") -> None:
    """Checks A [..., L, R] and B [..., L, R, d] (fp32) against fp64 references.

    The leading dimensions of mass and values must be [B, H, ...]; v is
    [B, H, T, d] and sets the value scale per head.
    """
    seq_len = v.shape[2]
    extra = mass.dim() - 2
    value_scale = v.double().abs().amax(dim=(2, 3)).reshape(*v.shape[:2], *([1] * (extra + 1)))
    mass_bound = STATE_RTOL[variant] * ref_mass.abs() + STATE_MASS_ATOL * seq_len
    mass_worst = ((mass.double() - ref_mass).abs() / mass_bound).max().item()
    assert mass_worst <= 1.0, f"A error exceeds tolerance by {mass_worst:.3f}x"
    values_bound = mass_bound.unsqueeze(-1) * value_scale
    values_worst = ((values.double() - ref_values).abs() / values_bound).max().item()
    assert values_worst <= 1.0, f"B error exceeds tolerance by {values_worst:.3f}x"
