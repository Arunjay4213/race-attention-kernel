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

Tensor-core forward (race_fwd_tc.cu, tensor_cores=True). The fragments hold
bf16 values, and each factor contributes its own rounding:
  - V and X (q, k) are bf16 already: exact.
  - phi is rounded to bf16 for the fragments, keys and queries alike:
    phi~ = phi (1 + e) with |e| <= 2^-8 in the normal range (phi >= 2^-126,
    which holds for beta <= 4 at P <= 5 since phi >= sigmoid(-8)^5 ~ 4.5e-18).
  - W enters as hi + lo, both bf16, which differs from W by at most 2^-16 |W|
    per element. The test computes that difference exactly (split_planes) and
    propagates it: dz = x . (W_hi + W_lo - W), |du| <= (1 - u^2 + |dz|) |dz|
    (tanh' = 1 - tanh^2 and |(1 - tanh^2)'| < 1), and each sigmoid factor has
    |d log sigmoid(+-2 beta u) / du| <= 2 |beta|, so the kernel's phi before
    rounding is phi (1 + r) with |r| <= rho = expm1(2 |beta| sum_t |du_t|).
    delta = (1 + 2^-8)(1 + max rho) - 1 bounds the relative perturbation of
    every phi entry. On the emulation inputs max rho is 3e-5 at beta = 1/8,
    up to 3e-4 at beta = 1 and up to 1.1e-3 at beta = 4, against 2^-8 = 3.9e-3
    from the rounding.
  - The reduced B enters the query as hi + lo: relative error <= 2^-16.
  - Everything is accumulated in fp32, as in the fp32 path.

A and Den are summed from the same rounded phi as B and Num, so the kernel
computes O~ = Num~ / Den~ with Num~ = sum_c phi_c (1 + eta_c) B~_c and
Den~ = sum_c phi_c (1 + eta_c) A~_c, where A~_c and B~_c are the sums of
phi_jc (1 + eps_jc) and phi_jc (1 + eps_jc) v_j, |eta|, |eps| <= delta. Since
Num - O Den = 0 exactly,

    Num~ - O Den~ = sum_c phi_c sum_j phi_jc ((1 + eta_c)(1 + eps_jc) - 1) (v_j - O),

so with Den~ >= (1 - delta)^2 Den, |v_j - O| <= |v_j| + |O| and no first-order
approximation:

    |O~ - O| <= (2 delta + delta^2) / (1 - delta)^2 * (M_abs + |O|)
              + 2^-16 (1 + delta) / (1 - delta)^2 * (M_signed + delta M_abs)

with M_abs = sum_c phi_qc sum_j phi_jc |v_j| / Den (the phi-weighted mean of
|v|, per column) and M_signed = sum_c phi_qc |B_c| / Den; the second line is the
hi + lo split of B, the only place where Num and Den use different weights.
The kernel output is then rounded to bf16 and carries the fp32 error of the
fp32 path, so the tensor-core check is

    |O_tc - O_ref| <= 2^-8 |O_ref| + (1 + 2^-8) E_tc + atol,   atol as above.

The fp32 part is covered by the same atol because the tensor-core path has
the same kinds of fp32 sums (per-tile sums of 2048 keys, the same tree, query
sums over at most L * R = 128 corners). The CPU emulation of its data flow
(test_kernel_emulation.py) measures it, against fp64 arithmetic on the same
rounded phi, at 0.1% to 3.9% of atol; the test asserts 10%.

E_tc is far looser than the fp32 path's tolerance: with v ~ N(0, 1) it is
1.2e-3 to 3.2e-3 of max|v| on the emulation inputs, because it holds for
every sign pattern of the roundings, while random roundings largely cancel
over many keys. The emulated error before the final rounding is at most
2e-4 of max|v|, 0.4% to 9.6% of E_tc. The bound is still tight enough to
catch layout, masking and corner-order errors, which are O(1).
A dropped or duplicated key moves O by about |v_j - O| / N, which E_tc only
catches at small N; the beta = 0 tests cover that case, because at beta = 0
every phi is exactly 2^-P, the roundings vanish, and the tensor-core output
must meet the fp32 tolerance.

Bucket sums of the tensor-core path, from the same phi~ (same fp32 terms as
the fp32 path, plus the rounding):

    |A~_c - A_c| <= delta A_c + (fp32 bound),
    |B~_c - B_c| <= delta sum_j phi_jc |v_j| + (fp32 bound).

Backward on the tensor-core totals (forward_train(..., tensor_cores=True)
then backward). The backward is unchanged: it recomputes phi of q and k in
fp32 and uses the saved A~, B~ only through Den, y = B . g and dDen. A~ and B~
are the exact totals of the perturbed key probabilities phi_jc (1 + eps_jc),
|eps| <= delta_K (delta from the keys alone), so the backward's result is
the exact VJP formulas at (A~, B~) plus the backward's own fp32 error. Two
checks follow:
  - Against the fp64 VJP evaluated at the kernel's A~, B~
    (race_backward_reference(..., bucket_totals=...)): the unchanged
    backward tolerance. This checks that the backward reads the tensor-core
    totals exactly as it reads the fp32 ones.
  - Against the exact fp64 VJP: the backward tolerance plus the change of
    the VJP when (A, B) moves to (A~, B~), bounded without approximation
    (tc_totals_backward_perturbation). With Den~ >= (1 - delta) Den,
    |1/Den~ - 1/Den| <= delta / ((1 - delta) Den), and the output shift
    |O~ - O| <= delta / (1 - delta) (M_abs + |O|) (the forward bound with
    exact query weights), the corner gradients move by
      query: |d phi~_r - d phi_r| <= (delta sum_j phi_jr |(v_j - O~) . g|
               + A_r |(O~ - O) . g| + delta |y_r - A_r O . g|) / ((1 - delta) Den),
      key:   |dB~_r - dB_r| <= delta / (1 - delta) sum_i phi_ir |g_i| / Den_i,
             |dA~_r - dA_r| <= sum_i phi_ir (|(O~ - O) . g_i| + delta |g_i . O_i|)
                               / ((1 - delta) Den_i),
    (from d phi_r = sum_j phi_jr (v_j - O) . g / Den and dDen = -g . O / Den),
    and these pass linearly through h, the tanh derivative and W to dq, dk
    and d beta, and through phi_k to dv. The bf16 rounding of the gradient
    applies to the moved value, hence the factor (1 + 2^-8) on the bound.
  On the inputs of test_backward_cuda.py the round trip exceeds the plain
  backward tolerance by up to 7.2x (dq), 2.8x (dk), 11.5x (dv) and 90x
  (d beta), which is why the looser check is needed: the backward tolerance
  is fp32-level, and the totals carry the forward's 2^-8 rounding. With the
  bound added it uses at most 4.3%, 7.3%, 56% and 0.1%, and the check at the
  kernel's totals uses up to 98% of the plain tolerance, as the fp32 path
  does.
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


def output_bound(ref: torch.Tensor, v: torch.Tensor, beta: float) -> torch.Tensor:
    """Per-element bound on |O_kernel - O_ref| for the fp32 path, see module docstring."""
    return BF16_UNIT_ROUNDOFF * ref.abs() + output_atol(v, beta)


def assert_output_close(out: torch.Tensor, ref: torch.Tensor, v: torch.Tensor, beta: float) -> None:
    """Checks a bf16 kernel output against the fp64 reference, see module docstring."""
    error = (out.double() - ref).abs()
    bound = output_bound(ref, v, beta)
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


def reference_backward(grad_o, q, k, v, W, beta: float, bucket_totals=None):
    """fp64 reference gradients and the error scales of the tolerances.

    bucket_totals: optional [B, H, L, R * (d + 1)] totals as forward_train
    returns them; the VJP is then evaluated at those A, B instead of the exact
    bucket sums (see race_backward_reference).

    Returns:
        (dq, dk, dv, dbeta) in fp64, and the scales (dq, dk, dv as [B, H, 1, 1],
        dbeta as a float) described in the module docstring.
    """
    from reference import race_backward_reference

    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    W64 = W.double()
    totals = None
    if bucket_totals is not None:
        corners, head_dim = 1 << W.shape[1], q.shape[-1]
        totals64 = bucket_totals.double()
        totals = (
            totals64[..., corners * head_dim :],
            totals64[..., : corners * head_dim].unflatten(-1, (corners, head_dim)),
        )
    *grads, _, _, h_q_scale, h_k_scale = race_backward_reference(
        grad_o.double(), q.double(), k.double(), v.double(), W64, beta64, return_scales=True,
        bucket_totals=totals,
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
                      name: str, perturbation: torch.Tensor | None = None) -> float:
    """Checks a bf16 kernel gradient against the fp64 reference, scale from reference_backward.

    perturbation: optional bound on how far the inputs of the backward move
    the exact gradient (tc_totals_backward_perturbation). Returns the worst
    error as a fraction of the bound.
    """
    error = (out.double() - ref).abs()
    bound = BF16_UNIT_ROUNDOFF * ref.abs() + GRAD_ATOL_FACTOR * max(1.0, beta) * scale
    if perturbation is not None:
        bound = bound + (1 + BF16_UNIT_ROUNDOFF) * perturbation
    worst = (error / bound).max().item()
    assert worst <= 1.0, f"{name} error exceeds tolerance by {worst:.3f}x"
    return worst


# Tensor-core forward, derived in the last part of the docstring.
TC_PROBS_ROUNDOFF = 2.0**-8
TC_SPLIT_ROUNDOFF = 2.0**-16


def split_planes(W: torch.Tensor) -> torch.Tensor:
    """The planes the tensor-core kernels use, hi + lo (both bf16, round to nearest even), in fp64."""
    hi = W.float().to(torch.bfloat16)
    lo = (W.float() - hi.float()).to(torch.bfloat16)  # the fp32 subtraction is exact
    return hi.double() + lo.double()


def tc_probs_perturbation(q, k, W, beta: float) -> float:
    """delta: bound on the relative deviation of every bf16 phi~ from the fp64 phi, see docstring."""
    W64 = W.double()
    plane_error = split_planes(W) - W64
    worst = 0.0
    for x in (q, k):
        x64 = x.double()
        u = torch.tanh(torch.einsum("...d,lpd->...lp", x64, W64))
        dz = torch.einsum("...d,lpd->...lp", x64, plane_error).abs()
        du = (1.0 - u.square() + dz) * dz
        if du.numel():
            worst = max(worst, torch.expm1(2.0 * abs(beta) * du.sum(dim=-1)).max().item())
    return (1.0 + TC_PROBS_ROUNDOFF) * (1.0 + worst) - 1.0


def tc_statistics(q, k, v, W, beta: float):
    """fp64 reference quantities for the tensor-core bounds.

    Returns:
        delta (float), A [B, H, L, R], B [B, H, L, R, d], sum_j phi_jc |v_j| as
        [B, H, L, R, d], and E_tc [B, H, N, d] (the docstring's bound before
        the final bf16 rounding and the fp32 atol).
    """
    from reference import corner_probs_bernoulli, race_forward_reference

    delta = tc_probs_perturbation(q, k, W, beta)
    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    q64, k64, v64, W64 = q.double(), k.double(), v.double(), W.double()
    phi_k = corner_probs_bernoulli(k64, W64, beta64)
    phi_q = corner_probs_bernoulli(q64, W64, beta64)
    mass = phi_k.sum(dim=2)
    weighted_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_k, v64)
    weighted_abs_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_k, v64.abs())
    den = torch.einsum("bhnlr,bhlr->bhn", phi_q, mass).unsqueeze(-1)
    mean_abs = torch.einsum("bhnlr,bhlrd->bhnd", phi_q, weighted_abs_values) / den
    mean_signed = torch.einsum("bhnlr,bhlrd->bhnd", phi_q, weighted_values.abs()) / den
    out = race_forward_reference(q64, k64, v64, W64, beta64)
    consistent = (2 * delta + delta**2) / (1 - delta) ** 2 * (mean_abs + out.abs())
    split = TC_SPLIT_ROUNDOFF * (1 + delta) / (1 - delta) ** 2 * (mean_signed + delta * mean_abs)
    return delta, mass, weighted_values, weighted_abs_values, consistent + split


def output_bound_tc(ref: torch.Tensor, rounding_bound: torch.Tensor, v: torch.Tensor,
                    beta: float) -> torch.Tensor:
    """Per-element bound on |O_tc - O_ref|; rounding_bound is E_tc from tc_statistics."""
    return (BF16_UNIT_ROUNDOFF * ref.abs() + (1 + BF16_UNIT_ROUNDOFF) * rounding_bound
            + output_atol(v, beta))


def assert_output_close_tc(out: torch.Tensor, ref: torch.Tensor, rounding_bound: torch.Tensor,
                           v: torch.Tensor, beta: float) -> float:
    """Checks a tensor-core output against the fp64 reference; rounding_bound is E_tc.

    Returns the worst error as a fraction of the bound.
    """
    error = (out.double() - ref).abs()
    worst = (error / output_bound_tc(ref, rounding_bound, v, beta)).max().item()
    assert worst <= 1.0, f"tensor-core output error exceeds tolerance by {worst:.3f}x"
    return worst


def assert_buckets_close_tc(
    mass: torch.Tensor, weighted_values: torch.Tensor, ref_mass: torch.Tensor,
    ref_weighted_values: torch.Tensor, ref_weighted_abs_values: torch.Tensor, v: torch.Tensor,
    delta: float,
) -> None:
    """Checks the tensor-core A and B against the fp64 reference, see docstring."""
    seq_len = v.shape[2]
    value_scale = v.double().abs().amax(dim=(2, 3), keepdim=True).unsqueeze(-1)
    fp32_mass_bound = BUCKET_RTOL * ref_mass + BUCKET_MASS_ATOL * seq_len
    mass_bound = delta * ref_mass + fp32_mass_bound
    mass_worst = ((mass.double() - ref_mass).abs() / mass_bound).max().item()
    assert mass_worst <= 1.0, f"tensor-core A error exceeds tolerance by {mass_worst:.3f}x"
    values_bound = delta * ref_weighted_abs_values + fp32_mass_bound.unsqueeze(-1) * value_scale
    values_worst = ((weighted_values.double() - ref_weighted_values).abs() / values_bound).max().item()
    assert values_worst <= 1.0, f"tensor-core B error exceeds tolerance by {values_worst:.3f}x"


def assert_beta_grad_close(got: torch.Tensor, ref: torch.Tensor, scale: float, beta: float,
                           perturbation: float = 0.0) -> float:
    """Checks the fp32 d beta against the fp64 reference, scale from reference_backward.

    Returns the error as a fraction of the bound.
    """
    error = abs(got.item() - ref.item())
    bound = BETA_GRAD_TOL_FACTOR * max(1.0, beta) * scale + perturbation
    assert error <= bound, f"d beta error {error:.3e} exceeds {bound:.3e}"
    return error / bound


def tc_totals_backward_perturbation(grad_o, q, k, v, W, beta: float):
    """Bounds on how far the exact VJP moves when A, B are the tensor-core totals.

    See the last part of the module docstring. All in fp64.

    Returns:
        bounds for dq, dk, dv (shapes of q, k, v) and for d beta (a float).
    """
    from reference import corner_signs, soft_hash

    delta = tc_probs_perturbation(k, k, W, beta)  # only the keys enter the totals
    ratio = delta / (1.0 - delta)
    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    g, q64, k64, v64, W64 = (t.double() for t in (grad_o, q, k, v, W))
    g_abs, W_abs = g.abs(), W64.abs()
    phi_q, u_q, plus_q, minus_q = soft_hash(q64, W64, beta64)
    phi_k, u_k, plus_k, minus_k = soft_hash(k64, W64, beta64)

    mass = phi_k.sum(dim=2)  # [B, H, L, R]
    weighted_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_k, v64)
    weighted_abs_values = torch.einsum("bhnlr,bhnd->bhlrd", phi_k, v64.abs())
    den = torch.einsum("bhnlr,bhlr->bhn", phi_q, mass)
    out = torch.einsum("bhnlr,bhlrd->bhnd", phi_q, weighted_values) / den[..., None]
    mean_abs = torch.einsum("bhnlr,bhlrd->bhnd", phi_q, weighted_abs_values) / den[..., None]
    out_shift = ratio * (mean_abs + out.abs())  # bound on |O~ - O|
    shift_dot = (g_abs * out_shift).sum(dim=-1)  # bound on |(O~ - O) . g|, [B, H, N]
    out_dot = (g * out).sum(dim=-1)  # g . O

    # Query side: sum_j phi_jr |(v_j - O~) . g| <= B^abs_r . |g| + A_r |O~| . |g|.
    y = torch.einsum("bhlrd,bhnd->bhnlr", weighted_values, g)
    spread = torch.einsum("bhlrd,bhnd->bhnlr", weighted_abs_values, g_abs) + mass[
        :, :, None
    ] * ((out.abs() + out_shift) * g_abs).sum(dim=-1)[..., None, None]
    centered = (y - mass[:, :, None] * out_dot[..., None, None]).abs()
    grad_phi_q_shift = (
        delta * spread + mass[:, :, None] * shift_dot[..., None, None] + delta * centered
    ) / ((1.0 - delta) * den[..., None, None])

    # Key side.
    grad_values_abs = torch.einsum("bhnlr,bhnd->bhlrd", phi_q, g_abs / den[..., None])
    grad_values_shift = ratio * grad_values_abs  # bound on |dB~ - dB|
    grad_mass_shift = torch.einsum(
        "bhnlr,bhn->bhlr", phi_q, (shift_dot + delta * out_dot.abs()) / ((1.0 - delta) * den)
    )
    dv_shift = torch.einsum("bhnlr,bhlrd->bhnd", phi_k, grad_values_shift)
    grad_phi_k_shift = (
        torch.einsum("bhlrd,bhnd->bhnlr", grad_values_shift, v64.abs()) + grad_mass_shift[:, :, None]
    )

    def through_hash(grad_phi_shift, phi, u, prob_plus, prob_minus):
        # |c_rt| is 2 sigmoid(-2 beta u) or 2 sigmoid(2 beta u) (corner_coefficient).
        plus = corner_signs(W.shape[1], u.dtype, u.device) > 0
        coef_abs = torch.where(plus, 2.0 * prob_minus[..., None, :], 2.0 * prob_plus[..., None, :])
        h_shift = torch.einsum("...lr,...lrp->...lp", phi * grad_phi_shift, coef_abs)
        dx_shift = torch.einsum("...lp,lpd->...d", abs(beta) * h_shift * (1.0 - u * u), W_abs)
        return dx_shift, (u.abs() * h_shift).sum().item()

    dq_shift, beta_shift_q = through_hash(grad_phi_q_shift, phi_q, u_q, plus_q, minus_q)
    dk_shift, beta_shift_k = through_hash(grad_phi_k_shift, phi_k, u_k, plus_k, minus_k)
    return dq_shift, dk_shift, dv_shift, beta_shift_q + beta_shift_k
