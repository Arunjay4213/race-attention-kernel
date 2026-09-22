# Backward pass of non-causal RACE Attention

This document derives the vector-Jacobian product (VJP) that `kernels/race_bwd.cu` implements, step by step, in the same decomposition the kernels use.
Every formula below is checked numerically in fp64 by `tests/test_backward_reference.py` (section 9).
`reference.py::race_backward_reference` is this document written as PyTorch.

## 1. The forward, as the backward sees it

Everything is per stream (one batch element and one head), except β, which is a single scalar shared by all streams.
Indices: i runs over queries, j over keys, l over the L tables, r over the R = 2^P corners, t over the P planes, c over the d columns.

For a row x (a query or a key) and table l:

- zₜ = w_{l,t} · x and uₜ = tanh(zₜ).
- pₜ = σ(2βuₜ), and 1 − pₜ = σ(−2βuₜ), which the kernels evaluate directly (never as 1 − pₜ).
- sᵣₜ ∈ {−1, +1} is the sign of corner r on plane t (plane 0 is the most significant bit of r).
- φ[r] = ∏ₜ σ(2β sᵣₜ uₜ), that is pₜ where sᵣₜ = +1 and 1 − pₜ where sᵣₜ = −1.

φ_l(qᵢ) is written φQ_li and φ_l(kⱼ) is written φK_lj.
The forward then computes:

- A_l[r] = ∑ⱼ φK_lj[r] and B_l[r, :] = ∑ⱼ φK_lj[r] vⱼ.
- Numᵢ = ∑ₗ ∑ᵣ φQ_li[r] B_l[r, :] and Denᵢ = ∑ₗ ∑ᵣ φQ_li[r] A_l[r].
- Oᵢ = Numᵢ / Denᵢ.

The paper's 1/L factors on Num and Den cancel in O, so they never appear.
W is a fixed random buffer, so there is no dW.
The backward receives gᵢ = ∂ℒ/∂Oᵢ (a d-vector per query) and returns ∂ℒ/∂qᵢ, ∂ℒ/∂kⱼ, ∂ℒ/∂vⱼ and ∂ℒ/∂β.
Below, "dX" is short for ∂ℒ/∂X.

## 2. Through the division

Oᵢ = Numᵢ / Denᵢ gives ∂Oᵢ/∂Numᵢ = I / Denᵢ and ∂Oᵢ/∂Denᵢ = −Numᵢ / Denᵢ² = −Oᵢ / Denᵢ.
So

- dNumᵢ = gᵢ / Denᵢ,
- dDenᵢ = −(gᵢ · Oᵢ) / Denᵢ = −(gᵢ · Numᵢ) / Denᵢ².

The second form is the one the kernel uses, because gᵢ · Numᵢ comes for free (section 3).

## 3. Query side: the gradient with respect to φQ

Numᵢ and Denᵢ are linear in φQ_li[r], so

  dφQ_li[r] = B_l[r, :] · dNumᵢ + A_l[r] dDenᵢ.

Define the per-token, per-corner dot product

  y_l[r] = B_l[r, :] · gᵢ.

Then B_l[r, :] · dNumᵢ = y_l[r] / Denᵢ, and the numerator of dDenᵢ is

  gᵢ · Numᵢ = ∑ₗ ∑ᵣ φQ_li[r] (B_l[r, :] · gᵢ) = ∑ₗ ∑ᵣ φQ_li[r] y_l[r] =: gNumᵢ.

So the query side needs, per token, only y (R·L dot products of length d), Den = ∑ φQ A and gNum = ∑ φQ y:

  dDenᵢ = −gNumᵢ / Denᵢ²,
  dφQ_li[r] = y_l[r] / Denᵢ + A_l[r] dDenᵢ.

O is never needed.
Written differently, dφQ_li[r] = A_l[r] gᵢ · (B_l[r, :] / A_l[r] − Oᵢ) / Denᵢ: moving mass into a bucket helps if that bucket's mean value points along gᵢ more than the current output does.

## 4. Bucket gradients: the reduction over queries

Numᵢ and Denᵢ are also linear in B and A, so

  dB_l[r, :] = ∑ᵢ φQ_li[r] dNumᵢ = ∑ᵢ φQ_li[r] gᵢ / Denᵢ,
  dA_l[r] = ∑ᵢ φQ_li[r] dDenᵢ.

This has exactly the shape of the forward's bucket build, with the hashed rows kⱼ replaced by qᵢ, the value rows vⱼ replaced by gᵢ / Denᵢ, and each token's unit mass replaced by the weight dDenᵢ.
The kernels therefore reuse the forward's build (`bucket_build_kernel<D, P, true>`) with a per-token weight pair (1/Denᵢ, dDenᵢ), and the forward's deterministic tree reduce.

## 5. Key side

B_l[r, :] = ∑ⱼ φK_lj[r] vⱼ and A_l[r] = ∑ⱼ φK_lj[r] give

  dvⱼ = ∑ₗ ∑ᵣ φK_lj[r] dB_l[r, :],
  dφK_lj[r] = dB_l[r, :] · vⱼ + dA_l[r].

dvⱼ has the shape of the forward's Num (a φ-weighted sum of bucket rows), and dB_l[r, :] · vⱼ has the shape of y.

## 6. Through φ: the Jacobian in product form

Only factor t of φ[r] = ∏ₜ σ(2β sᵣₜ uₜ) depends on uₜ, and d/dz log σ(z) = 1 − σ(z) = σ(−z), so

  ∂ log φ[r] / ∂uₜ = 2β sᵣₜ σ(−2β sᵣₜ uₜ).

The two cases are:

- sᵣₜ = +1: 2β σ(−2βuₜ) = 2β (1 − pₜ).
- sᵣₜ = −1: −2β σ(2βuₜ) = −2β pₜ.

Both equal β (sᵣₜ − (2pₜ − 1)): for s = +1, β (1 − 2pₜ + 1) = 2β (1 − pₜ), and for s = −1, β (−1 − 2pₜ + 1) = −2β pₜ.
Hence

  ∂φ[r] / ∂uₜ = β φ[r] cᵣₜ,  cᵣₜ = sᵣₜ − (2pₜ − 1) = sᵣₜ − tanh(βuₜ),

using 2σ(2a) − 1 = tanh(a).

Why β and not 2β: the factor 2 inside σ(2βu) is used up by the two-valued sign.
The jump between the two cases, 2β(1 − p) − (−2βp) = 2β, is spread over a sign whose range is sᵣₜ ∈ {−1, +1}, a width of 2, so the slope per unit of sign is β.
The softmax form gives the same result directly.
φ = softmax_r(β ∑ₜ sᵣₜ uₜ), and the softmax Jacobian is ∂φ[r]/∂uₜ = β φ[r] (sᵣₜ − ∑_{r'} φ[r'] s_{r't}).
The φ-weighted mean sign ∑_{r'} φ[r'] s_{r't} is the marginal pₜ − (1 − pₜ) = 2pₜ − 1, because the product form makes the planes independent Bernoullis.
`test_phi_jacobian_is_beta_not_two_beta` checks the formula against autograd of the softmax definition and checks that 2β is wrong.

The β derivative follows the same way, because β enters only through 2β sᵣₜ uₜ:

  ∂ log φ[r] / ∂β = ∑ₜ 2 sᵣₜ uₜ σ(−2β sᵣₜ uₜ) = ∑ₜ uₜ cᵣₜ,  so  ∂φ[r] / ∂β = φ[r] ∑ₜ uₜ cᵣₜ.

Two properties are used below:

- ∑ᵣ φ[r] cᵣₜ = 0, because ∑ᵣ φ[r] sᵣₜ = 2pₜ − 1 and ∑ᵣ φ[r] = 1.
- cᵣₜ is evaluated as 2σ(−2βuₜ) for s = +1 and −2σ(2βuₜ) for s = −1 (`corner_coefficient`).
  Both sigmoids come from the same exp as in the forward, so c has full relative precision, while 1 − (2pₜ − 1) would cancel when pₜ is close to 1.

## 7. From ∂ℒ/∂φ to the rows and to β

For one token, one table and a corner gradient γ[r] = ∂ℒ/∂φ[r] (γ = dφQ on the query side, dφK on the key side), define

  hₜ = ∑ᵣ φ[r] γ[r] cᵣₜ.

The chain rule through φ, tanh and the projection gives

- ∂ℒ/∂uₜ = ∑ᵣ γ[r] β φ[r] cᵣₜ = β hₜ,
- ∂ℒ/∂zₜ = β hₜ (1 − uₜ²), since tanh′(z) = 1 − tanh(z)²,
- ∂ℒ/∂x = ∑ₗ ∑ₜ ∂ℒ/∂z_{l,t} w_{l,t}, summed over every table the row was hashed with,
- the contribution to ∂ℒ/∂β is ∑ᵣ γ[r] φ[r] ∑ₜ uₜ cᵣₜ = ∑ₜ uₜ hₜ.

Writing the β term as ∑ₜ uₜ hₜ, rather than as ∑ₜ uₜ (∂ℒ/∂uₜ) / β, avoids a division by β, which may be 0.
Because ∑ᵣ φ[r] cᵣₜ = 0, adding a constant to γ does not change h: only differences of γ between corners reach the rows.

Putting the sides together:

- dqᵢ comes from γ = dφQ_i (section 3), for every table.
- dkⱼ comes from γ = dφK_j (section 5), for every table.
- dvⱼ is section 5.
- dβ is the sum of ∑ₜ uₜ hₜ over every query and every key, every table, and, since β is one scalar, every batch element and head.

Two consequences that the tests confirm:

- At β = 0 every chain to q and k carries the factor β, so dq = dk = 0 exactly (the kernels produce exact zeros).
- O is even in β, because φ(−β)[r] = φ(β)[r̄] with r̄ the complementary corner, and ∑ᵣ φQ[r] φK[r] does not depend on how corners are labelled.
  So dβ is odd in β and 0 at β = 0, and dq, dk, dv are even in β.

## 8. What is saved and what is recomputed

Saved by `RaceAttentionFunction` (`ops.py`):

- q, k, v, W, β (the inputs; autograd would keep q, k, v alive for any attention).
- The reduced bucket sums A and B, as `bucket_totals` [B, H, L, R·(d+1)] fp32: L·R·(d+1) floats per stream, 33 KB per stream at P = 4, L = 4, d = 128.
  This is the forward workspace's tile 0, cloned so the per-tile workspace can be freed.

Not saved:

- O.
  It is never needed: dDen uses gNum = ∑ φQ y, and y is needed anyway for dφQ, so avoiding O costs no extra length-d work.
  It also avoids using the bf16-rounded O in gᵢ · Oᵢ (FlashAttention's rowsum(dO ∘ O) does use the saved bf16 O); here the fp32 value comes for free.
- Den per token.
  It is recomputed as ∑ φQ A, R·L multiply-adds per token in a loop that already runs over the same φQ.
  Saving it would cost 4 bytes per token of activation memory and one more input to read, to save work that is noise next to the R·L·d of y.
- φ, u and the sigmoids.
  They are recomputed from q and k: P length-d projections per table per token, against R·d for each of the bucket passes.

Temporary memory of the backward (one fp32 buffer, `backward_workspace_floats`):

- the dA/dB partials, the same size as the forward workspace ([tiles, B·H, L, R·(d+1)], tiles = ⌈N / 2048⌉),
- the per-token weights (1/Denᵢ, dDenᵢ), 8 bytes per query,
- the dβ partials, 2·B·H·⌈N / 1024⌉ floats.

Second derivatives are not supported.
The backward is a kernel on saved intermediates, not a composition of differentiable operations, so a double backward through it would be silently wrong.
`backward` is decorated with `torch.autograd.function.once_differentiable`, so attempting it raises an error instead.

## 9. Numerical check of the derivation

`tests/test_backward_reference.py` runs on CPU in fp64 in about 2 seconds:

- `race_backward_reference` against `torch.autograd` of `race_forward_reference`, for every P ∈ {1, 2, 4, 5}, L ∈ {1, 2, 4}, d ∈ {64, 128} and β ∈ {1/√128, 1, 4}, with B = H = 2 and N = 23.
- The formulas of section 6 against autograd of the softmax definition, including the check that 2β is wrong.
- `torch.autograd.gradcheck` of an autograd function whose backward is `race_backward_reference`, which compares the decomposition with finite differences independently of autograd of the forward.
- β = 0, the β symmetry of section 7, and a query whose Den underflows (section 10).

Largest errors over the grid, relative to the largest entry of each autograd gradient (for dβ, relative to ∑|uₜhₜ| because dβ is a sum with heavy cancellation):

| gradient | max relative error |
|---|---|
| dq | 7.2 × 10⁻¹⁵ |
| dk | 6.8 × 10⁻¹⁵ |
| dv | 4.1 × 10⁻¹⁶ |
| dβ | 2.0 × 10⁻¹⁵ |

These are fp64 roundoff, so the decomposition is the exact VJP.

## 10. Edge cases and conditioning

- Den underflow.
  For very large β, a query's Den can underflow to exactly 0 in fp32; the forward then defines Oᵢ = 0, which does not depend on any input.
  The backward sets 1/Denᵢ = dDenᵢ = 0 for such a query, so it contributes nothing to dq, dA, dB or dβ, which matches autograd of the reference's `torch.where`.
- Saturated tanh.
  zₜ = w · x has a standard deviation of about √d, so most tokens have |zₜ| ≫ 1 and 1 − uₜ² ≈ 4 exp(−2|zₜ|).
  Many dq and dk rows are therefore tiny, and fp32 evaluates 1 − u² with an absolute (not relative) error near 2⁻²³.
  The test tolerances scale with ∂ℒ/∂u instead of ∂ℒ/∂x for this reason (`tests/numerics.py`).
- Cancellation in dφQ.
  y_l[r]/Den and A_l[r] dDen nearly cancel when a bucket's mean value is close to Oᵢ, for example when v has a large common offset.
  The fp32 error of dq then scales with the two terms, not with their difference.
  Softmax attention's backward has the same property (dS = P ∘ (dP − rowsum(dO ∘ O))), and the tolerances account for it the same way.

## 11. From the derivation to the kernels

| step | formula | kernel |
|---|---|---|
| hash q, y, Den, gNum, 1/Den, dDen, dφQ, h, dq, dβ terms | sections 2, 3, 6, 7 | `query_grad_kernel` |
| dA, dB per 2048-query tile | section 4 | `bucket_build_kernel<D, P, true>` |
| dA, dB totals | section 4 | `tree_reduce_pass_kernel` (forward's) |
| hash k, dv, dφK, h, dk, dβ terms | sections 5, 6, 7 | `key_grad_kernel` |
| dβ | section 7 | `beta_grad_reduce_kernel` |

Why the query side is two kernels: dB needs 1/Denᵢ and dDenᵢ, which sum over every table.
The forward's build keeps one table's R·d accumulators in registers per CTA; holding all tables at once would need up to L·R·d / 256 = 64 accumulators per thread at P = 5, L = 4, d = 128.
So `query_grad_kernel` writes the per-token pair (1/Denᵢ, dDenᵢ), and the build reads q and dO a second time (the L CTAs of a tile share that read through L2).

Each per-token kernel runs one warp per token.
The R dot products of length d (y on the query side, dB · v on the key side) are computed as lane partials over each lane's d/32 columns, followed by a reduce-scatter that leaves lane r holding corner r (`corner_reduce_scatter`, R − 1 + 5 − P shuffles).
Sums over corners (h, Den, gNum) are butterflies over the low P lane bits.
Every sum has a fixed order (serial per thread, a fixed pairwise tree over tiles, fixed warp order within a CTA, and a fixed one-CTA tree for dβ), and there are no atomics, so the gradients are bitwise reproducible.
