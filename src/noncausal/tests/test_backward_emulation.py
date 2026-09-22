"""CPU emulation of the backward kernels' index math and fp32 data flow.

Mirrors kernels/race_bwd.cu and the reduce-scatter in race_common.cuh, the
way test_kernel_emulation.py mirrors the forward:
  - corner_reduce_scatter, run lane by lane, delivers the sum of partial(r)
    for r = lane & (R - 1) on every lane and calls partial in order 0..R-1,
  - corner_allreduce_sum sums one copy of the R corner values,
  - the per-token kernels' shared-memory layouts are aligned, disjoint and
    fit the smallest opt-in limit, and the backward workspace regions are
    aligned and disjoint,
  - the whole backward in fp32, through the real workspace layout, the
    weighted bucket build and the tree reduce, stays within 10% of the
    absolute tolerance the CUDA tests use (numerics.py).

The constants and formulas below mirror race_bwd.cu and must be kept in sync.
"""
from __future__ import annotations

import pytest
import torch

from numerics import (
    BETA_GRAD_TOL_FACTOR,
    GRAD_ATOL_FACTOR,
    assert_beta_grad_close,
    assert_grad_close,
    make_bf16_grad_output,
    make_bf16_inputs,
    reference_backward,
)
from test_kernel_emulation import (
    BUILD_TILE_TOKENS,
    SM89_MAX_DYNAMIC_SMEM_BYTES,
    WARPS,
    build_tile_partials,
    butterfly_sum,
    emulate_forward,
    sigmoid_pair_fp32,
    tree_reduce,
)

QUERY_TILE_TOKENS = 1024
HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 3, 4, 5]


def round_up4(n: int) -> int:
    return (n + 3) & ~3


# ---------------------------------------------------------------------------
# Warp-level building blocks
# ---------------------------------------------------------------------------


def reduce_scatter_lanes(partials: list[list[float]], num_planes: int, add=lambda a, b: a + b):
    """corner_reduce_scatter run on 32 lanes.

    Args:
        partials: partials[lane][r], this lane's partial for corner r.

    Returns:
        (result per lane, call order of lane 0).
    """
    corners = 1 << num_planes
    group = 1 << (num_planes // 2)
    groups = corners // group
    calls: list[int] = []

    def fold_pairs(values_per_lane, count, lane_bit):
        out = []
        for lane in range(32):
            upper = bool(lane & lane_bit)
            partner = values_per_lane[lane ^ lane_bit]
            mine = values_per_lane[lane]
            new = list(mine)
            for i in range(count):
                send_partner = partner[2 * i] if not upper else partner[2 * i + 1]
                # The partner's upper bit is the opposite of ours, so what it
                # sends is the half we keep.
                keep = mine[2 * i + 1] if upper else mine[2 * i]
                new[i] = add(keep, send_partner)
            out.append(new)
        return out

    group_sums = [[None] * groups for _ in range(32)]
    for g in range(groups):
        values = [[partials[lane][g * group + i] for i in range(group)] for lane in range(32)]
        calls.extend(g * group + i for i in range(group))
        width = 1
        while width < group:
            values = fold_pairs(values, group // (2 * width), width)
            width <<= 1
        for lane in range(32):
            group_sums[lane][g] = values[lane][0]
    width = 1
    while width < groups:
        group_sums = fold_pairs(group_sums, groups // (2 * width), width * group)
        width <<= 1
    result = [group_sums[lane][0] for lane in range(32)]
    offset = corners
    while offset < 32:
        result = [add(result[lane], result[lane ^ offset]) for lane in range(32)]
        offset <<= 1
    return result, calls


@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_reduce_scatter_delivers_each_corner_sum(num_planes):
    corners = 1 << num_planes
    # Multisets of (lane, corner) make every mix-up visible.
    partials = [[frozenset({(lane, r)}) for r in range(corners)] for lane in range(32)]
    result, calls = reduce_scatter_lanes(partials, num_planes, add=lambda a, b: a | b)
    assert calls == list(range(corners))
    for lane in range(32):
        assert result[lane] == {(other, lane & (corners - 1)) for other in range(32)}


@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_reduce_scatter_live_values(num_planes):
    # The register argument in the kernel comment: at most G + R / G live.
    corners = 1 << num_planes
    group = 1 << (num_planes // 2)
    assert group + corners // group <= 12
    shuffles = corners // group * (group - 1) + (corners // group - 1) + (5 - num_planes)
    assert shuffles == corners - 1 + 5 - num_planes


@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_corner_allreduce_sums_one_copy(num_planes):
    corners = 1 << num_planes
    values = [float(1 << (lane & (corners - 1))) for lane in range(32)]
    offset = corners // 2
    while offset > 0:
        values = [values[lane] + values[lane ^ offset] for lane in range(32)]
        offset //= 2
    assert values == [float((1 << corners) - 1)] * 32


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def token_scratch_floats(num_planes: int, num_tables: int, with_corner_grads: bool) -> list[int]:
    """Region sizes of TokenScratch in order: probs, hash, grad_z, corner_grads."""
    corners = 1 << num_planes
    sizes = [
        round_up4(num_tables * corners),
        round_up4(num_tables * 3 * num_planes),
        round_up4(num_tables * num_planes),
    ]
    if with_corner_grads:
        sizes.append(round_up4(num_tables * corners))
    return sizes


def totals_smem_regions(head_dim, num_planes, num_tables, with_corner_grads):
    """(offset, length) in floats of each TotalsSmem region, then each warp's scratch."""
    corners = 1 << num_planes
    buckets = (0, num_tables * corners * head_dim)
    mass = (buckets[1], num_tables * corners)
    planes = (mass[0] + round_up4(mass[1]), num_tables * num_planes * head_dim)
    scratch_begin = planes[0] + planes[1]
    stride = sum(token_scratch_floats(num_planes, num_tables, with_corner_grads))
    regions = [buckets, mass, planes]
    for warp in range(WARPS):
        offset = scratch_begin + warp * stride
        for size in token_scratch_floats(num_planes, num_tables, with_corner_grads):
            regions.append((offset, size))
            offset += size
    return regions


@pytest.mark.parametrize("with_corner_grads", [True, False])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("num_tables", [1, 2, 3, 4])
def test_per_token_kernel_smem_layout(head_dim, num_planes, num_tables, with_corner_grads):
    regions = totals_smem_regions(head_dim, num_planes, num_tables, with_corner_grads)
    for (offset, length), (next_offset, _) in zip(regions[:-1], regions[1:], strict=True):
        assert offset + length <= next_offset
    for offset, _ in regions:
        assert offset % 4 == 0
    static_bytes = 4 * WARPS  # warp_beta_s
    total_bytes = 4 * (regions[-1][0] + regions[-1][1]) + static_bytes
    assert total_bytes <= SM89_MAX_DYNAMIC_SMEM_BYTES


@pytest.mark.parametrize("seq_len", [1, 17, 2048, 2049, 10**6 + 3])
@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 1, 1), (128, 5, 4), (64, 3, 3)])
def test_backward_workspace_regions(seq_len, head_dim, num_planes, num_tables):
    batch_heads = 3
    tiles = -(-seq_len // BUILD_TILE_TOKENS)
    query_tiles = -(-seq_len // QUERY_TILE_TOKENS)
    grad_buckets = tiles * batch_heads * num_tables * (1 << num_planes) * (head_dim + 1)
    token_weights = 2 * batch_heads * seq_len
    beta_partials = 2 * batch_heads * query_tiles
    # race_backward's carve: grad buckets, token weights, query-side then
    # key-side beta partials; backward_workspace_floats is the sum.
    weights_offset = round_up4(grad_buckets)
    partials_offset = weights_offset + round_up4(token_weights)
    key_partials_offset = partials_offset + batch_heads * query_tiles
    total = partials_offset + beta_partials
    assert weights_offset % 4 == 0 and partials_offset % 4 == 0
    assert weights_offset >= grad_buckets and partials_offset >= weights_offset + token_weights
    assert key_partials_offset + batch_heads * query_tiles == total


# ---------------------------------------------------------------------------
# fp32 data flow
# ---------------------------------------------------------------------------


def soft_hash_fp32(x: torch.Tensor, W: torch.Tensor, beta: float):
    """hash_to_scratch in fp32. x [BH, N, D] -> phi [BH, N, L, R], u, p+, p- [BH, N, L, P].

    Lane partial sums, the warp butterfly and the running product over planes
    follow the kernel. The kernel's 1 / (1 + e) is one Newton step from the
    hardware approximation, within 1 ulp of the division used here.
    """
    head_dim = x.shape[-1]
    num_planes = W.shape[1]
    vec = head_dim // 32
    products = x[:, :, None, None, :] * W
    lane_parts = products.reshape(*products.shape[:-1], 32, vec)
    partial = lane_parts[..., 0]
    for j in range(1, vec):
        partial = partial + lane_parts[..., j]
    u = torch.tanh(butterfly_sum(partial))
    prob_plus, prob_minus = sigmoid_pair_fp32(2.0 * beta * u)
    corners = torch.arange(1 << num_planes)
    phi = torch.ones(*u.shape[:-1], 1 << num_planes)
    for t in range(num_planes):
        plus = ((corners >> (num_planes - 1 - t)) & 1).bool()
        phi = phi * torch.where(plus, prob_plus[..., t : t + 1], prob_minus[..., t : t + 1])
    return phi, u, prob_plus, prob_minus


def hash_grad_fp32(weighted, u, prob_plus, prob_minus, W, beta: float):
    """hash_grad_to_scratch and projection_grads_to_row in fp32.

    Args:
        weighted: [BH, N, L, R], phi * dphi. u, prob_plus, prob_minus: [BH, N, L, P].

    Returns:
        dx [BH, N, D] and the beta terms u_t h_t as [BH, N, L * P] in the
        kernel's accumulation order (table-major, plane-minor).
    """
    num_tables, num_planes, _ = W.shape
    corners = torch.arange(1 << num_planes)
    plus = ((corners[:, None] >> (num_planes - 1 - torch.arange(num_planes))) & 1).bool()  # [R, P]
    coef = torch.where(plus, 2.0 * prob_minus[..., None, :], -2.0 * prob_plus[..., None, :])
    h = (weighted[..., None] * coef).sum(-2)  # [BH, N, L, P]
    grad_z = beta * h * (1.0 - u * u)
    dx = torch.zeros(*weighted.shape[:2], W.shape[-1])
    for table in range(num_tables):
        for t in range(num_planes):
            dx = dx + grad_z[..., table, t, None] * W[table, t]
    return dx, (u * h).flatten(-2)


def reduce_beta_partials(beta_terms: torch.Tensor) -> torch.Tensor:
    """One running sum per warp over its tokens' terms, warp-order sums per CTA.

    Args:
        beta_terms: [BH, N, T], each token's terms in accumulation order.

    Returns:
        [BH * tiles] CTA partials in the [bh][tile] layout.
    """
    batch_heads, seq_len, num_terms = beta_terms.shape
    tiles = -(-seq_len // QUERY_TILE_TOKENS)
    # Token tile * 1024 + step * WARPS + warp is handled by `warp` at `step`;
    # padding tokens add exact zeros.
    padded = torch.nn.functional.pad(beta_terms, (0, 0, 0, tiles * QUERY_TILE_TOKENS - seq_len))
    padded = padded.reshape(batch_heads, tiles, QUERY_TILE_TOKENS // WARPS, WARPS, num_terms)
    warp_sums = torch.zeros(batch_heads, tiles, WARPS)
    for step in range(QUERY_TILE_TOKENS // WARPS):
        for term in range(num_terms):
            warp_sums = warp_sums + padded[:, :, step, :, term]
    cta = warp_sums[..., 0]
    for warp in range(1, WARPS):
        cta = cta + warp_sums[..., warp]
    return cta.flatten()


def one_cta_reduce(partials: torch.Tensor) -> torch.Tensor:
    threads = 256
    sums = torch.zeros(threads)
    for i in range(partials.numel()):
        sums[i % threads] = sums[i % threads] + partials[i]
    width = threads // 2
    while width > 0:
        sums[:width] = sums[:width] + sums[width : 2 * width]
        width //= 2
    return sums[0]


def emulate_backward(grad_o, q, k, v, W, beta: float, build_tile: int,
                     dropped_token: int | None = None):
    """fp32 backward through the kernels' decomposition and workspace layout.

    dropped_token leaves that query token out of the dA/dB build, the kind of
    masking bug the tolerances must be able to see.

    Returns dq, dk, dv as fp32 [B, H, N, D] (before the bf16 rounding) and d beta.
    """
    batch, heads, seq_len, head_dim = q.shape
    num_tables, num_planes, _ = W.shape
    corners = 1 << num_planes
    batch_heads = batch * heads
    _, _, mass, buckets = emulate_forward(q, k, v, W, beta, build_tile)
    mass = mass.reshape(batch_heads, num_tables, corners)
    buckets = buckets.reshape(batch_heads, num_tables, corners, head_dim)

    def rows(t):
        return t.float().reshape(batch_heads, seq_len, head_dim)

    # Query side.
    grad = rows(grad_o)
    phi_q, u_q, plus_q, minus_q = soft_hash_fp32(rows(q), W, beta)
    y = torch.einsum("blrd,bnd->bnlr", buckets, grad)
    den_lanes = (phi_q * mass[:, None]).sum(2)  # per corner, summed over tables
    num_lanes = (phi_q * y).sum(2)
    den = den_lanes.sum(-1)
    grad_num = num_lanes.sum(-1)
    inv_den = torch.where(den == 0, torch.zeros_like(den), 1.0 / den)
    grad_den = -(grad_num * inv_den) * inv_den
    grad_phi_q = y * inv_den[..., None, None] + mass[:, None] * grad_den[..., None, None]
    dq, beta_terms_q = hash_grad_fp32(phi_q * grad_phi_q, u_q, plus_q, minus_q, W, beta)

    # Weighted bucket build (value rows g / Den, masses phi * dDen) and the tree.
    num_tiles = -(-seq_len // build_tile)
    padded = num_tiles * build_tile
    pad = padded - seq_len
    phi_pad = torch.nn.functional.pad(phi_q, (0, 0, 0, 0, 0, pad))
    phi_pad = phi_pad.reshape(batch_heads, num_tiles, build_tile, num_tables, corners)
    scaled = torch.nn.functional.pad(grad * inv_den[..., None], (0, 0, 0, pad))
    scaled = scaled.reshape(batch_heads, num_tiles, build_tile, head_dim)
    weight = torch.nn.functional.pad(grad_den, (0, pad)).reshape(batch_heads, num_tiles, build_tile)
    if dropped_token is not None:
        tile, slot = divmod(dropped_token, build_tile)
        phi_pad[:, tile, slot] = 0.0
    _, partial_b = build_tile_partials(phi_pad, scaled)
    partial_a, _ = build_tile_partials(phi_pad * weight[..., None, None], scaled[..., :1])
    slice_floats = corners * (head_dim + 1)
    workspace = torch.empty(num_tiles, batch_heads, num_tables, slice_floats)
    workspace[..., : corners * head_dim] = partial_b.permute(1, 0, 2, 3, 4).reshape(
        num_tiles, batch_heads, num_tables, -1
    )
    workspace[..., corners * head_dim :] = partial_a.permute(1, 0, 2, 3)
    tiles = list(workspace.unbind(0))
    tree_reduce(tiles, num_tiles, lambda a, b: b if a is None else (a if b is None else a + b))
    grad_buckets = tiles[0][..., : corners * head_dim].reshape(batch_heads, num_tables, corners, head_dim)
    grad_mass = tiles[0][..., corners * head_dim :]

    # Key side.
    values = rows(v)
    phi_k, u_k, plus_k, minus_k = soft_hash_fp32(rows(k), W, beta)
    dv = torch.einsum("bnlr,blrd->bnd", phi_k, grad_buckets)
    grad_phi_k = torch.einsum("blrd,bnd->bnlr", grad_buckets, values) + grad_mass[:, None]
    dk, beta_terms_k = hash_grad_fp32(phi_k * grad_phi_k, u_k, plus_k, minus_k, W, beta)

    partials = torch.cat([reduce_beta_partials(beta_terms_q), reduce_beta_partials(beta_terms_k)])
    grad_beta = one_cta_reduce(partials)
    shape = (batch, heads, seq_len, head_dim)
    return dq.reshape(shape), dk.reshape(shape), dv.reshape(shape), grad_beta


# (head_dim, num_planes, num_tables, seq_len, build_tile, beta, seed)
EMULATION_CASES = [
    (64, 1, 1, 127, 16, 1.0, 21),
    (64, 2, 2, 129, 16, 4.0, 21),
    (128, 4, 4, 1000, 16, 1.0, 21),
    (128, 5, 4, 1000, 8, 4.0, 21),
    (64, 5, 3, 2049, 2048, 1.0 / 8.0, 21),
    (128, 5, 4, 300, 16, 1.0, 21),
    (128, 4, 4, 2049, 2048, 4.0, 22),
    (128, 4, 4, 2049, 2048, 1.0 / 128**0.5, 23),
    # Tiny N: every row of a head can be saturated, and a query's buckets can
    # all have means close to O, the two cancellations the scales account for.
    (64, 1, 1, 2, 16, 4.0, 21),
    (64, 1, 1, 2, 16, 4.0, 22),
    (64, 1, 4, 2, 16, 4.0, 21),
    (64, 1, 4, 3, 16, 1.0, 22),
    (64, 3, 2, 2, 16, 1.0, 23),
    (128, 5, 4, 2, 16, 4.0, 21),
    (128, 5, 1, 16, 16, 4.0, 22),
    (128, 4, 4, 17, 16, 1.0, 21),
    (64, 2, 2, 17, 16, 1.0 / 8.0, 23),
]


@pytest.mark.parametrize(
    "head_dim, num_planes, num_tables, seq_len, build_tile, beta, seed", EMULATION_CASES
)
def test_emulated_backward_matches_reference(head_dim, num_planes, num_tables, seq_len, build_tile,
                                             beta, seed):
    q, k, v, W = make_bf16_inputs(seed, 1, 2, seq_len, head_dim, num_tables, num_planes)
    grad_o = make_bf16_grad_output(seed + 100, q)
    ref, scales = reference_backward(grad_o, q, k, v, W, beta)
    got = emulate_backward(grad_o, q, k, v, W, beta, build_tile)
    for name, out32, expected, scale in zip(("dq", "dk", "dv"), got[:3], ref[:3], scales[:3], strict=True):
        # The fp32 part of the error must use at most 10% of the atol budget.
        atol = GRAD_ATOL_FACTOR * max(1.0, beta) * scale
        worst = ((out32.double() - expected).abs() / atol).max().item()
        assert worst <= 0.1, f"{name}: fp32 error uses {worst:.3f} of atol"
        assert_grad_close(out32.to(torch.bfloat16), expected, scale, beta, name)
    beta_bound = BETA_GRAD_TOL_FACTOR * max(1.0, beta) * scales[3]
    beta_share = abs(got[3].item() - ref[3].item()) / beta_bound
    assert beta_share <= 0.1, f"d beta fp32 error uses {beta_share:.3f} of its tolerance"
    assert_beta_grad_close(got[3], ref[3], scales[3], beta)


def test_tolerance_sees_one_dropped_token():
    # A single query token missing from dA/dB (for example a tail-masking
    # bug) must fail the dk or dv check, not hide inside the tolerance.
    q, k, v, W = make_bf16_inputs(5, 1, 2, 2049, 64, 2, 3)
    grad_o = make_bf16_grad_output(6, q)
    beta = 1.0
    ref, scales = reference_backward(grad_o, q, k, v, W, beta)
    got = emulate_backward(grad_o, q, k, v, W, beta, 2048, dropped_token=2048)
    failures = 0
    for name, out32, expected, scale in zip(("dk", "dv"), got[1:3], ref[1:3], scales[1:3], strict=True):
        try:
            assert_grad_close(out32.to(torch.bfloat16), expected, scale, beta, name)
        except AssertionError:
            failures += 1
    assert failures >= 1
