"""CPU emulation of the CUDA kernels' index math and fp32 data flow.

The GPU is not available in every environment, so this file checks the parts
of kernels/race_fwd.cu that can be checked without one:
  - the thread ownership of B in the bucket build covers every (r, c) once,
  - the stage slots cover every token of a tile once,
  - the fan-in-8 reduce schedule adds every tile exactly once, for any count,
  - the query kernel's shared-memory regions are aligned, disjoint and fit
    the smallest opt-in limit of the supported GPUs,
  - the full pipeline in fp32 through the real workspace layout matches the
    fp64 reference well inside the tolerance the CUDA tests use.

The constants and formulas below mirror race_fwd.cu and race_fwd.h and must be
kept in sync with them.
"""
from __future__ import annotations

import pytest
import torch

from numerics import (
    BF16_UNIT_ROUNDOFF,
    assert_buckets_close,
    assert_output_close,
    make_bf16_inputs,
    output_atol,
)
from reference import bucket_sums, race_forward_reference

THREADS = 256
WARPS = THREADS // 32
STAGE_TOKENS = 16
BUILD_TILE_TOKENS = 2048
REDUCE_FAN_IN = 8
SM89_MAX_DYNAMIC_SMEM_BYTES = 99 * 1024  # smallest opt-in limit among sm_80/89/90

HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 3, 4, 5]


def build_ownership(head_dim: int, num_planes: int) -> list[tuple[int, int]]:
    """(corner, column) pairs written by each thread, mirroring BuildOwnership."""
    corners = 1 << num_planes
    row_groups = THREADS // head_dim
    active_row_groups = min(corners, row_groups)
    rows_per_thread = corners // active_row_groups
    owned = []
    for tid in range(THREADS):
        column, row_group = tid % head_dim, tid // head_dim
        if row_group < active_row_groups:
            first_row = row_group * rows_per_thread
            owned += [(first_row + i, column) for i in range(rows_per_thread)]
    return owned


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_build_ownership_covers_each_bucket_entry_once(head_dim, num_planes):
    owned = build_ownership(head_dim, num_planes)
    expected = {(r, c) for r in range(1 << num_planes) for c in range(head_dim)}
    assert len(owned) == len(expected)
    assert set(owned) == expected


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
def test_build_vector_reads_are_aligned(head_dim, num_planes):
    # probs_s[slot][first_row] is read with float4 when rows_per_thread % 4 == 0
    # and float2 when it is 2, so first_row must be a multiple of the width and
    # the row stride (R floats) must preserve it.
    corners = 1 << num_planes
    rows_per_thread = corners // min(corners, THREADS // head_dim)
    width = 4 if rows_per_thread % 4 == 0 else (2 if rows_per_thread == 2 else 1)
    assert corners % width == 0
    for row_group in range(min(corners, THREADS // head_dim)):
        assert (row_group * rows_per_thread) % width == 0


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 2047, 2048, 2049, 5000])
def test_stage_slots_cover_each_token_once(seq_len):
    tiles = (seq_len + BUILD_TILE_TOKENS - 1) // BUILD_TILE_TOKENS
    seen = []
    for tile in range(tiles):
        tile_begin = tile * BUILD_TILE_TOKENS
        tile_end = min(seq_len, tile_begin + BUILD_TILE_TOKENS)
        num_stages = (tile_end - tile_begin + STAGE_TOKENS - 1) // STAGE_TOKENS
        for stage in range(num_stages):
            stage_begin = tile_begin + stage * STAGE_TOKENS
            for warp in range(WARPS):
                for i in range(STAGE_TOKENS // WARPS):
                    token = stage_begin + warp + i * WARPS
                    if token < tile_end:
                        seen.append(token)
    assert sorted(seen) == list(range(seq_len))


def tree_reduce(workspace: list, num_tiles: int, add) -> None:
    """In-place fan-in-8 pairwise tree, mirroring race_bucket_reduce."""
    stride = 1
    while stride < num_tiles:
        span = stride * REDUCE_FAN_IN
        num_groups = (num_tiles + span - 1) // span
        for group in range(num_groups):
            first_tile = group * span
            terms = [
                workspace[t] if (t := first_tile + i * stride) < num_tiles else None
                for i in range(REDUCE_FAN_IN)
            ]
            width = 1
            while width < REDUCE_FAN_IN:
                for i in range(0, REDUCE_FAN_IN, 2 * width):
                    terms[i] = add(terms[i], terms[i + width])
                width *= 2
            workspace[first_tile] = terms[0]
        stride *= REDUCE_FAN_IN


def test_tree_reduce_adds_every_tile_exactly_once():
    def add_multisets(a, b):
        if a is None or b is None:
            return a if b is None else b
        return a + b

    for num_tiles in list(range(1, 130)) + [511, 512, 513, 4096, 4883]:
        workspace = [[t] for t in range(num_tiles)]
        tree_reduce(workspace, num_tiles, add_multisets)
        assert sorted(workspace[0]) == list(range(num_tiles)), num_tiles


def query_smem_regions(head_dim: int, num_planes: int, num_tables: int) -> list[tuple[int, int]]:
    """(offset, length) in floats of each region, mirroring QuerySmem."""
    corners = 1 << num_planes

    def round_up4(n: int) -> int:
        return (n + 3) & ~3

    buckets = (0, num_tables * corners * head_dim)
    mass = (buckets[1], num_tables * corners)
    planes = (mass[0] + round_up4(mass[1]), num_tables * num_planes * head_dim)
    probs = (planes[0] + planes[1], WARPS * round_up4(num_tables * corners))
    return [buckets, mass, planes, probs]


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("num_tables", [1, 2, 3, 4])
def test_query_smem_layout(head_dim, num_planes, num_tables):
    regions = query_smem_regions(head_dim, num_planes, num_tables)
    for (offset, length), (next_offset, _) in zip(regions[:-1], regions[1:], strict=True):
        assert offset + length <= next_offset
    for offset, _ in regions:
        assert offset % 4 == 0, "float4 accesses need 16-byte aligned regions"
    total_bytes = 4 * (regions[-1][0] + regions[-1][1])
    assert total_bytes <= SM89_MAX_DYNAMIC_SMEM_BYTES


def butterfly_sum(x: torch.Tensor) -> torch.Tensor:
    """warp_allreduce_sum over the last axis (32 lanes), same pairing order; returns lane 0."""
    lanes = torch.arange(32)
    for offset in (16, 8, 4, 2, 1):
        x = x + x[..., lanes ^ offset]
    return x[..., 0]


def sigmoid_pair_fp32(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e = torch.exp(-z.abs())
    large = 1.0 / (1.0 + e)
    small = e * large
    return torch.where(z >= 0, large, small), torch.where(z >= 0, small, large)


def corner_probs_fp32(x: torch.Tensor, W: torch.Tensor, beta: float) -> torch.Tensor:
    """fp32 corner probabilities in the kernels' order. x [BH, N, D] -> [BH, N, L, R].

    Each lane sums its D/32 products serially, then the warp butterfly adds
    the 32 lane partials; the Bernoulli product runs over planes in order.
    """
    head_dim = x.shape[-1]
    num_planes = W.shape[1]
    vec = head_dim // 32
    products = x[:, :, None, None, :] * W  # [BH, N, L, P, D]
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
    return phi


def build_tile_partials(phi_k: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile A and B with the bucket build's summation order.

    Args:
        phi_k: [BH, T, tile, L, R] (zero past N). values: [BH, T, tile, D] (zero past N).

    Returns:
        A [BH, T, L, R] and B [BH, T, L, R, D]. B is a serial sum over the tile's
        tokens (one thread per (r, c)); A is a serial sum per warp over the
        warp's stage slots, then a sum over warps in warp order.
    """
    build_tile = phi_k.shape[2]
    weighted = torch.zeros(*phi_k.shape[:2], *phi_k.shape[3:], values.shape[-1])
    for token in range(build_tile):
        weighted = weighted + phi_k[:, :, token, :, :, None] * values[:, :, token, None, None, :]
    warp_masses = []
    for warp in range(WARPS):
        warp_mass = torch.zeros_like(phi_k[:, :, 0])
        for stage_begin in range(0, build_tile, STAGE_TOKENS):
            for i in range(STAGE_TOKENS // WARPS):
                token = stage_begin + warp + i * WARPS
                if token < build_tile:
                    warp_mass = warp_mass + phi_k[:, :, token]
        warp_masses.append(warp_mass)
    mass = warp_masses[0]
    for warp_mass in warp_masses[1:]:
        mass = mass + warp_mass
    return mass, weighted


def emulate_forward(q, k, v, W, beta: float, build_tile: int):
    """fp32 forward through the kernels' workspace layout and summation orders.

    Returns:
        out: [B, H, N, D] bf16 output.
        out32: [B, H, N, D] fp32 output before the bf16 rounding.
        mass: [B, H, L, R] reduced A. weighted: [B, H, L, R, D] reduced B.
    """
    batch, heads, seq_len, head_dim = q.shape
    num_tables, num_planes, _ = W.shape
    corners = 1 << num_planes
    batch_heads = batch * heads
    slice_floats = corners * (head_dim + 1)
    num_tiles = (seq_len + build_tile - 1) // build_tile
    padded = num_tiles * build_tile

    keys = k.float().reshape(batch_heads, seq_len, head_dim)
    values = v.float().reshape(batch_heads, seq_len, head_dim)
    phi_k = corner_probs_fp32(keys, W, beta)
    phi_k = torch.nn.functional.pad(phi_k, (0, 0, 0, 0, 0, padded - seq_len))
    values = torch.nn.functional.pad(values, (0, 0, 0, padded - seq_len))
    phi_k = phi_k.reshape(batch_heads, num_tiles, build_tile, num_tables, corners)
    values = values.reshape(batch_heads, num_tiles, build_tile, head_dim)
    partial_a, partial_b = build_tile_partials(phi_k, values)

    # Workspace [T, BH, L, R * (D + 1)]: B row-major first, then A.
    workspace = torch.empty(num_tiles, batch_heads, num_tables, slice_floats)
    workspace[..., : corners * head_dim] = partial_b.permute(1, 0, 2, 3, 4).reshape(
        num_tiles, batch_heads, num_tables, -1
    )
    workspace[..., corners * head_dim :] = partial_a.permute(1, 0, 2, 3)
    tiles = list(workspace.unbind(0))
    tree_reduce(tiles, num_tiles, lambda a, b: b if a is None else (a if b is None else a + b))
    totals = tiles[0]

    buckets = totals[..., : corners * head_dim].reshape(batch_heads, num_tables, corners, head_dim)
    mass = totals[..., corners * head_dim :]
    queries = q.float().reshape(batch_heads, seq_len, head_dim)
    phi_q = corner_probs_fp32(queries, W, beta)  # [BH, N, L, R]

    # Lane r accumulates phi * A over tables; the warp butterfly adds the lanes.
    den_lanes = torch.zeros(batch_heads, seq_len, 32)
    numer = torch.zeros(batch_heads, seq_len, head_dim)
    for table in range(num_tables):
        den_lanes[..., :corners] = den_lanes[..., :corners] + phi_q[:, :, table] * mass[:, None, table]
        for r in range(corners):
            numer = numer + phi_q[:, :, table, r, None] * buckets[:, None, table, r]
    den = butterfly_sum(den_lanes).unsqueeze(-1)
    inv_den = torch.where(den == 0, torch.zeros_like(den), 1.0 / den)
    out32 = (numer * inv_den).reshape(batch, heads, seq_len, head_dim)
    return (
        out32.to(torch.bfloat16),
        out32,
        mass.reshape(batch, heads, num_tables, corners),
        buckets.reshape(batch, heads, num_tables, corners, head_dim),
    )


@pytest.mark.parametrize(
    "head_dim, num_planes, num_tables, seq_len, build_tile, beta",
    [
        (64, 1, 1, 127, 16, 1.0),
        (64, 2, 2, 129, 16, 1.0),
        (128, 4, 4, 1000, 16, 1.0),
        (128, 5, 4, 1000, 8, 4.0),
        (64, 5, 3, 2049, 2048, 1.0 / 8.0),
        (128, 5, 4, 10000, 2048, 1.0),
        (128, 4, 4, 10000, 2048, 4.0),
    ],
)
def test_emulated_pipeline_matches_reference(head_dim, num_planes, num_tables, seq_len, build_tile, beta):
    q, k, v, W = make_bf16_inputs(11, 2, 2, seq_len, head_dim, num_tables, num_planes)
    q64, k64, v64, W64 = q.double(), k.double(), v.double(), W.double()
    beta64 = torch.tensor(beta, dtype=torch.float64)
    ref = race_forward_reference(q64, k64, v64, W64, beta64)
    ref_mass, ref_weighted = bucket_sums(k64, v64, W64, beta64)
    out, out32, mass, weighted = emulate_forward(q, k, v, W, beta, build_tile)

    assert_buckets_close(mass, weighted, ref_mass, ref_weighted, v)
    # The fp32 part of the output error must use at most 10% of the atol budget.
    fp32_error = (out32.double() - ref).abs()
    assert (fp32_error <= 0.1 * output_atol(v, beta)).all()
    assert_output_close(out, ref, v, beta)
    # The rtol term is not loose: bf16 rounding alone uses a large part of it.
    rounding_share = ((out.double() - out32.double()).abs() / (BF16_UNIT_ROUNDOFF * ref.abs())).max()
    assert rounding_share > 0.5
