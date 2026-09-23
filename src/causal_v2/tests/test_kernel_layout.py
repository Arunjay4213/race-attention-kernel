"""CPU checks of the v2 kernels' index math (kernels/race_causal_fwd.cu for
v2a, kernels/race_causal_fwd_tc.cu for v2b).

No GPU is needed. Each test mirrors one mapping of the kernels and checks it
covers its index space exactly once, or that a layout is aligned, disjoint and
within the shared-memory limits.

v2a:
  - the sub-chunk choice C(D, P) and the output-pass shared-memory layout,
  - step 1 loads: every (tensor, row, 16-byte vector) once,
  - step 2 features: every (side, table, token) once, one (side, table) per warp,
  - step 3 G and step 5 Num micro-tiles: every element once, rows owned by
    half-warps (the Den shuffle reduction relies on it),
  - step 6 update: every state element B[s][c] once, A[s] once,
  - the default tile length rule gives 2048 (config A) and 4096 (config B).
v2b:
  - the launch shapes (warps, C) of both kernels and the resulting residency,
  - the shared-memory layout: 32-byte aligned, disjoint regions, wmma strides,
    the U tiles inside the scratch region, the footprints,
  - row staging, the Phi pairs, the G tiles, the output tiles (with the
    balanced causal work of the mirrored rounds), the state tiles and the A
    update groups: every index once, with the sharing the kernels rely on.

The constants below mirror the kernels and must be kept in sync.
"""
from __future__ import annotations

import itertools

import pytest

THREADS = 256
WARP = 32
GRID_SIDE = 16
MAX_TABLES = 4
PORTABLE_SMEM_BYTES = 99 * 1024
MAX_AUTO_TILE_TOKENS = 4096
OPTIN_LIMITS = {"sm_80": 163 * 1024, "sm_89": 99 * 1024, "sm_90": 227 * 1024}

HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 3, 4, 5]
TABLE_COUNTS = [1, 2, 3, 4]


def align16(n: int) -> int:
    return (n + 15) & ~15


def smem_layout(head_dim: int, num_planes: int, chunk: int, num_tables: int) -> dict[str, tuple[int, int]]:
    """(byte offset, byte length) of each region, mirroring output_smem_layout."""
    features = num_tables << num_planes
    x_bytes = align16(chunk * (head_dim + 2) * 2)
    phi_bytes = align16(chunk * (features + 1) * 4)
    planes = 3 * x_bytes
    phi_q = planes + align16(num_tables * num_planes * head_dim * 4)
    phi_k = phi_q + phi_bytes
    state = phi_k + phi_bytes
    inv_den = state + align16(features * (head_dim + 1) * 4)
    return {
        "x_q": (0, chunk * (head_dim + 2) * 2),
        "x_k": (x_bytes, chunk * (head_dim + 2) * 2),
        "x_v": (2 * x_bytes, chunk * (head_dim + 2) * 2),
        "planes": (planes, num_tables * num_planes * head_dim * 4),
        "phi_q": (phi_q, chunk * (features + 1) * 4),
        "phi_k": (phi_k, chunk * (features + 1) * 4),
        "state": (state, features * (head_dim + 1) * 4),
        "inv_den": (inv_den, chunk * 4),
    }


def smem_total(head_dim: int, num_planes: int, chunk: int, num_tables: int) -> int:
    offset, length = smem_layout(head_dim, num_planes, chunk, num_tables)["inv_den"]
    return offset + align16(length)


def sub_chunk(head_dim: int, num_planes: int) -> int:
    """choose_sub_chunk."""
    fits = smem_total(head_dim, num_planes, 64, MAX_TABLES) <= PORTABLE_SMEM_BYTES
    return 64 if 64 * head_dim <= 4096 and fits else 32


def all_shapes():
    return itertools.product(HEAD_DIMS, PLANE_COUNTS, TABLE_COUNTS)


def test_sub_chunk_choice_matches_the_plan():
    assert sub_chunk(64, 4) == 64   # config A
    assert sub_chunk(128, 5) == 32  # config B
    assert all(sub_chunk(64, p) == 64 for p in range(1, 5))
    assert sub_chunk(64, 5) == 32
    assert all(sub_chunk(128, p) == 32 for p in range(1, 6))
    assert smem_total(64, 4, 64, 4) == 79_616       # 77.75 KiB, plan section 6.2
    assert smem_total(128, 5, 32, 4) == 134_400     # 131.25 KiB, A100 and H100 only


@pytest.mark.parametrize("head_dim, num_planes, num_tables", list(all_shapes()))
def test_smem_regions_are_aligned_disjoint_and_hold_the_gram_alias(head_dim, num_planes, num_tables):
    chunk = sub_chunk(head_dim, num_planes)
    regions = sorted(smem_layout(head_dim, num_planes, chunk, num_tables).values())
    for (offset, length), (next_offset, _) in zip(regions[:-1], regions[1:], strict=True):
        assert offset % 16 == 0
        assert offset + length <= next_offset
    # G [C][C + 1] fp32 overwrites x_q and x_k only.
    x_v_offset = smem_layout(head_dim, num_planes, chunk, num_tables)["x_v"][0]
    assert chunk * (chunk + 1) * 4 <= x_v_offset


@pytest.mark.parametrize("head_dim, num_planes, num_tables", list(all_shapes()))
def test_smem_fits_where_the_plan_says(head_dim, num_planes, num_tables):
    total = smem_total(head_dim, num_planes, sub_chunk(head_dim, num_planes), num_tables)
    assert total <= OPTIN_LIMITS["sm_80"]  # every shape runs on A100 and H100
    # Only config B's d = 128, P = 5 with L >= 3 is too large for sm_89 (L40S).
    too_big_for_l40s = head_dim == 128 and num_planes == 5 and num_tables >= 3
    assert (total > OPTIN_LIMITS["sm_89"]) == too_big_for_l40s


def test_strides_are_odd_words_for_conflict_free_column_walks():
    for head_dim in HEAD_DIMS:
        assert ((head_dim + 2) // 2) % 2 == 1          # staged bf16 rows, in 32-bit words
        assert (head_dim + 1) % 2 == 1                  # state rows
    for num_tables, num_planes in itertools.product(TABLE_COUNTS, PLANE_COUNTS):
        assert ((num_tables << num_planes) + 1) % 2 == 1  # phi rows
    for chunk in (32, 64):
        assert (chunk + 1) % 2 == 1                     # G rows


@pytest.mark.parametrize("head_dim, chunk", [(64, 64), (64, 32), (128, 32)])
def test_loads_cover_each_vector_once(head_dim, chunk):
    vectors_per_row = head_dim // 8
    iters = chunk * vectors_per_row // THREADS
    assert chunk * vectors_per_row % THREADS == 0
    seen = []
    for tensor in range(3):
        for it in range(iters):
            for tid in range(THREADS):
                vector = it * THREADS + tid
                seen.append((tensor, vector // vectors_per_row, (vector % vectors_per_row) * 8))
    expected = {(t, r, c) for t in range(3) for r in range(chunk) for c in range(0, head_dim, 8)}
    assert len(seen) == len(expected) and set(seen) == expected


@pytest.mark.parametrize("chunk", [32, 64])
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
def test_feature_pairs_cover_each_side_table_token_once(chunk, num_tables):
    pairs_per_side = chunk * num_tables
    owners = {}
    for tid in range(THREADS):
        for pair in range(tid, 2 * pairs_per_side, THREADS):
            key_side = pair >= pairs_per_side
            within = pair - pairs_per_side if key_side else pair
            table, token = within // chunk, within % chunk
            key = (key_side, table, token)
            assert key not in owners
            owners[key] = (pair // WARP, pair % WARP)
    assert len(owners) == 2 * pairs_per_side
    # A warp (32 consecutive pairs) stays within one (side, table): plane reads are broadcasts.
    by_warp = {}
    for (key_side, table, _), (warp_slot, _) in owners.items():
        by_warp.setdefault(warp_slot, set()).add((key_side, table))
    assert all(len(v) == 1 for v in by_warp.values())


@pytest.mark.parametrize("rows, cols", [(64, 64), (32, 32), (32, 64), (32, 128)])
def test_micro_tiles_cover_each_element_once_with_half_warp_rows(rows, cols):
    row_tiles, col_tiles = rows // GRID_SIDE, cols // GRID_SIDE
    owners = {}
    for tid in range(THREADS):
        ty, tx = tid // GRID_SIDE, tid % GRID_SIDE
        for a in range(row_tiles):
            for b in range(col_tiles):
                element = (ty + GRID_SIDE * a, tx + GRID_SIDE * b)
                assert element not in owners
                owners[element] = tid
    assert len(owners) == rows * cols
    # The 16 owners of a row are the lanes of one half-warp, so a xor shuffle
    # with offsets 8, 4, 2, 1 reduces exactly over them.
    for i in range(rows):
        row_owners = {owners[(i, j)] for j in range(cols)}
        assert len(row_owners) == GRID_SIDE
        warps = {tid // WARP for tid in row_owners}
        halves = {(tid % WARP) // GRID_SIDE for tid in row_owners}
        assert len(warps) == 1 and len(halves) == 1


@pytest.mark.parametrize("head_dim, num_planes, num_tables", list(all_shapes()))
def test_state_update_covers_each_state_element_once(head_dim, num_planes, num_tables):
    features = num_tables << num_planes
    col_tiles = head_dim // GRID_SIDE
    update_row_tiles = GRID_SIDE // col_tiles
    assert update_row_tiles * col_tiles == 16  # 16 accumulators per thread
    rows_per_pass = GRID_SIDE * update_row_tiles
    written = []
    for first_row in range(0, features, rows_per_pass):
        for tid in range(THREADS):
            ty, tx = tid // GRID_SIDE, tid % GRID_SIDE
            for a in range(update_row_tiles):
                s = first_row + ty + GRID_SIDE * a
                if s < features:
                    written += [(s, tx + GRID_SIDE * b) for b in range(col_tiles)]
    written += [(tid, head_dim) for tid in range(THREADS) if tid < features]  # A column
    expected = {(s, c) for s in range(features) for c in range(head_dim + 1)}
    assert len(written) == len(expected) and set(written) == expected


def default_tile_tokens(head_dim: int, num_planes: int, num_tables: int) -> int:
    """default_tile_tokens."""
    features = num_tables << num_planes
    tile = sub_chunk(head_dim, num_planes)
    while tile < MAX_AUTO_TILE_TOKENS and tile * 3 * head_dim < 5 * 16 * features * (head_dim + 1):
        tile *= 2
    return tile


def test_default_tile_lengths_match_the_plan():
    assert default_tile_tokens(64, 4, 4) == 2048    # config A
    assert default_tile_tokens(128, 5, 4) == 4096   # config B
    for head_dim, num_planes, num_tables in all_shapes():
        tile = default_tile_tokens(head_dim, num_planes, num_tables)
        assert tile % sub_chunk(head_dim, num_planes) == 0
        state_bytes_per_token = 16 * (num_tables << num_planes) * (head_dim + 1) / tile
        assert tile == MAX_AUTO_TILE_TOKENS or state_bytes_per_token <= 0.05 * 12 * head_dim


# ---------------------------------------------------------------------------
# v2b (race_causal_fwd_tc.cu)
# ---------------------------------------------------------------------------

FRAG = 16
SCRATCH_STRIDE = FRAG
GRAM_TILE_STRIDE = FRAG + 8
PORTABLE_SMEM_PER_SM = 100 * 1024
RESERVED_SMEM_PER_CTA = 1024
THREADS_PER_SM = 512


def round_up16(n: int) -> int:
    return (n + 15) & ~15


def align32(n: int) -> int:
    return (n + 31) & ~31


def tc_smem_layout(head_dim: int, num_planes: int, num_tables: int, chunk: int, warps: int,
                   emit_output: bool, precise_carry: bool = True) -> dict[str, tuple[int, int]]:
    """(byte offset, byte length) of each non-empty region, mirroring tc_smem_layout."""
    features_padded = round_up16(num_tables << num_planes)
    plane_rows_padded = round_up16(num_tables * num_planes)
    x_stride = max(head_dim, features_padded) + 8
    value_stride = head_dim + 8
    gram_tiles = (chunk // FRAG) * (chunk // FRAG + 1) // 2
    gram_bytes = gram_tiles * FRAG * GRAM_TILE_STRIDE * 2 if emit_output else 0
    sizes = [
        ("x_q", chunk * x_stride * 2 if emit_output else 0),
        ("x_k", chunk * x_stride * 2),
        ("x_v", chunk * value_stride * 2),
        ("planes_hi", plane_rows_padded * value_stride * 2),
        ("planes_lo", plane_rows_padded * value_stride * 2),
        ("gram_hi", gram_bytes),
        ("gram_lo", gram_bytes),
        ("carry_hi", head_dim * (features_padded + 8) * 2 if emit_output else 0),
        ("carry_lo", head_dim * (features_padded + 8) * 2 if emit_output and precise_carry else 0),
        ("mass", features_padded * 4),
        ("scratch", warps * FRAG * SCRATCH_STRIDE * 4),
    ]
    layout, cursor = {}, 0
    for name, size in sizes:
        if size:
            layout[name] = (cursor, size)
        cursor += align32(size)
    layout["total"] = (cursor, 0)
    return layout


def tc_smem_total(*args, **kwargs) -> int:
    return tc_smem_layout(*args, **kwargs)["total"][0]


def output_pass_launch(head_dim: int, num_planes: int) -> tuple[int, int]:
    """(warps, C), mirroring output_pass_launch."""
    two_ctas = PORTABLE_SMEM_PER_SM // 2 - RESERVED_SMEM_PER_CTA
    for chunk in (64, 32):
        if tc_smem_total(head_dim, num_planes, MAX_TABLES, chunk, 8, True) <= two_ctas:
            return 8, chunk
    fits_64 = tc_smem_total(head_dim, num_planes, MAX_TABLES, 64, 16, True) <= PORTABLE_SMEM_BYTES
    return 16, 64 if fits_64 else 32


def tile_sums_launch(head_dim: int, num_planes: int) -> tuple[int, int]:
    """(warps, C), mirroring tile_sums_launch."""
    state_tiles = round_up16(MAX_TABLES << num_planes) // FRAG * (head_dim // FRAG)
    return (8 if state_tiles <= 4 * 8 else 16), output_pass_launch(head_dim, num_planes)[1]


def all_tc_instantiations():
    """(D, P, L, C, warps, emit_output) of every compiled v2b kernel."""
    for head_dim, num_planes, num_tables in all_shapes():
        warps, chunk = output_pass_launch(head_dim, num_planes)
        yield head_dim, num_planes, num_tables, chunk, warps, True
        warps, chunk = tile_sums_launch(head_dim, num_planes)
        yield head_dim, num_planes, num_tables, chunk, warps, False


def test_tc_launch_shapes():
    for head_dim, num_planes in itertools.product(HEAD_DIMS, PLANE_COUNTS):
        warps, chunk = output_pass_launch(head_dim, num_planes)
        expected = (8, 32) if head_dim == 64 and num_planes <= 4 else (16, 32)
        assert (warps, chunk) == expected
        assert tile_sums_launch(head_dim, num_planes) == ((16, 32) if (head_dim, num_planes) == (128, 5) else (8, 32))
    # Config A: two output-pass CTAs share a 100 KiB SM; config B needs A100 or H100.
    assert tc_smem_total(64, 4, 4, 32, 8, True) == 49_920
    assert 2 * (49_920 + RESERVED_SMEM_PER_CTA) <= PORTABLE_SMEM_PER_SM
    assert tc_smem_total(128, 4, 4, 32, 16, True) == 92_928
    assert OPTIN_LIMITS["sm_89"] < tc_smem_total(128, 5, 4, 32, 16, True) <= OPTIN_LIMITS["sm_80"]


@pytest.mark.parametrize("head_dim, num_planes, num_tables, chunk, warps, emit",
                         list(all_tc_instantiations()))
def test_tc_smem_layout_is_aligned_disjoint_and_fits(head_dim, num_planes, num_tables, chunk, warps, emit):
    layout = tc_smem_layout(head_dim, num_planes, num_tables, chunk, warps, emit)
    total = layout.pop("total")[0]
    regions = sorted(layout.values())
    for (offset, length), (next_offset, _) in zip(regions[:-1], regions[1:], strict=True):
        assert offset % 32 == 0 and offset + length <= next_offset  # wmma fragment pointers
    assert total <= OPTIN_LIMITS["sm_80"]
    # Only d = 128, P = 5 with L >= 3 is too large for sm_86 / sm_89.
    too_big = emit and head_dim == 128 and num_planes == 5 and num_tables >= 3
    assert (total > OPTIN_LIMITS["sm_89"]) == too_big
    # The U tiles, fp32 [sides][C][LP16 + 4], fit in the scratch region.
    u_bytes = (2 if emit else 1) * chunk * (round_up16(num_tables * num_planes) + 4) * 4
    assert u_bytes <= layout["scratch"][1]


@pytest.mark.parametrize("head_dim, num_planes, num_tables", list(all_shapes()))
def test_tc_strides_are_wmma_legal_and_spread_over_banks(head_dim, num_planes, num_tables):
    features_padded = round_up16(num_tables << num_planes)
    chunk = output_pass_launch(head_dim, num_planes)[1]
    for stride in (max(head_dim, features_padded) + 8, head_dim + 8, features_padded + 8, GRAM_TILE_STRIDE):
        assert stride % 8 == 0  # ldm: a multiple of 16 bytes
        words = stride // 2
        assert words % 4 == 0 and (words // 4) % 2 == 1  # 8 rows hit 8 distinct 4-bank groups
        assert len({(row * words) % 32 // 4 for row in range(8)}) == 8
    assert SCRATCH_STRIDE % 4 == 0  # fp32 ldm: a multiple of 16 bytes


@pytest.mark.parametrize("head_dim, num_planes, num_tables, chunk, warps, emit",
                         list(all_tc_instantiations()))
def test_tc_row_staging_covers_each_vector_once(head_dim, num_planes, num_tables, chunk, warps, emit):
    threads = warps * WARP
    vectors_per_row = head_dim // 8
    vectors = chunk * vectors_per_row
    iters = -(-vectors // threads)
    seen = [(it * threads + tid) for it in range(iters) for tid in range(threads) if it * threads + tid < vectors]
    assert sorted(seen) == list(range(vectors))
    # A warp's vectors are consecutive: whole 128-byte row segments.
    assert all(v // vectors_per_row == (v - v % vectors_per_row) // vectors_per_row for v in seen)


@pytest.mark.parametrize("head_dim, num_planes, num_tables, chunk, warps, emit",
                         list(all_tc_instantiations()))
def test_tc_phi_pairs_cover_each_side_token_table_once(head_dim, num_planes, num_tables, chunk, warps, emit):
    threads = warps * WARP
    sides = 2 if emit else 1
    per_side = chunk * num_tables
    owners, by_warp = {}, {}
    for tid in range(threads):
        for pair in range(tid, sides * per_side, threads):
            side, table, token = pair // per_side, (pair - (pair // per_side) * per_side) // chunk, pair % chunk
            assert (side, table, token) not in owners
            owners[(side, table, token)] = tid
            by_warp.setdefault(pair // WARP, set()).add((side, table))
    assert len(owners) == sides * per_side
    assert all(len(pairs) == 1 for pairs in by_warp.values())  # one (side, table) per warp


def gram_tile_coords(t: int) -> tuple[int, int]:
    ti = 0
    while (ti + 1) * (ti + 2) // 2 <= t:
        ti += 1
    return ti, t - ti * (ti + 1) // 2


@pytest.mark.parametrize("chunk", [32, 64])
def test_tc_gram_tiles_cover_the_lower_triangle_once(chunk):
    row_tiles = chunk // FRAG
    tiles = [gram_tile_coords(t) for t in range(row_tiles * (row_tiles + 1) // 2)]
    assert sorted(tiles) == [(ti, tj) for ti in range(row_tiles) for tj in range(ti + 1)]


def output_row_tile(warp: int, round_index: int, warps: int, col_tiles: int) -> int:
    rows_per_round = warps // col_tiles
    ti = (warp + round_index * warps) // col_tiles
    first = round_index * rows_per_round
    return 2 * first + rows_per_round - 1 - ti if round_index % 2 == 1 else ti


@pytest.mark.parametrize("head_dim, chunk, warps", [(64, 32, 8), (64, 32, 16), (128, 32, 16), (64, 64, 8),
                                                    (128, 64, 16), (64, 64, 16)])
def test_tc_output_tiles_cover_each_tile_once_with_balanced_rounds(head_dim, chunk, warps):
    row_tiles, col_tiles = chunk // FRAG, head_dim // FRAG
    tiles = row_tiles * col_tiles
    rounds = -(-tiles // warps)
    assert rounds <= 2 and warps % col_tiles == 0 and (rounds == 1 or tiles % warps == 0)
    owners = {}
    intra_steps = {}
    for warp in range(min(warps, tiles)):
        for r in range(rounds):
            tile = (output_row_tile(warp, r, warps, col_tiles), warp % col_tiles)
            assert tile not in owners
            owners[tile] = warp
            intra_steps[warp] = intra_steps.get(warp, 0) + tile[0] + 1
    assert set(owners) == {(ti, tc) for ti in range(row_tiles) for tc in range(col_tiles)}
    if rounds == 2:  # every warp gets the same causal intra work
        assert len(set(intra_steps.values())) == 1


@pytest.mark.parametrize("head_dim, num_planes, num_tables, chunk, warps, emit",
                         list(all_tc_instantiations()))
def test_tc_state_tiles_and_mass_groups(head_dim, num_planes, num_tables, chunk, warps, emit):
    features_padded = round_up16(num_tables << num_planes)
    feature_tiles, col_tiles = features_padded // FRAG, head_dim // FRAG
    state_tiles = feature_tiles * col_tiles
    per_warp = -(-state_tiles // warps)
    owners = {}
    for warp in range(warps):
        for f in range(per_warp):
            idx = warp + f * warps
            if idx < state_tiles:
                fs, tc = idx // col_tiles, idx % col_tiles
                assert tc == warp % col_tiles  # a warp's tiles share one V fragment per step
                assert (fs, tc) not in owners
                owners[(fs, tc)] = warp
    assert len(owners) == state_tiles
    # 128 registers per thread: at most 4 resident state tiles (32 registers).
    assert per_warp <= 4
    # A update: groups of G consecutive lanes inside one warp, every feature once.
    threads = warps * WARP
    group = min(1 << ((threads // features_padded).bit_length() - 1), WARP)
    assert chunk % group == 0 and WARP % group == 0
    features = {tid // group for tid in range(threads) if tid // group < features_padded}
    assert features == set(range(features_padded))
    assert THREADS_PER_SM % threads == 0
