"""GPU tests of the race_causal_v2 extension (v2a) against the fp64 reference.

Skipped without a CUDA device. The reference runs in fp64 on the GPU on the
same bf16-rounded inputs the kernel sees; the tolerance is derived in
causal_numerics.py. Shapes whose output pass does not fit the device's shared
memory (d = 128, P = 5, L >= 3 on L40S) are skipped, not failed.

The 2^31-element offset test is opt-in (RACE_CAUSAL_HUGE=1): it needs about
25 GB of GPU memory and a few minutes of fp64 reference time.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

from causal_numerics import (
    ATOL_FACTOR,
    assert_states_close,
    assert_within_rounding_floor,
    make_bf16_inputs,
)
from reference import race_causal_chunked_emulation, race_causal_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 4, 5]
TABLE_COUNTS = [1, 2, 4]
SEQ_LENS = [1, 63, 64, 65, 2047, 2048, 2049, 4097, 20000]
BATCH_HEADS = [(1, 1), (2, 4)]  # B * H = 1 and 8


def betas_for(head_dim: int) -> list[float]:
    return [1.0 / math.sqrt(head_dim), 1.0, 2.0]


@pytest.fixture(scope="module")
def race():
    from build import load_extension

    return load_extension(verbose=False)


def case_seed(*params) -> int:
    return hash(params) % (2**31)


def skip_unless_fits(race, head_dim: int, num_planes: int, num_tables: int) -> None:
    if not race.fits_on_device(head_dim, num_planes, num_tables):
        pytest.skip(
            f"output pass needs {race.output_pass_smem_bytes(head_dim, num_planes, num_tables)} bytes "
            "of shared memory, more than this GPU allows"
        )


def reference_output(q, k, v, W, beta: float) -> torch.Tensor:
    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    return race_causal_reference(q.double(), k.double(), v.double(), W.double(), beta64)


@pytest.mark.parametrize("batch, heads", BATCH_HEADS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_forward_matches_reference(race, head_dim, num_planes, num_tables, seq_len, batch, heads):
    skip_unless_fits(race, head_dim, num_planes, num_tables)
    q, k, v, W = make_bf16_inputs(
        case_seed(head_dim, num_planes, num_tables, seq_len, batch, heads),
        batch, heads, seq_len, head_dim, num_tables, num_planes,
    )
    for beta in betas_for(head_dim):
        out = race.forward(q, k, v, W, torch.tensor(beta, device="cuda"))
        assert out.dtype == torch.bfloat16 and out.shape == q.shape
        assert torch.isfinite(out).all()
        assert_within_rounding_floor(out, reference_output(q, k, v, W, beta), v, beta)


@pytest.mark.parametrize("seq_len", [1, 2, 15, 17, 127, 129, 2047, 2049, 3 * 2048 + 17, 10000])
@pytest.mark.parametrize("head_dim, num_planes, num_tables",
                         [(64, 4, 4), (128, 4, 4), (64, 5, 3), (64, 3, 3), (128, 3, 2)])
def test_forced_tile_lengths_agree(race, head_dim, num_planes, num_tables, seq_len):
    # Tiles of C, 2C and 2048 tokens exercise the scan with many, some and one
    # tile; every run must pass the reference test, and runs with different
    # summation orders must agree within one bf16 ulp plus the fp32 allowance
    # (outputs that nearly cancel to 0 have ulps far below fp32 noise; see
    # causal_numerics.py).
    skip_unless_fits(race, head_dim, num_planes, num_tables)
    chunk = race.sub_chunk_tokens(head_dim, num_planes)
    q, k, v, W = make_bf16_inputs(case_seed(seq_len, head_dim, num_planes), 1, 3, seq_len,
                                  head_dim, num_tables, num_planes)
    beta = 0.5
    ref = reference_output(q, k, v, W, beta)
    outputs = []
    for tile_tokens in (chunk, 2 * chunk, 2048):
        out = race.forward(q, k, v, W, torch.tensor(beta), tile_tokens=tile_tokens)
        assert_within_rounding_floor(out, ref, v, beta)
        outputs.append(out.double())
    fp32_allowance = ATOL_FACTOR * max(1.0, beta) * v.double().abs().max().item()
    for other in outputs[1:]:
        ulp = torch.maximum(outputs[0].abs(), other.abs()) * 2.0**-7
        assert ((outputs[0] - other).abs() <= ulp + fp32_allowance).all()


@pytest.mark.parametrize("seq_len", [1, 65, 2049, 20000])
@pytest.mark.parametrize("beta", [0.125, 1.0, 2.0])
@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 4, 4), (128, 5, 2), (64, 1, 1)])
def test_constant_values_come_back_exactly(race, head_dim, num_planes, num_tables, seq_len, beta):
    # The output is a convex combination of the values, so V = c must return c
    # after rounding: this catches any Num / Den inconsistency (a wrong A).
    skip_unless_fits(race, head_dim, num_planes, num_tables)
    q, k, _, W = make_bf16_inputs(case_seed(seq_len, beta, head_dim), 2, 2, seq_len, head_dim,
                                  num_tables, num_planes)
    v = torch.full_like(q, 0.7)
    assert v[0, 0, 0, 0].item() == torch.tensor(0.7).to(torch.bfloat16).item()
    out = race.forward(q, k, v, W, torch.tensor(beta))
    assert torch.equal(out, v)


@pytest.mark.parametrize("head_dim, num_planes", [(64, 4), (128, 2)])
def test_first_output_is_the_first_value(race, head_dim, num_planes):
    q, k, v, W = make_bf16_inputs(11, 2, 3, 1, head_dim, 2, num_planes)
    assert torch.equal(race.forward(q, k, v, W, torch.tensor(1.0)), v)


@pytest.mark.parametrize("position", [0, 63, 64, 2047, 2048, 5000])
def test_future_tokens_do_not_change_the_past(race, position):
    # Rows at or before `position` must be bitwise unchanged when every later
    # key and value is permuted and perturbed: masked G entries are exact
    # zeros, so the sums over earlier rows see identical terms.
    seq_len = 6000
    q, k, v, W = make_bf16_inputs(12, 1, 2, seq_len, 64, 4, 4)
    beta = torch.tensor(1.0)
    future = torch.arange(position + 1, seq_len, device="cuda")
    shuffled = future[torch.randperm(len(future), device="cuda")]
    k2, v2 = k.clone(), v.clone()
    k2[:, :, future] = (3.0 * k[:, :, shuffled].float()).to(torch.bfloat16)
    v2[:, :, future] = -v[:, :, shuffled]
    for tile_tokens in (64, 2048):
        out = race.forward(q, k, v, W, beta, tile_tokens=tile_tokens)
        out2 = race.forward(q, k2, v2, W, beta, tile_tokens=tile_tokens)
        assert torch.equal(out[:, :, : position + 1], out2[:, :, : position + 1])
        assert not torch.equal(out[:, :, position + 1 :], out2[:, :, position + 1 :])


@pytest.mark.parametrize("seq_len", [129, 10000, 70000])
def test_runs_are_bitwise_identical(race, seq_len):
    q, k, v, W = make_bf16_inputs(13, 2, 4, seq_len, 64, 4, 4)
    beta = torch.tensor(1.0, device="cuda")
    first = race.forward_debug(q, k, v, W, beta)
    for _ in range(3):
        again = race.forward_debug(q, k, v, W, beta)
        for a, b in zip(first[:5], again[:5], strict=True):
            assert torch.equal(a, b)
        assert first[5] == again[5]


@pytest.mark.parametrize("seq_len, tile_tokens", [(1, 0), (777, 64), (5000, 128), (20000, 0), (20000, 2048)])
@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 4, 4), (128, 5, 2), (64, 2, 3)])
def test_debug_prefix_states_match_emulation(race, head_dim, num_planes, num_tables, seq_len, tile_tokens):
    skip_unless_fits(race, head_dim, num_planes, num_tables)
    q, k, v, W = make_bf16_inputs(case_seed(seq_len, head_dim, num_planes, num_tables), 1, 2,
                                  seq_len, head_dim, num_tables, num_planes)
    beta = 1.0
    out, prefix_mass, prefix_values, final_mass, final_values, used_tile = race.forward_debug(
        q, k, v, W, torch.tensor(beta), tile_tokens=tile_tokens
    )
    chunk = race.sub_chunk_tokens(head_dim, num_planes)
    assert used_tile % chunk == 0 and (tile_tokens == 0 or used_tile == tile_tokens)
    num_tiles = -(-seq_len // used_tile)
    corners = 1 << num_planes
    assert prefix_mass.shape == (1, 2, num_tiles, num_tables, corners)
    assert prefix_values.shape == (1, 2, num_tiles, num_tables, corners, head_dim)

    emulated = race_causal_chunked_emulation(
        q.double(), k.double(), v.double(), W.double(),
        torch.tensor(beta, dtype=torch.float64, device="cuda"), used_tile, chunk,
    )
    assert not prefix_mass[:, :, 0].any() and not prefix_values[:, :, 0].any()
    assert_states_close(prefix_mass, prefix_values, emulated.prefix_mass, emulated.prefix_values, v)
    assert_states_close(final_mass, final_values, emulated.final_mass, emulated.final_values, v)
    assert_within_rounding_floor(out, emulated.out, v, beta)
    assert torch.equal(out, race.forward(q, k, v, W, torch.tensor(beta), tile_tokens=used_tile))


def test_other_streams_cannot_leak_into_a_stream(race):
    # NaN in every other stream must not reach stream 0 (cross-stream indexing
    # and reads past T would show up as NaN).
    seq_len = 3000
    q, k, v, W = make_bf16_inputs(14, 1, 3, seq_len, 64, 4, 4)
    for tensor in (q, k, v):
        tensor[:, 1:] = float("nan")
    out = race.forward(q, k, v, W, torch.tensor(1.0), tile_tokens=128)
    assert torch.isfinite(out[:, 0]).all()
    ref = reference_output(q[:, :1], k[:, :1], v[:, :1], W, 1.0)
    assert_within_rounding_floor(out[:, :1], ref, v[:, :1], 1.0)


def test_beta_on_cpu_matches_beta_on_gpu(race):
    q, k, v, W = make_bf16_inputs(15, 1, 1, 777, 128, 2, 2)
    on_gpu = race.forward(q, k, v, W, torch.tensor(0.5, device="cuda"))
    on_cpu = race.forward(q, k, v, W, torch.tensor(0.5, dtype=torch.float64))
    assert torch.equal(on_gpu, on_cpu)


def test_automatic_tile_lengths(race):
    for head_dim, num_planes in ((64, 4), (128, 5)):
        if not race.fits_on_device(head_dim, num_planes, 2):
            continue
        chunk = race.sub_chunk_tokens(head_dim, num_planes)
        small = race.select_tile_tokens(1, 100, head_dim, num_planes, 2)
        assert small == chunk  # a short sequence gets the smallest tile
        large = race.select_tile_tokens(8, 1 << 21, head_dim, num_planes, 2)
        assert large % chunk == 0 and chunk <= large <= 4096


def test_empty_sequence(race):
    q, k, v, W = make_bf16_inputs(16, 1, 2, 0, 64, 2, 2)
    out, prefix_mass, prefix_values, final_mass, final_values, tile = race.forward_debug(
        q, k, v, W, torch.tensor(1.0)
    )
    assert out.shape == q.shape and tile == 0
    assert prefix_mass.shape[2] == 0 and not final_mass.any() and not final_values.any()


def test_rejects_invalid_inputs(race):
    q, k, v, W = make_bf16_inputs(17, 1, 1, 64, 64, 2, 2)
    q96, k96, v96, W96 = make_bf16_inputs(17, 1, 1, 64, 96, 2, 2)
    misaligned = torch.empty(1, 1, 64 * 64 + 1, dtype=torch.bfloat16, device="cuda")[..., 1:]
    misaligned = misaligned.view(1, 1, 64, 64)
    beta = torch.tensor(1.0)
    bad_calls = {
        "fp32 q": ((q.float(), k, v, W, beta), {}),
        "cpu k": ((q, k.cpu(), v, W, beta), {}),
        "non-contiguous v": ((q, k, v.transpose(2, 3).contiguous().transpose(2, 3), W, beta), {}),
        "misaligned q": ((misaligned, k, v, W, beta), {}),
        "head_dim 96": ((q96, k96, v96, W96, beta), {}),
        "bf16 W": ((q, k, v, W.to(torch.bfloat16), beta), {}),
        "P = 6": ((q, k, v, torch.randn(2, 6, 64, device="cuda"), beta), {}),
        "L = 5": ((q, k, v, torch.randn(5, 2, 64, device="cuda"), beta), {}),
        "shape mismatch": ((q, k[:, :, :32].contiguous(), v, W, beta), {}),
        "tile not a multiple of C": ((q, k, v, W, beta), {"tile_tokens": 48}),
        "negative tile": ((q, k, v, W, beta), {"tile_tokens": -64}),
    }
    for label, (args, kwargs) in bad_calls.items():
        with pytest.raises(RuntimeError):
            race.forward(*args, **kwargs)
            pytest.fail(f"accepted {label}")


@pytest.mark.skipif(os.environ.get("RACE_CAUSAL_HUGE") != "1", reason="set RACE_CAUSAL_HUGE=1")
def test_offsets_past_2_to_the_31(race):
    # 8 streams x 2^21 tokens x d = 128 is 2^31 elements: a 32-bit offset
    # overflows on the last stream. Check that stream's last 4096 outputs.
    head_dim, num_planes, seq_len, streams = 128, 5, 1 << 21, 8
    num_tables = 4 if race.fits_on_device(head_dim, num_planes, 4) else 2
    q, k, v, W = make_bf16_inputs(18, 1, streams, seq_len, head_dim, num_tables, num_planes)
    beta = 1.0 / math.sqrt(head_dim)
    out = race.forward(q, k, v, W, torch.tensor(beta))
    last = slice(streams - 1, streams)
    ref = reference_output(q[:, last], k[:, last], v[:, last], W, beta)[..., -4096:, :]
    assert_within_rounding_floor(out[:, last, -4096:], ref, v[:, last], beta)
