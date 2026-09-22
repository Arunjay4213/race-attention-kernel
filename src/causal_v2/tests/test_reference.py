"""CPU tests of the causal references and of the chunked kernel emulation (reference.py).

Every tensor is tiny (d = 16, S <= 32, B * H <= 2), so the file runs in a few
seconds and well under 1 GB.
"""
from __future__ import annotations

import math

import pytest
import torch

from race_baseline import BatchedACE
from reference import (
    features,
    race_causal_chunked_emulation,
    race_causal_dense_reference,
    race_causal_perbucket_reference,
    race_causal_reference,
)

HEAD_DIM = 16
BETAS = [1.0 / math.sqrt(HEAD_DIM), 0.5, 1.0, 2.0]
FP64_TOL = 1e-12


def make_inputs(seed: int, batch: int, heads: int, seq_len: int, num_tables: int, num_planes: int,
                head_dim: int = HEAD_DIM, dtype: torch.dtype = torch.float64):
    """Random q, k, v [B, H, T, d] and planes W [L, P, d] ~ N(0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    q, k, v = (torch.randn(shape, generator=gen, dtype=dtype) for _ in range(3))
    W = torch.randn(num_tables, num_planes, head_dim, generator=gen, dtype=dtype)
    return q, k, v, W


def as_beta(value: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype)


@pytest.mark.parametrize("num_planes, num_tables", [(1, 1), (2, 3), (4, 2), (5, 1)])
@pytest.mark.parametrize("beta_value", BETAS)
def test_prefix_reference_matches_dense_masked_form(num_planes, num_tables, beta_value):
    # T = 45 with 16-token blocks puts block edges inside the sequence.
    q, k, v, W = make_inputs(1, 2, 1, 45, num_tables, num_planes)
    beta = as_beta(beta_value)
    dense = race_causal_dense_reference(q, k, v, W, beta)
    prefix = race_causal_reference(q, k, v, W, beta, block_tokens=16)
    torch.testing.assert_close(prefix, dense, rtol=0.0, atol=FP64_TOL)


@pytest.mark.parametrize("num_planes, num_tables", [(2, 2), (4, 4), (5, 1)])
def test_bridge_perbucket_reference_matches_race_baseline(num_planes, num_tables):
    # BatchedACE computes softmax(tanh(x planes_T) / sqrt(d) @ corners): beta = 1/sqrt(d).
    batch, heads, seq_len = 2, 2, 37
    q, k, v, W = make_inputs(2, batch, heads, seq_len, num_tables, num_planes)
    ace = BatchedACE(HEAD_DIM, num_planes, num_tables, 1).double()
    ace.planes_T = W.reshape(num_tables * num_planes, HEAD_DIM).T.contiguous()

    def to_repo(x: torch.Tensor) -> torch.Tensor:  # [B, H, T, d] -> [M=1, B, T, H, d]
        return x.permute(0, 2, 1, 3).unsqueeze(0)

    repo = ace(to_repo(k), to_repo(v), to_repo(q))[0].permute(0, 2, 1, 3)
    ours = race_causal_perbucket_reference(q, k, v, W, as_beta(1.0 / math.sqrt(HEAD_DIM)), block_tokens=16)
    torch.testing.assert_close(ours, repo, rtol=0.0, atol=FP64_TOL)


@pytest.mark.parametrize("beta_value", BETAS)
def test_constant_values_come_back_unchanged(beta_value):
    # Algorithm 1 output is a convex combination of the values, so V = c gives c.
    q, k, _, W = make_inputs(3, 1, 2, 300, 2, 4)
    v = torch.full_like(q, 0.7)
    beta = as_beta(beta_value)
    reference = race_causal_reference(q, k, v, W, beta)
    emulated = race_causal_chunked_emulation(q, k, v, W, beta, tile_tokens=128, chunk_tokens=32).out
    torch.testing.assert_close(reference, v, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(emulated, v, rtol=0.0, atol=1e-14)


def test_first_output_is_the_first_value():
    q, k, v, W = make_inputs(4, 1, 2, 1, 3, 3)
    out = race_causal_chunked_emulation(q, k, v, W, as_beta(1.0), tile_tokens=64, chunk_tokens=64).out
    torch.testing.assert_close(out, v, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("seq_len", [1, 63, 64, 65, 2047, 2048, 2049, 4097])
@pytest.mark.parametrize("beta_value", BETAS)
def test_chunked_emulation_matches_reference(seq_len, beta_value):
    q, k, v, W = make_inputs(5, 1, 2, seq_len, 2, 2)
    beta = as_beta(beta_value)
    reference = race_causal_reference(q, k, v, W, beta)
    emulated = race_causal_chunked_emulation(q, k, v, W, beta, tile_tokens=256, chunk_tokens=64)
    torch.testing.assert_close(emulated.out, reference, rtol=0.0, atol=FP64_TOL)


@pytest.mark.parametrize(
    "seq_len, tile_tokens, chunk_tokens",
    [(1, 32, 32), (31, 32, 32), (33, 64, 32), (200, 64, 64), (1000, 128, 32), (777, 2048, 64)],
)
@pytest.mark.parametrize("num_planes, num_tables", [(1, 1), (3, 3), (5, 2)])
def test_chunked_emulation_tile_and_chunk_sizes(seq_len, tile_tokens, chunk_tokens, num_planes, num_tables):
    q, k, v, W = make_inputs(6, 2, 1, seq_len, num_tables, num_planes)
    beta = as_beta(1.0)
    reference = race_causal_reference(q, k, v, W, beta)
    emulated = race_causal_chunked_emulation(q, k, v, W, beta, tile_tokens, chunk_tokens)
    torch.testing.assert_close(emulated.out, reference, rtol=0.0, atol=FP64_TOL)


@pytest.mark.parametrize("seq_len", [64, 65, 1000])
def test_emulation_prefix_states_are_exclusive_tile_prefixes(seq_len):
    tile_tokens, num_tables, num_planes = 128, 2, 3
    q, k, v, W = make_inputs(7, 1, 2, seq_len, num_tables, num_planes)
    beta = as_beta(0.5)
    result = race_causal_chunked_emulation(q, k, v, W, beta, tile_tokens, chunk_tokens=32)
    phi_k = features(k, W, beta)  # [B, H, T, S]
    num_tiles = -(-seq_len // tile_tokens)
    corners = 1 << num_planes
    assert result.prefix_mass.shape == (1, 2, num_tiles, num_tables, corners)
    assert result.prefix_values.shape == (1, 2, num_tiles, num_tables, corners, HEAD_DIM)
    for tile in range(num_tiles):
        before = slice(0, tile * tile_tokens)  # keys strictly before the tile
        mass = phi_k[:, :, before].sum(2).reshape(1, 2, num_tables, corners)
        values = torch.einsum("bhts,bhtd->bhsd", phi_k[:, :, before], v[:, :, before])
        torch.testing.assert_close(result.prefix_mass[:, :, tile], mass, rtol=0.0, atol=1e-11)
        torch.testing.assert_close(
            result.prefix_values[:, :, tile], values.reshape(1, 2, num_tables, corners, HEAD_DIM),
            rtol=0.0, atol=1e-11,
        )
    torch.testing.assert_close(result.final_mass, phi_k.sum(2).reshape(1, 2, num_tables, corners),
                               rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("position", [0, 63, 64, 100, 299])
def test_future_tokens_do_not_change_the_past(position):
    # Permute and perturb every key and value after `position`; outputs at and
    # before `position` must not move, in both the reference and the emulation.
    seq_len = 300
    q, k, v, W = make_inputs(8, 1, 2, seq_len, 2, 4)
    beta = as_beta(1.0)
    future = torch.arange(position + 1, seq_len)
    shuffled = future[torch.randperm(len(future), generator=torch.Generator().manual_seed(9))]
    k2, v2 = k.clone(), v.clone()
    k2[:, :, future] = 3.0 * k[:, :, shuffled]
    v2[:, :, future] = -v[:, :, shuffled]
    past = slice(0, position + 1)
    for run in (
        lambda k_, v_: race_causal_reference(q, k_, v_, W, beta),
        lambda k_, v_: race_causal_chunked_emulation(q, k_, v_, W, beta, 128, 32).out,
    ):
        torch.testing.assert_close(run(k2, v2)[:, :, past], run(k, v)[:, :, past], rtol=0.0, atol=FP64_TOL)


def test_last_query_matches_noncausal_forward():
    # The last query sees every key, so it equals the non-causal output there.
    from reference import _noncausal

    q, k, v, W = make_inputs(10, 1, 2, 90, 3, 2)
    beta = as_beta(1.5)
    causal = race_causal_reference(q, k, v, W, beta)
    noncausal = _noncausal.race_forward_reference(q[:, :, -1:], k, v, W, beta)
    torch.testing.assert_close(causal[:, :, -1:], noncausal, rtol=0.0, atol=FP64_TOL)


def test_output_is_a_convex_combination_of_past_values():
    q, k, v, W = make_inputs(11, 1, 1, 120, 2, 3)
    out = race_causal_reference(q, k, v, W, as_beta(2.0))
    running_max = torch.cummax(v, dim=2).values
    running_min = -torch.cummax(-v, dim=2).values
    assert (out <= running_max + 1e-12).all()
    assert (out >= running_min - 1e-12).all()


def test_fp32_emulation_is_within_fp32_accuracy():
    # The kernels' fp32 structure should be ~1e-6 relative to max|v|, far below
    # bf16 rounding (2^-8); a gross structural error would show up here.
    q, k, v, W = make_inputs(12, 1, 2, 1500, 4, 4, dtype=torch.float32)
    beta = torch.tensor(1.0)
    emulated = race_causal_chunked_emulation(q, k, v, W, beta, 256, 64).out
    reference = race_causal_reference(q.double(), k.double(), v.double(), W.double(), beta.double())
    error = (emulated.double() - reference).abs().max() / v.abs().max()
    assert error < 1e-5


def test_rejects_tile_not_a_multiple_of_chunk():
    q, k, v, W = make_inputs(13, 1, 1, 10, 1, 1)
    with pytest.raises(ValueError):
        race_causal_chunked_emulation(q, k, v, W, as_beta(1.0), tile_tokens=96, chunk_tokens=64)
