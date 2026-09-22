"""GPU tests of the race_noncausal extension against the fp64 reference.

Skipped when no CUDA device is present. The reference runs in fp64 on the GPU
on the same bf16-rounded inputs the kernel sees; tolerances are derived in
numerics.py.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from numerics import assert_buckets_close, assert_output_close, make_bf16_inputs
from reference import bucket_sums, race_forward_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 4, 5]
TABLE_COUNTS = [1, 2, 4]
SEQ_LENS = [1, 16, 17, 127, 128, 129, 2047, 2048, 2049, 10000]
BATCH_HEADS = [(1, 1), (2, 2)]  # B * H = 1 and 4, exercising both grid axes
BETA = 1.0


@pytest.fixture(scope="module")
def race():
    from build import load_extension

    return load_extension(verbose=False)


def case_seed(*params: int) -> int:
    return hash(params) % (2**31)


def reference_output(q, k, v, W, beta: float) -> torch.Tensor:
    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    return race_forward_reference(q.double(), k.double(), v.double(), W.double(), beta64)


def run_and_check(race, batch, heads, seq_len, head_dim, num_planes, num_tables, beta):
    q, k, v, W = make_bf16_inputs(
        case_seed(batch, heads, seq_len, head_dim, num_planes, num_tables),
        batch, heads, seq_len, head_dim, num_tables, num_planes, device="cuda",
    )
    out = race.forward(q, k, v, W, torch.tensor(beta, device="cuda"))
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    assert torch.isfinite(out).all()
    assert_output_close(out, reference_output(q, k, v, W, beta), v, beta)


@pytest.mark.parametrize("batch, heads", BATCH_HEADS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_forward_matches_reference(race, head_dim, num_planes, num_tables, seq_len, batch, heads):
    run_and_check(race, batch, heads, seq_len, head_dim, num_planes, num_tables, BETA)


@pytest.mark.parametrize("beta", [1.0 / 128**0.5, 4.0])
@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 3, 3), (128, 5, 4)])
def test_forward_other_betas_and_odd_shapes(race, head_dim, num_planes, num_tables, beta):
    run_and_check(race, 1, 3, 3001, head_dim, num_planes, num_tables, beta)


@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 2, 2), (128, 5, 4)])
@pytest.mark.parametrize("seq_len", [2048 * 9 + 1, 2048 * 65])
def test_forward_multi_level_tree_reduce(race, seq_len, head_dim, num_planes, num_tables):
    # 10 tiles need two reduce passes, 65 tiles need three; (128, 5, 4) is the
    # largest slice, so the most work per pass.
    run_and_check(race, 1, 2, seq_len, head_dim, num_planes, num_tables, BETA)


@pytest.mark.parametrize("beta_value", [BETA, 4.0])
@pytest.mark.parametrize("seq_len", [127, 2049, 10000])
@pytest.mark.parametrize("num_tables", [2, 4])
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_debug_bucket_sums_match_reference(race, head_dim, num_planes, num_tables, seq_len, beta_value):
    q, k, v, W = make_bf16_inputs(
        case_seed(seq_len, head_dim, num_planes, num_tables), 2, 2, seq_len, head_dim,
        num_tables, num_planes, device="cuda",
    )
    beta = torch.tensor(beta_value, device="cuda")
    out, mass, weighted_values = race.forward_debug(q, k, v, W, beta)
    corners = 1 << num_planes
    assert mass.shape == (2, 2, num_tables, corners)
    assert weighted_values.shape == (2, 2, num_tables, corners, head_dim)
    ref_mass, ref_weighted_values = bucket_sums(k.double(), v.double(), W.double(), beta.double())
    assert_buckets_close(mass, weighted_values, ref_mass, ref_weighted_values, v)
    assert torch.equal(out, race.forward(q, k, v, W, beta))


@pytest.mark.parametrize("seq_len", [129, 10000, 2048 * 65])
def test_runs_are_bitwise_identical(race, seq_len):
    q, k, v, W = make_bf16_inputs(3, 1, 4, seq_len, 128, 4, 4, device="cuda")
    beta = torch.tensor(BETA, device="cuda")
    first = race.forward_debug(q, k, v, W, beta)
    for _ in range(3):
        again = race.forward_debug(q, k, v, W, beta)
        for a, b in zip(first, again, strict=True):
            assert torch.equal(a, b)


def test_zero_beta_gives_mean_of_values(race):
    q, k, v, W = make_bf16_inputs(4, 1, 2, 5000, 64, 2, 3, device="cuda")
    out = race.forward(q, k, v, W, torch.tensor(0.0, device="cuda"))
    expected = v.double().mean(dim=2, keepdim=True).expand_as(out)
    assert_output_close(out, expected, v, 0.0)


def test_beta_on_cpu_matches_beta_on_gpu(race):
    q, k, v, W = make_bf16_inputs(5, 1, 1, 777, 128, 2, 2, device="cuda")
    on_gpu = race.forward(q, k, v, W, torch.tensor(0.5, device="cuda"))
    on_cpu = race.forward(q, k, v, W, torch.tensor(0.5))
    assert torch.equal(on_gpu, on_cpu)


def test_empty_sequence(race):
    q, k, v, W = make_bf16_inputs(6, 1, 2, 0, 64, 2, 2, device="cuda")
    out, mass, weighted_values = race.forward_debug(q, k, v, W, torch.tensor(1.0))
    assert out.shape == q.shape
    assert not mass.any() and not weighted_values.any()


def test_rejects_invalid_inputs(race):
    q, k, v, W = make_bf16_inputs(7, 1, 1, 64, 64, 2, 2, device="cuda")
    q96, k96, v96, W96 = make_bf16_inputs(7, 1, 1, 64, 96, 2, 2, device="cuda")
    beta = torch.tensor(1.0)
    bad_calls = {
        "fp32 q": (q.float(), k, v, W, beta),
        "cpu k": (q, k.cpu(), v, W, beta),
        "non-contiguous v": (q, k, v.transpose(2, 3).contiguous().transpose(2, 3), W, beta),
        "head_dim 96": (q96, k96, v96, W96, beta),
        "bf16 W": (q, k, v, W.to(torch.bfloat16), beta),
        "P = 6": (q, k, v, torch.randn(2, 6, 64, device="cuda"), beta),
        "L = 5": (q, k, v, torch.randn(5, 2, 64, device="cuda"), beta),
        "shape mismatch": (q, k[:, :, :32].contiguous(), v, W, beta),
    }
    for label, args in bad_calls.items():
        with pytest.raises(RuntimeError):
            race.forward(*args)
            pytest.fail(f"accepted {label}")


def test_rejects_misaligned_contiguous_view(race):
    # A view starting one element into its storage is contiguous but only
    # 2-byte aligned; the kernels' 8-byte vector loads need a clear error.
    q, k, v, W = make_bf16_inputs(8, 1, 1, 64, 64, 2, 2, device="cuda")
    storage = torch.empty(q.numel() + 1, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1:].view(q.shape)
    misaligned.copy_(q)
    assert misaligned.is_contiguous()
    with pytest.raises(RuntimeError, match="8-byte aligned"):
        race.forward(misaligned, k, v, W, torch.tensor(1.0))


def test_every_template_instantiation_launches(race):
    # Cheap smoke test over the full (d, P, L) grid, including P = 3 and L = 3
    # that the main cross product skips.
    for head_dim, num_planes, num_tables in itertools.product(HEAD_DIMS, range(1, 6), range(1, 5)):
        run_and_check(race, 1, 1, 300, head_dim, num_planes, num_tables, BETA)
