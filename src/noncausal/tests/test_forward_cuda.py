"""GPU tests of the race_noncausal extension against the fp64 reference.

Skipped when no CUDA device is present. The reference runs in fp64 on the GPU
on the same bf16-rounded inputs the kernel sees; tolerances are derived in
numerics.py.

Every test runs for both forward variants: the fp32-core kernels ("fp32") and
the tensor-core bucket build and query pass ("tc", tensor_cores=True). The tc
output is checked against its own derived bound and against the fp32 kernels
on the same inputs. At beta = 0 its corner probabilities are exact (every phi
is 2^-P), so there it must also meet the fp32 tolerance.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from numerics import (
    assert_buckets_close,
    assert_buckets_close_tc,
    assert_output_close,
    assert_output_close_tc,
    make_bf16_inputs,
    output_bound,
    output_bound_tc,
    tc_statistics,
)
from reference import bucket_sums, race_forward_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 4, 5]
TABLE_COUNTS = [1, 2, 4]
SEQ_LENS = [1, 16, 17, 127, 128, 129, 2047, 2048, 2049, 10000]
BATCH_HEADS = [(1, 1), (2, 2)]  # B * H = 1 and 4, exercising both grid axes
BETA = 1.0
VARIANTS = ["fp32", "tc"]


@pytest.fixture(scope="module")
def race():
    from build import load_extension

    return load_extension(verbose=False)


@pytest.fixture(params=VARIANTS)
def tensor_cores(request) -> bool:
    return request.param == "tc"


def case_seed(*params: int) -> int:
    return hash(params) % (2**31)


def reference_output(q, k, v, W, beta: float) -> torch.Tensor:
    beta64 = torch.tensor(beta, dtype=torch.float64, device=q.device)
    return race_forward_reference(q.double(), k.double(), v.double(), W.double(), beta64)


def check_output(race, out, q, k, v, W, beta: float, tensor_cores: bool) -> None:
    """fp32 path: against the fp64 reference. tc path: against the reference and the fp32 path."""
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    assert torch.isfinite(out).all()
    ref = reference_output(q, k, v, W, beta)
    if not tensor_cores:
        assert_output_close(out, ref, v, beta)
        return
    *_, rounding_bound = tc_statistics(q, k, v, W, beta)
    assert_output_close_tc(out, ref, rounding_bound, v, beta)
    # Each path is within its bound of the reference, so the two are within
    # the sum of the bounds of each other. Asserted separately so that a
    # failure names the fp32 kernels as the comparison point.
    fp32_out = race.forward(q, k, v, W, torch.tensor(beta, device="cuda"))
    difference = (out.double() - fp32_out.double()).abs()
    allowed = output_bound(ref, v, beta) + output_bound_tc(ref, rounding_bound, v, beta)
    worst = (difference / allowed).max().item()
    assert worst <= 1.0, f"tc and fp32 outputs differ by {worst:.3f}x the combined bound"


def run_and_check(race, batch, heads, seq_len, head_dim, num_planes, num_tables, beta,
                  tensor_cores):
    q, k, v, W = make_bf16_inputs(
        case_seed(batch, heads, seq_len, head_dim, num_planes, num_tables),
        batch, heads, seq_len, head_dim, num_tables, num_planes, device="cuda",
    )
    out = race.forward(q, k, v, W, torch.tensor(beta, device="cuda"), tensor_cores=tensor_cores)
    check_output(race, out, q, k, v, W, beta, tensor_cores)


@pytest.mark.parametrize("batch, heads", BATCH_HEADS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_forward_matches_reference(race, tensor_cores, head_dim, num_planes, num_tables, seq_len,
                                   batch, heads):
    run_and_check(race, batch, heads, seq_len, head_dim, num_planes, num_tables, BETA,
                  tensor_cores)


@pytest.mark.parametrize("beta", [1.0 / 128**0.5, 4.0])
@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 3, 3), (128, 5, 4)])
def test_forward_other_betas_and_odd_shapes(race, tensor_cores, head_dim, num_planes, num_tables,
                                            beta):
    run_and_check(race, 1, 3, 3001, head_dim, num_planes, num_tables, beta, tensor_cores)


@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 2, 2), (128, 5, 4)])
@pytest.mark.parametrize("seq_len", [2048 * 9 + 1, 2048 * 65])
def test_forward_multi_level_tree_reduce(race, tensor_cores, seq_len, head_dim, num_planes,
                                         num_tables):
    # 10 tiles need two reduce passes, 65 tiles need three; (128, 5, 4) is the
    # largest slice, so the most work per pass.
    run_and_check(race, 1, 2, seq_len, head_dim, num_planes, num_tables, BETA, tensor_cores)


@pytest.mark.parametrize("beta_value", [BETA, 4.0])
@pytest.mark.parametrize("seq_len", [127, 2049, 10000])
@pytest.mark.parametrize("num_tables", [2, 4])
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_debug_bucket_sums_match_reference(race, tensor_cores, head_dim, num_planes, num_tables,
                                           seq_len, beta_value):
    q, k, v, W = make_bf16_inputs(
        case_seed(seq_len, head_dim, num_planes, num_tables), 2, 2, seq_len, head_dim,
        num_tables, num_planes, device="cuda",
    )
    beta = torch.tensor(beta_value, device="cuda")
    out, mass, weighted_values = race.forward_debug(q, k, v, W, beta, tensor_cores=tensor_cores)
    corners = 1 << num_planes
    assert mass.shape == (2, 2, num_tables, corners)
    assert weighted_values.shape == (2, 2, num_tables, corners, head_dim)
    if tensor_cores:
        delta, ref_mass, ref_weighted_values, ref_weighted_abs_values, _ = tc_statistics(
            q, k, v, W, beta_value
        )
        assert_buckets_close_tc(mass, weighted_values, ref_mass, ref_weighted_values,
                                ref_weighted_abs_values, v, delta)
        # Against the fp32 kernels: each is within its bound of the reference.
        _, fp32_mass, fp32_weighted_values = race.forward_debug(q, k, v, W, beta)
        assert_buckets_close(fp32_mass, fp32_weighted_values, ref_mass, ref_weighted_values, v)
    else:
        ref_mass, ref_weighted_values = bucket_sums(k.double(), v.double(), W.double(),
                                                    beta.double())
        assert_buckets_close(mass, weighted_values, ref_mass, ref_weighted_values, v)
    assert torch.equal(out, race.forward(q, k, v, W, beta, tensor_cores=tensor_cores))


@pytest.mark.parametrize("seq_len", [129, 10000, 2048 * 65])
def test_runs_are_bitwise_identical(race, tensor_cores, seq_len):
    q, k, v, W = make_bf16_inputs(3, 1, 4, seq_len, 128, 4, 4, device="cuda")
    beta = torch.tensor(BETA, device="cuda")
    first = race.forward_debug(q, k, v, W, beta, tensor_cores=tensor_cores)
    for _ in range(3):
        again = race.forward_debug(q, k, v, W, beta, tensor_cores=tensor_cores)
        for a, b in zip(first, again, strict=True):
            assert torch.equal(a, b)


def test_zero_beta_gives_mean_of_values(race, tensor_cores):
    q, k, v, W = make_bf16_inputs(4, 1, 2, 5000, 64, 2, 3, device="cuda")
    out = race.forward(q, k, v, W, torch.tensor(0.0, device="cuda"), tensor_cores=tensor_cores)
    expected = v.double().mean(dim=2, keepdim=True).expand_as(out)
    assert_output_close(out, expected, v, 0.0)


@pytest.mark.parametrize("seq_len", [1, 17, 129, 2049, 10000])
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
@pytest.mark.parametrize("num_planes", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_tc_zero_beta_meets_fp32_tolerance(race, head_dim, num_planes, num_tables, seq_len):
    # At beta = 0 every phi is exactly 2^-P, which bf16 represents, so the
    # tensor-core path has no rounding beyond fp32 and must meet the fp32
    # path's tolerance. That tolerance sees a single key dropped from or added
    # to the sums (|v_j - mean| / N is well above 1e-5 max|v| here), which the
    # beta = 1 bound cannot at large N.
    q, k, v, W = make_bf16_inputs(
        case_seed(0, head_dim, num_planes, num_tables, seq_len), 1, 2, seq_len, head_dim,
        num_tables, num_planes, device="cuda",
    )
    beta = torch.tensor(0.0, device="cuda")
    out, mass, weighted_values = race.forward_debug(q, k, v, W, beta, tensor_cores=True)
    assert_output_close(out, v.double().mean(dim=2, keepdim=True).expand_as(out), v, 0.0)
    ref_mass, ref_weighted_values = bucket_sums(k.double(), v.double(), W.double(), beta.double())
    assert_buckets_close(mass, weighted_values, ref_mass, ref_weighted_values, v)


def test_beta_on_cpu_matches_beta_on_gpu(race, tensor_cores):
    q, k, v, W = make_bf16_inputs(5, 1, 1, 777, 128, 2, 2, device="cuda")
    on_gpu = race.forward(q, k, v, W, torch.tensor(0.5, device="cuda"), tensor_cores=tensor_cores)
    on_cpu = race.forward(q, k, v, W, torch.tensor(0.5), tensor_cores=tensor_cores)
    assert torch.equal(on_gpu, on_cpu)


def test_empty_sequence(race, tensor_cores):
    q, k, v, W = make_bf16_inputs(6, 1, 2, 0, 64, 2, 2, device="cuda")
    out, mass, weighted_values = race.forward_debug(q, k, v, W, torch.tensor(1.0),
                                                    tensor_cores=tensor_cores)
    assert out.shape == q.shape
    assert not mass.any() and not weighted_values.any()


def test_rejects_invalid_inputs(race, tensor_cores):
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
            race.forward(*args, tensor_cores=tensor_cores)
            pytest.fail(f"accepted {label}")


def offset_view(t: torch.Tensor, offset_elements: int) -> torch.Tensor:
    """A contiguous copy of t whose data starts offset_elements into a fresh allocation."""
    storage = torch.empty(t.numel() + offset_elements, dtype=t.dtype, device=t.device)
    view = storage[offset_elements:].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous()
    return view


def test_rejects_misaligned_contiguous_view(race, tensor_cores):
    # A view starting one element into its storage is contiguous but only
    # 2-byte aligned; the kernels' vector loads need a clear error.
    q, k, v, W = make_bf16_inputs(8, 1, 1, 64, 64, 2, 2, device="cuda")
    message = "16-byte aligned" if tensor_cores else "8-byte aligned"
    for position in range(3):
        args = [q, k, v]
        args[position] = offset_view(args[position], 1)
        with pytest.raises(RuntimeError, match=message):
            race.forward(*args, W, torch.tensor(1.0), tensor_cores=tensor_cores)


def test_tc_rejects_8_byte_aligned_view(race):
    # 8-byte alignment is enough for the fp32 path, not for the 16-byte
    # copies of the tensor-core path.
    q, k, v, W = make_bf16_inputs(9, 1, 1, 64, 64, 2, 2, device="cuda")
    beta = torch.tensor(1.0)
    shifted = offset_view(k, 4)
    assert torch.equal(race.forward(q, shifted, v, W, beta), race.forward(q, k, v, W, beta))
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        race.forward(q, shifted, v, W, beta, tensor_cores=True)


def test_every_template_instantiation_launches(race, tensor_cores):
    # Cheap smoke test over the full (d, P, L) grid, including P = 3 and L = 3
    # that the main cross product skips.
    for head_dim, num_planes, num_tables in itertools.product(HEAD_DIMS, range(1, 6), range(1, 5)):
        run_and_check(race, 1, 1, 300, head_dim, num_planes, num_tables, BETA, tensor_cores)
