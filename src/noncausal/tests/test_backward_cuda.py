"""GPU tests of the backward kernels and the autograd wrapper (ops.py).

Skipped when no CUDA device is present. The reference is the fp64 vector-
Jacobian product race_backward_reference (checked against torch.autograd of
race_forward_reference in test_backward_reference.py) on the GPU, evaluated on
the same bf16-rounded q, k, v and dO the kernels see. Tolerances and their
derivation are in numerics.py.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from numerics import (
    assert_beta_grad_close,
    assert_grad_close,
    make_bf16_grad_output,
    make_bf16_inputs,
    reference_backward,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HEAD_DIMS = [64, 128]
PLANE_COUNTS = [1, 2, 4, 5]
TABLE_COUNTS = [1, 2, 4]
SEQ_LENS = [1, 16, 17, 127, 129, 2049]
BATCH_HEADS = [(1, 1), (2, 2)]
BETA = 1.0


@pytest.fixture(scope="module")
def race():
    from build import load_extension

    return load_extension(verbose=False)


def case_seed(*params) -> int:
    return hash(params) % (2**31)


def kernel_backward(race, grad_o, q, k, v, W, beta):
    out, bucket_totals = race.forward_train(q, k, v, W, beta)
    assert torch.equal(out, race.forward(q, k, v, W, beta))
    return race.backward(grad_o, q, k, v, W, beta, bucket_totals)


def run_and_check(race, batch, heads, seq_len, head_dim, num_planes, num_tables, beta):
    seed = case_seed(batch, heads, seq_len, head_dim, num_planes, num_tables, beta)
    q, k, v, W = make_bf16_inputs(
        seed, batch, heads, seq_len, head_dim, num_tables, num_planes, device="cuda"
    )
    grad_o = make_bf16_grad_output(seed + 1, q)
    grads = kernel_backward(race, grad_o, q, k, v, W, torch.tensor(beta, device="cuda"))
    for grad, like in zip(grads[:3], (q, k, v), strict=True):
        assert grad.dtype == torch.bfloat16 and grad.shape == like.shape
        assert torch.isfinite(grad).all()
    assert grads[3].dtype == torch.float32 and grads[3].dim() == 0
    ref, scales = reference_backward(grad_o, q, k, v, W, beta)
    for name, grad, expected, scale in zip(("dq", "dk", "dv"), grads[:3], ref[:3], scales[:3], strict=True):
        assert_grad_close(grad, expected, scale, beta, name)
    assert_beta_grad_close(grads[3], ref[3], scales[3], beta)


@pytest.mark.parametrize("batch, heads", BATCH_HEADS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_tables", TABLE_COUNTS)
@pytest.mark.parametrize("num_planes", PLANE_COUNTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_backward_matches_reference(race, head_dim, num_planes, num_tables, seq_len, batch, heads):
    run_and_check(race, batch, heads, seq_len, head_dim, num_planes, num_tables, BETA)


@pytest.mark.parametrize("beta", [None, 1.0, 4.0])  # None: 1 / sqrt(d)
@pytest.mark.parametrize("head_dim, num_planes, num_tables", [(64, 3, 3), (128, 5, 4), (128, 4, 4)])
def test_backward_betas(race, head_dim, num_planes, num_tables, beta):
    beta = head_dim**-0.5 if beta is None else beta
    run_and_check(race, 1, 3, 3001, head_dim, num_planes, num_tables, beta)


@pytest.mark.parametrize("seq_len", [2048 * 9 + 1, 2048 * 65])
def test_backward_multi_level_tree_reduce(race, seq_len):
    # 10 build tiles need two reduce passes over dA/dB, 65 tiles need three;
    # P = 5, d = 128 is the largest slice.
    run_and_check(race, 1, 1, seq_len, 128, 5, 2, BETA)


@pytest.mark.parametrize("seq_len", [129, 2048 * 65])
def test_backward_runs_are_bitwise_identical(race, seq_len):
    q, k, v, W = make_bf16_inputs(3, 1, 4, seq_len, 128, 4, 5, device="cuda")
    grad_o = make_bf16_grad_output(4, q)
    beta = torch.tensor(BETA, device="cuda")
    first = kernel_backward(race, grad_o, q, k, v, W, beta)
    for _ in range(2):
        again = kernel_backward(race, grad_o, q, k, v, W, beta)
        for a, b in zip(first, again, strict=True):
            assert torch.equal(a, b)


def test_every_template_instantiation_backward(race):
    # Cheap smoke test over the full (d, P, L) grid, including P = 3 and L = 3.
    for head_dim, num_planes, num_tables in itertools.product(HEAD_DIMS, range(1, 6), range(1, 5)):
        run_and_check(race, 1, 1, 300, head_dim, num_planes, num_tables, BETA)


def test_zero_beta_gives_exactly_zero_dq_dk(race):
    # Every dq, dk chain carries a factor beta; d beta is 0 at beta = 0 too
    # (O is even in beta, see test_backward_reference.py), up to fp32 noise.
    q, k, v, W = make_bf16_inputs(5, 1, 2, 777, 64, 2, 3, device="cuda")
    grad_o = make_bf16_grad_output(6, q)
    dq, dk, dv, dbeta = kernel_backward(race, grad_o, q, k, v, W, torch.tensor(0.0, device="cuda"))
    assert not dq.any() and not dk.any()
    ref, scales = reference_backward(grad_o, q, k, v, W, 0.0)
    assert_grad_close(dv, ref[2], scales[2], 0.0, "dv")
    assert_beta_grad_close(dbeta, ref[3], scales[3], 0.0)


def test_empty_sequence_backward(race):
    q, k, v, W = make_bf16_inputs(7, 1, 2, 0, 64, 2, 2, device="cuda")
    beta = torch.tensor(1.0)
    out, bucket_totals = race.forward_train(q, k, v, W, beta)
    assert bucket_totals.shape == (1, 2, 2, 4 * 65) and not bucket_totals.any()
    dq, dk, dv, dbeta = race.backward(torch.empty_like(q), q, k, v, W, beta, bucket_totals)
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    assert dbeta.item() == 0.0


def test_backward_rejects_invalid_inputs(race):
    q, k, v, W = make_bf16_inputs(8, 1, 1, 64, 64, 2, 2, device="cuda")
    beta = torch.tensor(1.0)
    _, bucket_totals = race.forward_train(q, k, v, W, beta)
    grad_o = make_bf16_grad_output(9, q)
    misaligned = torch.empty(q.numel() + 1, dtype=torch.bfloat16, device="cuda")[1:].view(q.shape)
    bad_calls = {
        "fp32 grad_o": ((grad_o.float(), q, k, v, W, beta, bucket_totals), "bfloat16"),
        "non-contiguous grad_o": (
            (grad_o.transpose(2, 3).contiguous().transpose(2, 3), q, k, v, W, beta, bucket_totals),
            "contiguous",
        ),
        "grad_o shape": ((grad_o[:, :, :32].contiguous(), q, k, v, W, beta, bucket_totals), "shape"),
        "misaligned grad_o": ((misaligned, q, k, v, W, beta, bucket_totals), "8-byte aligned"),
        "totals shape": ((grad_o, q, k, v, W, beta, bucket_totals[..., :-1].contiguous()), "bucket_totals"),
        "totals dtype": ((grad_o, q, k, v, W, beta, bucket_totals.double()), "bucket_totals"),
        "totals on cpu": ((grad_o, q, k, v, W, beta, bucket_totals.cpu()), "bucket_totals"),
    }
    for label, (args, message) in bad_calls.items():
        with pytest.raises(RuntimeError, match=message):
            race.backward(*args)
            pytest.fail(f"accepted {label}")


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ops(race):
    import ops as ops_module

    return ops_module


def leaf_inputs(seed, seq_len=500, head_dim=64, num_planes=3, num_tables=2):
    q, k, v, W = make_bf16_inputs(seed, 1, 2, seq_len, head_dim, num_tables, num_planes, device="cuda")
    beta = torch.tensor(0.9, device="cuda")
    return [t.clone().requires_grad_() for t in (q, k, v, beta)], W


def test_autograd_matches_binding(race, ops):
    (q, k, v, beta), W = leaf_inputs(10)
    out = ops.race_attention(q, k, v, W, beta)
    grad_o = make_bf16_grad_output(11, out)
    out.backward(grad_o)
    expected = kernel_backward(race, grad_o, q.detach(), k.detach(), v.detach(), W, beta.detach())
    for leaf, grad in zip((q, k, v, beta), expected, strict=True):
        assert leaf.grad.dtype == leaf.dtype and leaf.grad.shape == leaf.shape
        assert torch.equal(leaf.grad, grad.reshape(leaf.shape))


def test_stride_zero_grad_output(race, ops):
    # o.sum() hands backward an expanded (stride-0) gradient of ones.
    (q, k, v, beta), W = leaf_inputs(12)
    out = ops.race_attention(q, k, v, W, beta)
    out.sum().backward()
    ones = torch.ones_like(out)
    expected = kernel_backward(race, ones, q.detach(), k.detach(), v.detach(), W, beta.detach())
    for leaf, grad in zip((q, k, v, beta), expected, strict=True):
        assert torch.equal(leaf.grad, grad.reshape(leaf.shape))


def test_only_requested_gradients(ops):
    (q, k, v, _), W = leaf_inputs(13)
    q, k = q.detach(), k.detach()
    out = ops.race_attention(q, k, v, W, 0.7)  # a float beta has no gradient
    out.float().sum().backward()
    assert q.grad is None and k.grad is None
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_cpu_beta_gets_a_cpu_gradient(race, ops):
    (q, k, v, _), W = leaf_inputs(14)
    beta = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)  # CPU, fp64, shape [1]
    out = ops.race_attention(q, k, v, W, beta)
    grad_o = make_bf16_grad_output(15, out)
    out.backward(grad_o)
    assert beta.grad.device.type == "cpu" and beta.grad.dtype == torch.float64
    assert beta.grad.shape == (1,)
    expected = kernel_backward(race, grad_o, q.detach(), k.detach(), v.detach(), W, beta.detach())
    assert beta.grad.item() == expected[3].item()


def test_double_backward_raises(ops):
    (q, k, v, beta), W = leaf_inputs(16)
    out = ops.race_attention(q, k, v, W, beta)
    # The loss is quadratic in out so that the gradient reaching the wrapped
    # backward (2 * out) carries a graph: once_differentiable only installs
    # its error node when the incoming gradient requires grad. With a plain
    # sum the gradient is a constant, dq comes back as a plain tensor, and
    # the second backward would fail with a generic "does not require grad"
    # instead of exercising the decorator.
    (grad_q,) = torch.autograd.grad(out.float().square().sum(), q, create_graph=True)
    with pytest.raises(RuntimeError, match="differentiate twice"):
        grad_q.float().sum().backward()


def test_trainable_w_is_rejected(ops):
    (q, k, v, beta), W = leaf_inputs(17)
    with pytest.raises(ValueError, match="no gradient"):
        ops.race_attention(q, k, v, W.requires_grad_(), beta)
