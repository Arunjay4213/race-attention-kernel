// PyTorch bindings for the non-causal RACE kernels (race_fwd.cu,
// race_fwd_tc.cu, race_bwd.cu).
//
//   forward(q, k, v, W, beta, tensor_cores=False) -> o
//   forward_debug(q, k, v, W, beta, tensor_cores=False) -> (o, A, B)
//   forward_train(q, k, v, W, beta, tensor_cores=False) -> (o, bucket_totals)
//   backward(grad_o, q, k, v, W, beta, bucket_totals) -> (dq, dk, dv, dbeta)
//
// tensor_cores=True runs the tensor-core bucket build and query pass
// (race_fwd_tc.h) instead of the fp32-core ones. It computes the same
// function with bf16-rounded corner probabilities (tolerance in
// tests/numerics.py), writes the same bucket totals, and additionally needs
// q, k and v 16-byte aligned. The backward always runs on fp32 cores and
// accepts the totals of either forward.
//
// q, k, v: [B, H, N, d] bf16 CUDA, contiguous, identical shapes.
// W: [L, P, d] fp32 on the same device. beta: one-element floating tensor on
// any device (a CUDA beta is read by the kernels in place, no host sync).
// o: [B, H, N, d] bf16. A: [B, H, L, R] fp32. B: [B, H, L, R, d] fp32.
// bucket_totals: [B, H, L, R * (d + 1)] fp32, the reduced A and B in the
// kernels' slice layout (B rows, then A), which is all the backward needs
// from the forward. grad_o, dq, dk, dv: [B, H, N, d] bf16. dbeta: 0-dim fp32
// on q's device, summed over every batch and head.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "race_bwd.h"
#include "race_fwd.h"
#include "race_fwd_tc.h"

namespace {

// The widest global access in the fp32-core kernels is 8 bytes per lane (4 bf16);
// the tensor-core kernels copy rows with 16-byte cp.async and store 16-byte vectors.
constexpr int64_t kRequiredAlignmentBytes = 8;
constexpr int64_t kTensorCoreAlignmentBytes = static_cast<int64_t>(race::kTensorCoreAlignmentBytes);

void check_token_tensor(const at::Tensor& t, const char* name, const at::Tensor& like,
                        bool tensor_cores = false) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", t.scalar_type());
    TORCH_CHECK(t.dim() == 4, name, " must be [B, H, N, d], got ", t.sizes());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.sizes() == like.sizes(), name, " shape ", t.sizes(), " differs from q ", like.sizes());
    TORCH_CHECK(t.device() == like.device(), name, " must be on the same device as q");
    const uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
    if (tensor_cores) {
        TORCH_CHECK(address % kTensorCoreAlignmentBytes == 0, name,
                    " data must be 16-byte aligned for tensor_cores=True (clone the tensor if it "
                    "is a view)");
    } else {
        TORCH_CHECK(address % kRequiredAlignmentBytes == 0, name,
                    " data must be 8-byte aligned (clone the tensor if it is a view)");
    }
}

race::ForwardShape validate_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   const at::Tensor& W, const at::Tensor& beta,
                                   bool tensor_cores = false) {
    check_token_tensor(q, "q", q, tensor_cores);
    check_token_tensor(k, "k", q, tensor_cores);
    check_token_tensor(v, "v", q, tensor_cores);

    TORCH_CHECK(W.device() == q.device(), "W must be on the same device as q");
    TORCH_CHECK(W.scalar_type() == at::kFloat, "W must be float32, got ", W.scalar_type());
    TORCH_CHECK(W.dim() == 3, "W must be [L, P, d], got ", W.sizes());
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(W.size(2) == q.size(3), "W head_dim ", W.size(2), " differs from q ", q.size(3));
    TORCH_CHECK(beta.numel() == 1, "beta must have exactly one element");
    TORCH_CHECK(at::isFloatingType(beta.scalar_type()), "beta must be a floating tensor");

    const int64_t batch_heads = q.size(0) * q.size(1);
    const int64_t seq_len = q.size(2);
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "head_dim must be 64 or 128, got ", q.size(3));
    TORCH_CHECK(W.size(1) >= 1 && W.size(1) <= 5, "num_planes P must be in 1..5, got ", W.size(1));
    TORCH_CHECK(W.size(0) >= 1 && W.size(0) <= 4, "num_tables L must be in 1..4, got ", W.size(0));
    TORCH_CHECK(batch_heads <= 65535, "batch * heads must be <= 65535, got ", batch_heads);
    TORCH_CHECK(seq_len <= std::numeric_limits<int>::max() - race::kBuildTileTokens,
                "sequence length too large: ", seq_len);

    race::ForwardShape shape;
    shape.batch_heads = static_cast<int>(batch_heads);
    shape.seq_len = static_cast<int>(seq_len);
    shape.head_dim = static_cast<int>(q.size(3));
    shape.num_planes = static_cast<int>(W.size(1));
    shape.num_tables = static_cast<int>(W.size(0));
    return shape;
}

const __nv_bfloat16* bf16_ptr(const at::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

__nv_bfloat16* bf16_ptr(at::Tensor& t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

// Runs the three kernels and returns (o, workspace). Tile 0 of the workspace
// holds the reduced A and B afterwards. The workspace is undefined when there
// is nothing to compute (N == 0 or B * H == 0).
std::tuple<at::Tensor, at::Tensor> run_forward(const at::Tensor& q, const at::Tensor& k,
                                               const at::Tensor& v, const at::Tensor& W,
                                               const at::Tensor& beta,
                                               const race::ForwardShape& shape,
                                               bool tensor_cores) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    at::Tensor out = at::empty_like(q);
    if (shape.seq_len == 0 || shape.batch_heads == 0) return {out, at::Tensor()};

    const at::Tensor beta_device = beta.to(q.device(), at::kFloat).contiguous();
    at::Tensor workspace = at::empty({static_cast<int64_t>(race::workspace_floats(shape))},
                                     q.options().dtype(at::kFloat));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // at::empty_like allocates out from the caching allocator, whose blocks
    // are at least 256-byte aligned, so out meets the tensor-core alignment.
    const auto launch = tensor_cores ? race::race_forward_tc : race::race_forward;
    C10_CUDA_CHECK(launch(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), W.data_ptr<float>(),
                          beta_device.data_ptr<float>(), workspace.data_ptr<float>(),
                          bf16_ptr(out), shape, stream));
    return {out, workspace};
}

at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   const at::Tensor& W, const at::Tensor& beta, bool tensor_cores) {
    const race::ForwardShape shape = validate_inputs(q, k, v, W, beta, tensor_cores);
    return std::get<0>(run_forward(q, k, v, W, beta, shape, tensor_cores));
}

// Tile 0 of the forward workspace, [B, H, L, R * (d + 1)]. Cloned so that
// the (much larger) per-tile workspace can be freed.
at::Tensor bucket_totals_from_workspace(const at::Tensor& workspace, const at::Tensor& q,
                                        const at::Tensor& W) {
    const int64_t batch = q.size(0), heads = q.size(1), head_dim = q.size(3);
    const int64_t num_tables = W.size(0), corners = int64_t{1} << W.size(1);
    const int64_t slice = corners * (head_dim + 1);
    if (!workspace.defined()) {
        return at::zeros({batch, heads, num_tables, slice}, q.options().dtype(at::kFloat));
    }
    return workspace.narrow(0, 0, batch * heads * num_tables * slice)
        .view({batch, heads, num_tables, slice})
        .clone();
}

std::tuple<at::Tensor, at::Tensor> forward_train(const at::Tensor& q, const at::Tensor& k,
                                                 const at::Tensor& v, const at::Tensor& W,
                                                 const at::Tensor& beta, bool tensor_cores) {
    const race::ForwardShape shape = validate_inputs(q, k, v, W, beta, tensor_cores);
    auto [out, workspace] = run_forward(q, k, v, W, beta, shape, tensor_cores);
    return {out, bucket_totals_from_workspace(workspace, q, W)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward(
    const at::Tensor& grad_o, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& W, const at::Tensor& beta, const at::Tensor& bucket_totals) {
    const race::ForwardShape shape = validate_inputs(q, k, v, W, beta);
    check_token_tensor(grad_o, "grad_o", q);
    const int64_t slice = (int64_t{1} << W.size(1)) * (q.size(3) + 1);
    TORCH_CHECK(bucket_totals.device() == q.device(),
                "bucket_totals must be on the same device as q");
    TORCH_CHECK(bucket_totals.scalar_type() == at::kFloat, "bucket_totals must be float32, got ",
                bucket_totals.scalar_type());
    TORCH_CHECK(bucket_totals.is_contiguous(), "bucket_totals must be contiguous");
    const std::vector<int64_t> totals_shape = {q.size(0), q.size(1), W.size(0), slice};
    TORCH_CHECK(bucket_totals.sizes() == at::IntArrayRef(totals_shape),
                "bucket_totals must be [B, H, L, R * (d + 1)] = ", at::IntArrayRef(totals_shape),
                ", got ", bucket_totals.sizes());

    const c10::cuda::CUDAGuard device_guard(q.device());
    at::Tensor grad_q = at::empty_like(q);
    at::Tensor grad_k = at::empty_like(k);
    at::Tensor grad_v = at::empty_like(v);
    at::Tensor grad_beta = at::zeros({}, q.options().dtype(at::kFloat));
    if (shape.seq_len == 0 || shape.batch_heads == 0) return {grad_q, grad_k, grad_v, grad_beta};

    const at::Tensor beta_device = beta.to(q.device(), at::kFloat).contiguous();
    // The caching allocator aligns every block to at least 256 bytes, which
    // covers the 16 bytes race_backward requires.
    at::Tensor workspace = at::empty({static_cast<int64_t>(race::backward_workspace_floats(shape))},
                                     q.options().dtype(at::kFloat));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(race::race_backward(
        bf16_ptr(grad_o), bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), W.data_ptr<float>(),
        beta_device.data_ptr<float>(), bucket_totals.data_ptr<float>(), workspace.data_ptr<float>(),
        bf16_ptr(grad_q), bf16_ptr(grad_k), bf16_ptr(grad_v), grad_beta.data_ptr<float>(), shape,
        stream));
    return {grad_q, grad_k, grad_v, grad_beta};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_debug(const at::Tensor& q,
                                                             const at::Tensor& k,
                                                             const at::Tensor& v,
                                                             const at::Tensor& W,
                                                             const at::Tensor& beta,
                                                             bool tensor_cores) {
    const race::ForwardShape shape = validate_inputs(q, k, v, W, beta, tensor_cores);
    auto [out, workspace] = run_forward(q, k, v, W, beta, shape, tensor_cores);

    const int64_t batch = q.size(0), heads = q.size(1), head_dim = q.size(3);
    const int64_t num_tables = W.size(0), corners = int64_t{1} << W.size(1);
    const auto float_options = q.options().dtype(at::kFloat);
    if (!workspace.defined()) {
        return {out, at::zeros({batch, heads, num_tables, corners}, float_options),
                at::zeros({batch, heads, num_tables, corners, head_dim}, float_options)};
    }
    // Tile 0 of the [tiles, BH, L, R * (d + 1)] workspace: B row-major, then A.
    const int64_t slice = corners * (head_dim + 1);
    const at::Tensor totals = workspace.narrow(0, 0, batch * heads * num_tables * slice)
                                  .view({batch, heads, num_tables, slice});
    at::Tensor bucket_values = totals.narrow(-1, 0, corners * head_dim)
                                   .reshape({batch, heads, num_tables, corners, head_dim})
                                   .clone();
    at::Tensor bucket_mass = totals.narrow(-1, corners * head_dim, corners).clone();
    return {out, bucket_mass, bucket_values};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Non-causal RACE Attention forward and backward (bf16 CUDA kernels)";
    m.def("forward", &forward, "RACE forward: o = race(q, k, v, W, beta)", pybind11::arg("q"),
          pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("W"), pybind11::arg("beta"),
          pybind11::arg("tensor_cores") = false);
    m.def("forward_debug", &forward_debug,
          "RACE forward that also returns the reduced bucket sums (o, A, B)", pybind11::arg("q"),
          pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("W"), pybind11::arg("beta"),
          pybind11::arg("tensor_cores") = false);
    m.def("forward_train", &forward_train,
          "RACE forward that also returns what backward needs: (o, bucket_totals)",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("W"),
          pybind11::arg("beta"), pybind11::arg("tensor_cores") = false);
    m.def("backward", &backward, "RACE backward: (dq, dk, dv, dbeta)", pybind11::arg("grad_o"),
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("W"),
          pybind11::arg("beta"), pybind11::arg("bucket_totals"));
}
