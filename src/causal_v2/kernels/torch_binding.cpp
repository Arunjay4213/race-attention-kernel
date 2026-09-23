// PyTorch bindings for the causal RACE v2 forward kernels (race_causal_fwd.cu
// for v2a, race_causal_fwd_tc.cu for v2b).
//
//   forward(q, k, v, W, beta, tile_tokens=0, tensor_cores=False) -> o
//   forward_debug(q, k, v, W, beta, tile_tokens=0, tensor_cores=False)
//       -> (o, prefix_A, prefix_B, final_A, final_B, tile_tokens)
//   sub_chunk_tokens(head_dim, num_planes, tensor_cores=False) -> C
//   select_tile_tokens(batch_heads, seq_len, head_dim, num_planes, num_tables,
//                      tensor_cores=False) -> T_blk
//   output_pass_smem_bytes(head_dim, num_planes, num_tables, tensor_cores=False) -> bytes
//   fits_on_device(head_dim, num_planes, num_tables, tensor_cores=False) -> bool
//   precise_carry() -> bool, whether v2b was built with the hi/lo carry
//
// tensor_cores=False runs v2a (fp32 CUDA cores), True runs v2b (bf16 tensor
// cores, fp32 accumulation). The variant sets C and therefore the valid and
// automatic tile lengths, so pass the same flag to the queries and the forward.
//
// q, k, v: [B, H, T, d] bf16 CUDA, contiguous, 16-byte aligned, identical
// shapes. W: [L, P, d] fp32 on the same device. beta: one-element floating
// tensor on any device (a CUDA fp32 beta is read by the kernels in place).
// tile_tokens: 0 picks the tile length automatically; a positive value (a
// multiple of sub_chunk_tokens) forces it, which the tests use.
// o: [B, H, T, d] bf16. prefix_A: [B, H, tiles, L, R] and prefix_B:
// [B, H, tiles, L, R, d] fp32, the exclusive prefix state at each tile start
// (the workspace after the scan). final_A: [B, H, L, R], final_B:
// [B, H, L, R, d], the state after all T tokens.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "race_causal_fwd.h"

namespace {

// The output pass reads q, k, v with 16-byte vector loads.
constexpr int64_t kRequiredAlignmentBytes = 16;

void check_token_tensor(const at::Tensor& t, const char* name, const at::Tensor& like) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", t.scalar_type());
    TORCH_CHECK(t.dim() == 4, name, " must be [B, H, T, d], got ", t.sizes());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.sizes() == like.sizes(), name, " shape ", t.sizes(), " differs from q ", like.sizes());
    TORCH_CHECK(t.device() == like.device(), name, " must be on the same device as q");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % kRequiredAlignmentBytes == 0, name,
                " data must be 16-byte aligned (clone the tensor if it is a view)");
}

race::causal::CausalShape make_shape(int64_t batch_heads, int64_t seq_len, int64_t head_dim,
                                     int64_t num_planes, int64_t num_tables) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_planes >= 1 && num_planes <= 5, "num_planes P must be in 1..5, got ", num_planes);
    TORCH_CHECK(num_tables >= 1 && num_tables <= 4, "num_tables L must be in 1..4, got ", num_tables);
    TORCH_CHECK(batch_heads >= 0 && batch_heads <= 65535, "batch * heads must be <= 65535, got ",
                batch_heads);
    TORCH_CHECK(seq_len >= 0 &&
                    seq_len <= std::numeric_limits<int>::max() - race::causal::kMaxAutoTileTokens - 1,
                "sequence length out of range: ", seq_len);
    race::causal::CausalShape shape;
    shape.batch_heads = static_cast<int>(batch_heads);
    shape.seq_len = static_cast<int>(seq_len);
    shape.head_dim = static_cast<int>(head_dim);
    shape.num_planes = static_cast<int>(num_planes);
    shape.num_tables = static_cast<int>(num_tables);
    return shape;
}

race::causal::CausalShape validate_inputs(const at::Tensor& q, const at::Tensor& k,
                                          const at::Tensor& v, const at::Tensor& W,
                                          const at::Tensor& beta) {
    check_token_tensor(q, "q", q);
    check_token_tensor(k, "k", q);
    check_token_tensor(v, "v", q);

    TORCH_CHECK(W.device() == q.device(), "W must be on the same device as q");
    TORCH_CHECK(W.scalar_type() == at::kFloat, "W must be float32, got ", W.scalar_type());
    TORCH_CHECK(W.dim() == 3, "W must be [L, P, d], got ", W.sizes());
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(W.size(2) == q.size(3), "W head_dim ", W.size(2), " differs from q ", q.size(3));
    TORCH_CHECK(beta.numel() == 1, "beta must have exactly one element");
    TORCH_CHECK(at::isFloatingType(beta.scalar_type()), "beta must be a floating tensor");

    return make_shape(q.size(0) * q.size(1), q.size(2), q.size(3), W.size(1), W.size(0));
}

race::causal::Variant to_variant(bool tensor_cores) {
    return tensor_cores ? race::causal::Variant::kTensorCores : race::causal::Variant::kCudaCores;
}

// Rejects shapes whose output pass does not fit this GPU's shared memory,
// with a message that says why (config B at L = 4 needs A100 or H100).
void check_fits_device(const race::causal::CausalShape& shape, race::causal::Variant variant) {
    bool fits = false;
    int limit_bytes = 0;
    C10_CUDA_CHECK(race::causal::race_causal_output_fits(shape, &fits, &limit_bytes, variant));
    TORCH_CHECK(fits, "the causal output pass (",
                variant == race::causal::Variant::kTensorCores ? "v2b" : "v2a", ") needs ",
                race::causal::output_pass_smem_bytes(shape, variant),
                " bytes of shared memory per block for d=", shape.head_dim, ", P=", shape.num_planes,
                ", L=", shape.num_tables, "; this GPU allows ", limit_bytes);
}

int resolve_tile_tokens(const race::causal::CausalShape& shape, int64_t requested,
                        race::causal::Variant variant) {
    if (requested == 0) {
        int tile_tokens = 0;
        C10_CUDA_CHECK(race::causal::race_causal_select_tile_tokens(shape, &tile_tokens, variant));
        return tile_tokens;
    }
    TORCH_CHECK(requested > 0 && requested <= std::numeric_limits<int>::max() &&
                    race::causal::is_valid_tile_tokens(shape, static_cast<int>(requested), variant),
                "tile_tokens must be a positive multiple of ",
                race::causal::sub_chunk_tokens(shape.head_dim, shape.num_planes, variant), ", got ",
                requested);
    return static_cast<int>(requested);
}

const __nv_bfloat16* bf16_ptr(const at::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

__nv_bfloat16* bf16_ptr(at::Tensor& t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

struct ForwardResult {
    at::Tensor out;
    at::Tensor workspace;    // [tiles, BH, L, R * (d + 1)], exclusive prefixes; undefined if empty
    at::Tensor final_state;  // [BH, L, R * (d + 1)]; undefined unless requested and non-empty
    int tile_tokens = 0;
};

ForwardResult run_forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                          const at::Tensor& W, const at::Tensor& beta,
                          const race::causal::CausalShape& shape, int64_t requested_tile_tokens,
                          bool want_final_state, race::causal::Variant variant) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    ForwardResult result;
    result.out = at::empty_like(q);
    if (shape.seq_len == 0 || shape.batch_heads == 0) return result;

    check_fits_device(shape, variant);
    result.tile_tokens = resolve_tile_tokens(shape, requested_tile_tokens, variant);

    const at::Tensor beta_device = beta.to(q.device(), at::kFloat).contiguous();
    const auto float_options = q.options().dtype(at::kFloat);
    result.workspace = at::empty(
        {static_cast<int64_t>(race::causal::workspace_floats(shape, result.tile_tokens))},
        float_options);
    float* final_state = nullptr;
    if (want_final_state) {
        result.final_state = at::empty(
            {static_cast<int64_t>(race::causal::final_state_floats(shape))}, float_options);
        final_state = result.final_state.data_ptr<float>();
    }
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(race::causal::race_causal_forward(
        bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), W.data_ptr<float>(), beta_device.data_ptr<float>(),
        result.workspace.data_ptr<float>(), final_state, bf16_ptr(result.out), shape,
        result.tile_tokens, stream, variant));
    return result;
}

at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   const at::Tensor& W, const at::Tensor& beta, int64_t tile_tokens,
                   bool tensor_cores) {
    const race::causal::CausalShape shape = validate_inputs(q, k, v, W, beta);
    return run_forward(q, k, v, W, beta, shape, tile_tokens, false, to_variant(tensor_cores)).out;
}

// Splits [..., L, R * (d + 1)] slices into A [..., L, R] and B [..., L, R, d].
std::tuple<at::Tensor, at::Tensor> split_slices(const at::Tensor& slices, int64_t corners,
                                                int64_t head_dim) {
    std::vector<int64_t> values_shape(slices.sizes().begin(), slices.sizes().end() - 1);
    values_shape.push_back(corners);
    values_shape.push_back(head_dim);
    at::Tensor values = slices.narrow(-1, 0, corners * head_dim).reshape(values_shape).clone();
    at::Tensor mass = slices.narrow(-1, corners * head_dim, corners).clone();
    return {mass, values};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t> forward_debug(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& W,
    const at::Tensor& beta, int64_t tile_tokens, bool tensor_cores) {
    const race::causal::CausalShape shape = validate_inputs(q, k, v, W, beta);
    ForwardResult result =
        run_forward(q, k, v, W, beta, shape, tile_tokens, true, to_variant(tensor_cores));

    const int64_t batch = q.size(0), heads = q.size(1), head_dim = q.size(3);
    const int64_t num_tables = W.size(0), corners = int64_t{1} << W.size(1);
    const int64_t slice = corners * (head_dim + 1);
    const auto float_options = q.options().dtype(at::kFloat);
    if (!result.workspace.defined()) {
        return {result.out,
                at::zeros({batch, heads, 0, num_tables, corners}, float_options),
                at::zeros({batch, heads, 0, num_tables, corners, head_dim}, float_options),
                at::zeros({batch, heads, num_tables, corners}, float_options),
                at::zeros({batch, heads, num_tables, corners, head_dim}, float_options),
                0};
    }
    const int64_t tiles = race::causal::num_tiles(shape, result.tile_tokens);
    // Workspace [tiles, B * H, L, slice] -> [B, H, tiles, L, slice].
    const at::Tensor prefixes =
        result.workspace.view({tiles, batch, heads, num_tables, slice}).permute({1, 2, 0, 3, 4});
    auto [prefix_mass, prefix_values] = split_slices(prefixes, corners, head_dim);
    auto [final_mass, final_values] = split_slices(
        result.final_state.view({batch, heads, num_tables, slice}), corners, head_dim);
    return {result.out, prefix_mass, prefix_values, final_mass, final_values, result.tile_tokens};
}

int64_t sub_chunk_tokens(int64_t head_dim, int64_t num_planes, bool tensor_cores) {
    make_shape(1, 1, head_dim, num_planes, 1);
    return race::causal::sub_chunk_tokens(static_cast<int>(head_dim), static_cast<int>(num_planes),
                                          to_variant(tensor_cores));
}

int64_t select_tile_tokens(int64_t batch_heads, int64_t seq_len, int64_t head_dim,
                           int64_t num_planes, int64_t num_tables, bool tensor_cores) {
    const race::causal::CausalShape shape =
        make_shape(batch_heads, seq_len, head_dim, num_planes, num_tables);
    TORCH_CHECK(shape.seq_len > 0 && shape.batch_heads > 0, "empty shape has no tiles");
    check_fits_device(shape, to_variant(tensor_cores));
    return resolve_tile_tokens(shape, 0, to_variant(tensor_cores));
}

int64_t output_pass_smem_bytes(int64_t head_dim, int64_t num_planes, int64_t num_tables,
                               bool tensor_cores) {
    return static_cast<int64_t>(race::causal::output_pass_smem_bytes(
        make_shape(1, 1, head_dim, num_planes, num_tables), to_variant(tensor_cores)));
}

bool fits_on_device(int64_t head_dim, int64_t num_planes, int64_t num_tables, bool tensor_cores) {
    bool fits = false;
    int limit_bytes = 0;
    C10_CUDA_CHECK(race::causal::race_causal_output_fits(
        make_shape(1, 1, head_dim, num_planes, num_tables), &fits, &limit_bytes,
        to_variant(tensor_cores)));
    return fits;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.doc() = "Causal RACE Attention forward, chunk-parallel v2a and v2b (bf16 CUDA kernels)";
    m.def("forward", &forward, "Causal RACE forward: o = race_causal(q, k, v, W, beta)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("W"), py::arg("beta"),
          py::arg("tile_tokens") = 0, py::arg("tensor_cores") = false);
    m.def("forward_debug", &forward_debug,
          "Causal RACE forward that also returns the per-tile prefix states and the final state: "
          "(o, prefix_A, prefix_B, final_A, final_B, tile_tokens)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("W"), py::arg("beta"),
          py::arg("tile_tokens") = 0, py::arg("tensor_cores") = false);
    m.def("sub_chunk_tokens", &sub_chunk_tokens, "Sub-chunk length C of the output pass",
          py::arg("head_dim"), py::arg("num_planes"), py::arg("tensor_cores") = false);
    m.def("select_tile_tokens", &select_tile_tokens,
          "Tile length the forward picks for this shape on the current device",
          py::arg("batch_heads"), py::arg("seq_len"), py::arg("head_dim"), py::arg("num_planes"),
          py::arg("num_tables"), py::arg("tensor_cores") = false);
    m.def("output_pass_smem_bytes", &output_pass_smem_bytes,
          "Dynamic shared memory of the output pass in bytes", py::arg("head_dim"),
          py::arg("num_planes"), py::arg("num_tables"), py::arg("tensor_cores") = false);
    m.def("fits_on_device", &fits_on_device,
          "Whether the output pass fits the current device's shared memory", py::arg("head_dim"),
          py::arg("num_planes"), py::arg("num_tables"), py::arg("tensor_cores") = false);
    m.def("precise_carry", &race::causal::tensor_cores_precise_carry,
          "Whether v2b was built with the hi/lo carry (RACE_CAUSAL_PRECISE_CARRY)");
}
