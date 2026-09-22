// Host launchers for the non-causal RACE Attention forward pass.
//
// Tensor layouts (all row-major and contiguous):
//   q, k, v, out : [BH, N, D] bf16, where BH = batch * heads
//   planes       : [L, P, D] fp32, the fixed random hash planes W
//   beta         : one fp32 value in device memory (read by the kernels, so a
//                  trainable beta never forces a host sync)
//   workspace    : [num_tiles, BH, L, R * (D + 1)] fp32, see below
//
// Workspace slice: for one (tile, bh, table) the R * (D + 1) floats hold
//   B[r][c] at r * D + c        (sum of phi[r] * v[c] over the tile's keys)
//   A[r]    at R * D + r        (sum of phi[r] over the tile's keys)
// After race_bucket_reduce, tile 0 holds the totals over all N keys.
//
// The forward pass is three launches on one stream:
//   race_bucket_build  -> per-tile partial A, B
//   race_bucket_reduce -> deterministic pairwise tree over tiles, in place
//   race_query         -> O = (sum_l phi_q B) / (sum_l phi_q A), bf16
// Results are bitwise reproducible: no atomics, fixed summation orders.
//
// Supported: D in {64, 128}, P in {1..5}, L in {1..4}, 1 <= N <= 2^31 - 2049,
// 1 <= BH <= 65535.
// Pointers to bf16 rows must be 8-byte aligned (16 recommended).
// Each launcher returns the first CUDA error it hits (cudaErrorInvalidValue
// for unsupported shapes) and never synchronizes.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace race {

struct ForwardShape {
    int batch_heads;  // BH
    int seq_len;      // N (keys and queries)
    int head_dim;     // D
    int num_planes;   // P, R = 2^P corners
    int num_tables;   // L
};

// Tokens per CTA in the bucket build; one workspace tile per this many keys.
constexpr int kBuildTileTokens = 2048;
// Tokens per CTA in the query pass.
constexpr int kQueryTileTokens = 1024;

bool is_supported(const ForwardShape& shape);
int num_build_tiles(const ForwardShape& shape);
// Floats in one (tile, bh, table) slice: R * (D + 1).
size_t workspace_slice_floats(const ForwardShape& shape);
// Floats in the whole workspace.
size_t workspace_floats(const ForwardShape& shape);

cudaError_t race_bucket_build(const __nv_bfloat16* k, const __nv_bfloat16* v, const float* planes,
                              const float* beta, float* workspace, const ForwardShape& shape,
                              cudaStream_t stream);

cudaError_t race_bucket_reduce(float* workspace, const ForwardShape& shape, cudaStream_t stream);

cudaError_t race_query(const __nv_bfloat16* q, const float* planes, const float* beta,
                       const float* workspace, __nv_bfloat16* out, const ForwardShape& shape,
                       cudaStream_t stream);

// All three launches in order.
cudaError_t race_forward(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                         const float* planes, const float* beta, float* workspace,
                         __nv_bfloat16* out, const ForwardShape& shape, cudaStream_t stream);

}  // namespace race
