// Host launchers for the non-causal RACE Attention backward pass.
//
// Given the forward's reduced bucket sums (workspace tile 0 of race_forward,
// [BH, L, R * (D + 1)] fp32) and dO, computes dq, dk, dv (bf16) and the
// scalar d beta (fp32, summed over every batch and head). W is fixed, so there
// is no dW. The derivation is in BACKWARD_DERIVATION.md.
//
// Tensor layouts (all row-major and contiguous), in addition to race_fwd.h:
//   grad_out, grad_q, grad_k, grad_v : [BH, N, D] bf16
//   bucket_totals : [BH, L, R * (D + 1)] fp32, the forward's reduced A and B
//                   in the workspace slice layout (B rows, then A)
//   grad_beta     : one fp32 value in device memory
//   workspace     : backward_workspace_floats(shape) fp32, carved as
//     grad buckets  [num_build_tiles, BH, L, R * (D + 1)]  (dB rows, then dA),
//                   reduced in place so tile 0 holds the totals
//     token weights [BH, N] float2 = (1 / Den_i, dDen_i) per query token
//     beta partials [2, BH, num_query_tiles]: query side, then key side
//
// The backward is five steps on one stream:
//   race_query_grad        -> dq, token weights, query-side beta partials
//   race_bucket_grad_build -> per-tile partial dA, dB (the forward's bucket
//                             build with per-token weights)
//   race_bucket_reduce     -> deterministic tree over tiles (race_fwd.h)
//   race_key_grad          -> dk, dv, key-side beta partials
//   race_beta_grad_reduce  -> d beta
// Results are bitwise reproducible: no atomics, fixed summation orders.
//
// Supported shapes and alignment are those of race_fwd.h (is_supported).
// Each launcher returns the first CUDA error it hits (cudaErrorInvalidValue
// for unsupported shapes) and never synchronizes.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "race_fwd.h"

namespace race {

// Tiles of kQueryTileTokens tokens used by both per-token backward kernels.
int num_query_tiles(const ForwardShape& shape);
// Floats in each region of the backward workspace, and in the whole buffer.
size_t token_weight_floats(const ForwardShape& shape);  // 2 * BH * N
size_t beta_partial_floats(const ForwardShape& shape);  // 2 * BH * num_query_tiles
size_t backward_workspace_floats(const ForwardShape& shape);

// Query side: per query token, with g = dO_i, y_l[r] = B_l[r] . g,
// Den = sum phi A, gNum = sum phi y (sums over l and r):
//   token_weights[i] = (1 / Den, -gNum / Den^2)
//   grad_q[i]        = sum_l sum_t beta h_t (1 - u_t^2) w_{l,t}
// where h_t = sum_r phi[r] dphi[r] c_{r,t} and dphi = y / Den + A dDen.
// Writes one beta partial per (bh, query tile) to beta_partials.
cudaError_t race_query_grad(const __nv_bfloat16* q, const __nv_bfloat16* grad_out,
                            const float* planes, const float* beta, const float* bucket_totals,
                            __nv_bfloat16* grad_q, float2* token_weights, float* beta_partials,
                            const ForwardShape& shape, cudaStream_t stream);

// dB_l[r] = sum_i phi_l(q_i)[r] dO_i / Den_i and dA_l[r] = sum_i phi_l(q_i)[r] dDen_i,
// one partial per build tile, in the forward workspace layout.
cudaError_t race_bucket_grad_build(const __nv_bfloat16* q, const __nv_bfloat16* grad_out,
                                   const float2* token_weights, const float* planes,
                                   const float* beta, float* grad_workspace,
                                   const ForwardShape& shape, cudaStream_t stream);

// Key side: per key token j, with grad_totals the reduced dA, dB (tile 0):
//   grad_v[j] = sum_l sum_r phi_l(k_j)[r] dB_l[r]
//   grad_k[j] = sum_l sum_t beta h_t (1 - u_t^2) w_{l,t},  dphi = dB_l[r] . v_j + dA_l[r]
// Writes one beta partial per (bh, key tile) to beta_partials.
cudaError_t race_key_grad(const __nv_bfloat16* k, const __nv_bfloat16* v, const float* planes,
                          const float* beta, const float* grad_totals, __nv_bfloat16* grad_k,
                          __nv_bfloat16* grad_v, float* beta_partials, const ForwardShape& shape,
                          cudaStream_t stream);

// Sums beta_partial_floats(shape) partials into *grad_beta in a fixed order.
cudaError_t race_beta_grad_reduce(const float* beta_partials, float* grad_beta,
                                  const ForwardShape& shape, cudaStream_t stream);

// All five steps in order. `workspace` holds backward_workspace_floats(shape)
// floats and must be 16-byte aligned; its contents on entry are ignored.
cudaError_t race_backward(const __nv_bfloat16* grad_out, const __nv_bfloat16* q,
                          const __nv_bfloat16* k, const __nv_bfloat16* v, const float* planes,
                          const float* beta, const float* bucket_totals, float* workspace,
                          __nv_bfloat16* grad_q, __nv_bfloat16* grad_k, __nv_bfloat16* grad_v,
                          float* grad_beta, const ForwardShape& shape, cudaStream_t stream);

}  // namespace race
