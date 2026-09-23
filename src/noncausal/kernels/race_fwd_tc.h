// Host launchers for the tensor-core variant of the non-causal RACE forward.
//
// Same math, tensor layouts, workspace layout and launch sequence as
// race_fwd.h; the bucket build and the query pass are replaced:
//   race_bucket_build_tc -> per-tile partial A, B (same workspace layout)
//   race_bucket_reduce   -> unchanged, from race_fwd.h
//   race_query_tc        -> O = (sum_l phi_q B) / (sum_l phi_q A), bf16
// The three GEMM-shaped stages (the projection X W^T, the bucket build
// Phi^T V and the query mixing Phi_Q B) run on bf16 tensor cores with fp32
// accumulation. Their precision is described at the top of race_fwd_tc.cu:
// the corner probabilities are rounded to bf16, so results differ from the
// fp32 path by more than fp32 rounding (tests/numerics.py derives the bound).
// Results are still bitwise reproducible: no atomics, fixed summation orders.
//
// Requirements beyond race_fwd.h:
//   - q, k, v and out must be 16-byte aligned (rows are copied with 16-byte
//     cp.async and the output is stored with 16-byte vectors); the launchers
//     return cudaErrorInvalidValue otherwise,
//   - an sm_80 or newer GPU (bf16 tensor cores and cp.async).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "race_fwd.h"

namespace race {

constexpr size_t kTensorCoreAlignmentBytes = 16;

// Same contract as race_bucket_build: writes one partial A/B slice per
// (tile, head, table) to the workspace.
cudaError_t race_bucket_build_tc(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                 const float* planes, const float* beta, float* workspace,
                                 const ForwardShape& shape, cudaStream_t stream);

// Same contract as race_query: reads the reduced totals from workspace tile 0.
cudaError_t race_query_tc(const __nv_bfloat16* q, const float* planes, const float* beta,
                          const float* workspace, __nv_bfloat16* out, const ForwardShape& shape,
                          cudaStream_t stream);

// race_bucket_build_tc, race_bucket_reduce and race_query_tc in order.
cudaError_t race_forward_tc(const __nv_bfloat16* q, const __nv_bfloat16* k,
                            const __nv_bfloat16* v, const float* planes, const float* beta,
                            float* workspace, __nv_bfloat16* out, const ForwardShape& shape,
                            cudaStream_t stream);

}  // namespace race
