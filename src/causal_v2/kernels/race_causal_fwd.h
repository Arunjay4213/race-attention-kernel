// Host launchers for the chunk-parallel causal RACE Attention forward, v2a
// (fp32 CUDA cores). Design: docs/causal_v2_design.md.
//
// Math (paper Algorithm 1 made causal, global normalization):
//   O_i = (Phi_Q,i . B_i) / (Phi_Q,i . A_i),
//   B_i = sum_{j<=i} Phi_K,j (x) v_j,   A_i = sum_{j<=i} Phi_K,j,
// where Phi(x) in R^S, S = L * R, holds the L tables' corner probabilities.
//
// Tensor layouts (all row-major and contiguous):
//   q, k, v, out : [BH, T, D] bf16, BH = batch * heads
//   planes       : [L, P, D] fp32, the fixed random hash planes W
//   beta         : one fp32 value in device memory (read by the kernels, so a
//                  trainable beta never forces a host sync)
//   workspace    : [tiles, BH, L, R * (D + 1)] fp32, the non-causal layout:
//                  for one (tile, bh, table), B[r][c] at r * D + c and A[r] at
//                  R * D + r. After race_causal_tile_sums a slice holds that
//                  tile's own sums; after race_causal_tile_scan it holds the
//                  exclusive prefix, i.e. the sums over all earlier tiles.
//   final_state  : [BH, L, R * (D + 1)] fp32, same slice layout, the sums over
//                  all T tokens (optional output of the scan).
//
// The forward pass is three launches on one stream:
//   race_causal_tile_sums  (K1) -> per-tile A, B (the non-causal bucket build)
//   race_causal_tile_scan  (K2) -> exclusive prefix over tiles, in place
//   race_causal_output     (K3) -> one CTA per tile walks sub-chunks of C
//                                  tokens with the running state on chip
// Results are bitwise reproducible for a fixed tile length: no atomics and
// every sum has a fixed order. A different tile length changes the order.
//
// Tile length (T_blk): a positive multiple of the sub-chunk length C. Pass the
// value from race_causal_select_tile_tokens (or a test override) to all three
// launchers; they must agree.
//
// Supported: D in {64, 128}, P in {1..5}, L in {1..4}, 1 <= T <= 2^31 - 4097,
// 1 <= BH <= 65535, and an output-pass shared-memory footprint within the
// device's opt-in limit (race_causal_output_fits).
// q, k, v and out must be 16-byte aligned (the output pass uses 16-byte loads).
// Each launcher returns the first CUDA error it hits (cudaErrorInvalidValue
// for unsupported shapes or tile lengths) and never synchronizes.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace race {
namespace causal {

struct CausalShape {
    int batch_heads;  // BH
    int seq_len;      // T
    int head_dim;     // D
    int num_planes;   // P, R = 2^P corners
    int num_tables;   // L
};

// Largest tile length the launcher picks by itself.
constexpr int kMaxAutoTileTokens = 4096;

bool is_supported(const CausalShape& shape);

// Sub-chunk length C of the output pass for (D, P): 64 or 32.
int sub_chunk_tokens(int head_dim, int num_planes);

// Dynamic shared memory of the output pass in bytes (0 if unsupported).
size_t output_pass_smem_bytes(const CausalShape& shape);

// Whether the output pass fits the current device's per-CTA opt-in shared
// memory limit; *limit_bytes receives that limit.
cudaError_t race_causal_output_fits(const CausalShape& shape, bool* fits, int* limit_bytes);

// Tile length for this shape on the current device: the default for the
// state size (plan section 3.3), shrunk for short sequences so the output
// pass still has at least ~4 waves of CTAs, never below C.
cudaError_t race_causal_select_tile_tokens(const CausalShape& shape, int* tile_tokens);

bool is_valid_tile_tokens(const CausalShape& shape, int tile_tokens);
int num_tiles(const CausalShape& shape, int tile_tokens);
// Floats in one (tile, bh, table) slice: R * (D + 1).
size_t slice_floats(const CausalShape& shape);
// Floats in the whole workspace for this tile length.
size_t workspace_floats(const CausalShape& shape, int tile_tokens);
// Floats in the final state: BH * L * R * (D + 1).
size_t final_state_floats(const CausalShape& shape);

// K1: per-tile sums of Phi_K and Phi_K (x) v.
cudaError_t race_causal_tile_sums(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                  const float* planes, const float* beta, float* workspace,
                                  const CausalShape& shape, int tile_tokens, cudaStream_t stream);

// K2: in-place exclusive scan over tiles; writes the inclusive total to
// final_state unless it is nullptr.
cudaError_t race_causal_tile_scan(float* workspace, float* final_state, const CausalShape& shape,
                                  int tile_tokens, cudaStream_t stream);

// K3: the outputs, reading the scanned workspace.
cudaError_t race_causal_output(const __nv_bfloat16* q, const __nv_bfloat16* k,
                               const __nv_bfloat16* v, const float* planes, const float* beta,
                               const float* workspace, __nv_bfloat16* out,
                               const CausalShape& shape, int tile_tokens, cudaStream_t stream);

// All three launches in order.
cudaError_t race_causal_forward(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                const __nv_bfloat16* v, const float* planes, const float* beta,
                                float* workspace, float* final_state, __nv_bfloat16* out,
                                const CausalShape& shape, int tile_tokens, cudaStream_t stream);

}  // namespace causal
}  // namespace race
