// Kernel building blocks shared by race_fwd.cu and race_bwd.cu: launch
// constants, shape traits, index helpers, the (D, P) dispatch, and the bucket
// build kernel.
//
// The bucket build is used twice:
//   - forward (kWeighted = false): A_l[r] = sum_j phi_l(k_j)[r] and
//     B_l[r][:] = sum_j phi_l(k_j)[r] v_j over keys,
//   - backward (kWeighted = true): dA_l[r] = sum_i phi_l(q_i)[r] dDen_i and
//     dB_l[r][:] = sum_i phi_l(q_i)[r] dO_i / Den_i over queries, where each
//     token carries a pair (value scale, mass weight) = (1 / Den_i, dDen_i).
// Both write one partial slice per (tile, head, table) in the workspace layout
// of race_fwd.h, which race_bucket_reduce then sums over tiles.
//
// Everything here lives in race::detail and is internal to the two .cu files.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

#include "race_common.cuh"
#include "race_fwd.h"

namespace race {
namespace detail {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
// Tokens per accumulation stage in the bucket build. Each warp stages
// kStageTokens / kWarps tokens, then all threads fold the stage into B.
constexpr int kStageTokens = 16;
constexpr int kStageTokensPerWarp = kStageTokens / kWarps;
constexpr int kMaxTables = 4;
constexpr int kMaxPlanes = 5;
constexpr int kMaxGridY = 65535;
constexpr size_t kDefaultDynamicSmemBytes = 48 * 1024;

static_assert(kStageTokens % kWarps == 0, "stage must split evenly across warps");

template <int D, int P>
struct Dims {
    static_assert(D == 64 || D == 128, "head_dim must be 64 or 128");
    static_assert(P >= 1 && P <= kMaxPlanes, "num_planes must be 1..5");
    static constexpr int kCorners = 1 << P;
    // Elements of a token row held by one lane.
    static constexpr int kVec = D / kWarpSize;
    static constexpr int kSliceFloats = kCorners * (D + 1);
};

// Thread ownership of B during the bucket build. Thread tid owns column
// tid % D and a contiguous run of kRowsPerThread corners starting at
// (tid / D) * kRowsPerThread. Because kThreads is a multiple of D, the column
// is fixed per thread, so each staged token costs one value read (a
// conflict-free row walk across the warp) plus a broadcast read of the
// corner probabilities, and the accumulators never leave registers.
// Only (D = 64, P = 1) has fewer corners than row groups; there half the
// threads idle during accumulation, which is negligible at R = 2.
template <int D, int P>
struct BuildOwnership {
    static constexpr int kRowGroups = kThreads / D;
    static constexpr int kCorners = Dims<D, P>::kCorners;
    static constexpr int kActiveRowGroups = kCorners < kRowGroups ? kCorners : kRowGroups;
    static constexpr int kRowsPerThread = kCorners / kActiveRowGroups;
    static_assert(kThreads % D == 0, "the column-per-thread mapping needs D | kThreads");
};

__host__ __device__ constexpr int round_up4(int n) { return (n + 3) & ~3; }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Offset of the (tile, bh, table) slice inside the workspace.
__host__ __device__ inline size_t workspace_slice_offset(int tile, int bh, int table,
                                                         int batch_heads, int num_tables,
                                                         int slice_floats) {
    const size_t slice_index = (static_cast<size_t>(tile) * batch_heads + bh) * num_tables + table;
    return slice_index * slice_floats;
}

// Offset of the first element of token `token` of head `bh` in a [BH, N, D] tensor.
__device__ inline size_t token_row_offset(int bh, int token, int seq_len, int head_dim) {
    return (static_cast<size_t>(bh) * seq_len + token) * head_dim;
}

template <int D, typename Fn>
cudaError_t dispatch_planes(int num_planes, Fn&& fn) {
    switch (num_planes) {
        case 1: return fn(std::integral_constant<int, D>{}, std::integral_constant<int, 1>{});
        case 2: return fn(std::integral_constant<int, D>{}, std::integral_constant<int, 2>{});
        case 3: return fn(std::integral_constant<int, D>{}, std::integral_constant<int, 3>{});
        case 4: return fn(std::integral_constant<int, D>{}, std::integral_constant<int, 4>{});
        case 5: return fn(std::integral_constant<int, D>{}, std::integral_constant<int, 5>{});
        default: return cudaErrorInvalidValue;
    }
}

// Calls fn(integral_constant<D>, integral_constant<P>) for the runtime shape.
template <typename Fn>
cudaError_t dispatch_shape(const ForwardShape& shape, Fn&& fn) {
    switch (shape.head_dim) {
        case 64: return dispatch_planes<64>(shape.num_planes, fn);
        case 128: return dispatch_planes<128>(shape.num_planes, fn);
        default: return cudaErrorInvalidValue;
    }
}

// Computes phi and copies v for the kStageTokens tokens starting at
// stage_begin. Warp w handles stage slots w, w + kWarps, ...; slots past
// tile_end get phi = 0 and v = 0 so the accumulation needs no masking (and
// never multiplies uninitialized shared memory, which could hold NaN).
// With kWeighted, token j's staged value row is v_j * weights[j].x and its
// contribution to the mass is phi * weights[j].y.
template <int D, int P, bool kWeighted>
__device__ __forceinline__ void stage_tokens(const bf16* __restrict__ key_rows,
                                             const bf16* __restrict__ value_rows,
                                             const float2* __restrict__ token_weights,
                                             const float* planes_s, float beta, int stage_begin,
                                             int tile_end, float (*probs_s)[Dims<D, P>::kCorners],
                                             float (*values_s)[D], int warp, int lane,
                                             float& mass) {
    constexpr int kVec = Dims<D, P>::kVec;
    constexpr int kCorners = Dims<D, P>::kCorners;

    // Issue every global load of the stage before the dependent math. Values
    // go straight to shared memory so their registers die before the
    // projections, which keeps the P = 5, D = 128 instantiation spill-free.
    float keys[kStageTokensPerWarp][kVec];
    float values[kStageTokensPerWarp][kVec];
    float mass_weight[kStageTokensPerWarp];
#pragma unroll
    for (int i = 0; i < kStageTokensPerWarp; ++i) {
        const int token = stage_begin + warp + i * kWarps;
#pragma unroll
        for (int j = 0; j < kVec; ++j) keys[i][j] = values[i][j] = 0.0f;
        mass_weight[i] = 0.0f;
        if (token < tile_end) {
            const size_t offset = static_cast<size_t>(token) * D + lane * kVec;
            load_bf16_vec<kVec>(key_rows + offset, keys[i]);
            load_bf16_vec<kVec>(value_rows + offset, values[i]);
            if constexpr (kWeighted) {
                // Every lane reads the same 8 bytes: one broadcast transaction.
                const float2 weight = token_weights[token];
#pragma unroll
                for (int j = 0; j < kVec; ++j) values[i][j] *= weight.x;
                mass_weight[i] = weight.y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kStageTokensPerWarp; ++i) {
        store_f32_vec<kVec>(&values_s[warp + i * kWarps][lane * kVec], values[i]);
    }

#pragma unroll
    for (int i = 0; i < kStageTokensPerWarp; ++i) {
        const int slot = warp + i * kWarps;
        const bool valid = stage_begin + slot < tile_end;  // warp-uniform
        const float phi_all = corner_probability<D, P>(keys[i], planes_s, beta, lane);
        const float phi = valid ? phi_all : 0.0f;
        if (lane < kCorners) {
            probs_s[slot][lane] = phi;
            if constexpr (kWeighted) {
                mass = fmaf(phi, mass_weight[i], mass);
            } else {
                mass += phi;
            }
        }
    }
}

// Folds one staged batch into this thread's B accumulators:
// acc[i] += phi[slot][first_row + i] * v[slot][column] for every slot.
template <int D, int P>
__device__ __forceinline__ void accumulate_stage(
    const float (*probs_s)[Dims<D, P>::kCorners], const float (*values_s)[D], int column,
    int first_row, float (&acc)[BuildOwnership<D, P>::kRowsPerThread]) {
    constexpr int kRows = BuildOwnership<D, P>::kRowsPerThread;
#pragma unroll
    for (int slot = 0; slot < kStageTokens; ++slot) {
        const float value = values_s[slot][column];
        float phi[kRows];
        load_f32_vec<kRows>(&probs_s[slot][first_row], phi);
#pragma unroll
        for (int i = 0; i < kRows; ++i) acc[i] = fmaf(phi[i], value, acc[i]);
    }
}

// Grid: x = num_tiles * num_tables with the table index fastest, so the L
// CTAs that read the same key tile are scheduled together and all but the
// first find K and V in L2. y = batch_heads.
// token_weights is [BH, N] and only read when kWeighted (nullptr otherwise).
template <int D, int P, bool kWeighted>
__global__ void __launch_bounds__(kThreads)
    bucket_build_kernel(const bf16* __restrict__ keys, const bf16* __restrict__ values,
                        const float2* __restrict__ token_weights,
                        const float* __restrict__ planes, const float* __restrict__ beta_ptr,
                        float* __restrict__ workspace, int seq_len, int batch_heads,
                        int num_tables) {
    using Own = BuildOwnership<D, P>;
    constexpr int kCorners = Dims<D, P>::kCorners;
    constexpr int kRows = Own::kRowsPerThread;

    // Double-buffered stage: stage s + 1 is written while stage s is read,
    // so one barrier per stage suffices (see the loop below).
    __shared__ __align__(16) float planes_s[P * D];
    __shared__ __align__(16) float probs_s[2][kStageTokens][kCorners];
    __shared__ __align__(16) float values_s[2][kStageTokens][D];
    __shared__ float warp_mass_s[kWarps][kWarpSize];

    const int table = blockIdx.x % num_tables;
    const int tile = blockIdx.x / num_tables;
    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    const int tile_begin = tile * kBuildTileTokens;
    const int tile_end = min(seq_len, tile_begin + kBuildTileTokens);
    const int num_stages = (tile_end - tile_begin + kStageTokens - 1) / kStageTokens;

    const float* table_planes = planes + static_cast<size_t>(table) * P * D;
    for (int i = threadIdx.x; i < P * D; i += kThreads) planes_s[i] = table_planes[i];
    const float beta = *beta_ptr;

    const bf16* key_rows = keys + token_row_offset(bh, 0, seq_len, D);
    const bf16* value_rows = values + token_row_offset(bh, 0, seq_len, D);
    const float2* head_weights =
        kWeighted ? token_weights + static_cast<size_t>(bh) * seq_len : nullptr;

    const int column = threadIdx.x % D;
    const int row_group = threadIdx.x / D;
    const bool owns_rows = row_group < Own::kActiveRowGroups;
    const int first_row = row_group * kRows;

    float acc[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) acc[i] = 0.0f;
    float mass = 0.0f;  // lane r: sum of (weighted) phi[r] over the tokens this warp staged

    __syncthreads();  // planes_s is ready
    stage_tokens<D, P, kWeighted>(key_rows, value_rows, head_weights, planes_s, beta, tile_begin,
                                  tile_end, probs_s[0], values_s[0], warp, lane, mass);
    __syncthreads();

    // Invariant at the top of iteration s: buffer s & 1 holds stage s and no
    // thread still reads buffer (s + 1) & 1 (its last reader was iteration
    // s - 1, which ended with a barrier).
    for (int s = 0; s < num_stages; ++s) {
        const int buffer = s & 1;
        if (s + 1 < num_stages) {
            stage_tokens<D, P, kWeighted>(key_rows, value_rows, head_weights, planes_s, beta,
                                          tile_begin + (s + 1) * kStageTokens, tile_end,
                                          probs_s[buffer ^ 1], values_s[buffer ^ 1], warp, lane,
                                          mass);
        }
        if (owns_rows) {
            accumulate_stage<D, P>(probs_s[buffer], values_s[buffer], column, first_row, acc);
        }
        __syncthreads();
    }

    float* slice = workspace + workspace_slice_offset(tile, bh, table, batch_heads, num_tables,
                                                      Dims<D, P>::kSliceFloats);
    if (owns_rows) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) slice[(first_row + i) * D + column] = acc[i];
    }

    // A: combine the per-warp masses in a fixed warp order.
    if (lane < kCorners) warp_mass_s[warp][lane] = mass;
    __syncthreads();
    if (threadIdx.x < kCorners) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) total += warp_mass_s[w][threadIdx.x];
        slice[kCorners * D + threadIdx.x] = total;
    }
}

}  // namespace detail
}  // namespace race
