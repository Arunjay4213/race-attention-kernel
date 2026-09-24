// Non-causal RACE Attention forward pass: three kernels plus host launchers.
//
// Math (paper Algorithm 1, global normalization), per head and table l:
//   phi_l(x)[r] = prod_t (sigmoid(2 beta u_t) if corner r has +1 on plane t
//                         else sigmoid(-2 beta u_t)),   u = tanh(W_l x)
//   A_l[r] = sum_j phi_l(k_j)[r],   B_l[r][:] = sum_j phi_l(k_j)[r] v_j
//   O_i = (sum_l sum_r phi_l(q_i)[r] B_l[r][:]) / (sum_l sum_r phi_l(q_i)[r] A_l[r])
// The 1/L factors of the paper's Num and Den cancel and are omitted.
//
// Kernels (layouts in race_fwd.h):
//   1. bucket_build_kernel (race_internal.cuh, shared with the backward): one
//      CTA per (key tile, table, head). Streams
//      kBuildTileTokens keys through shared memory in stages of kStageTokens
//      and accumulates B for the tile in registers, then writes one partial
//      A/B slice to the workspace.
//   2. tree_reduce_pass_kernel: fan-in-8 pairwise tree over the tile axis,
//      one launch per level, in place. The summation order depends only on
//      the tile count, so results are bitwise reproducible.
//   3. query_kernel: one CTA per (query tile, head). Loads the final A, B and
//      all planes into shared memory once, then one warp per query token.
//
// Precision: bf16 in and out, fp32 everywhere in between, accurate tanhf and
// expf. No atomics anywhere.
//
// Where this departs from docs/noncausal_design.md, and why:
//   - The build keeps B in registers (each thread owns fixed (r, c) slots for
//     the whole tile) instead of a shared B[R][D + 1]: the plan's staging and
//     slot ownership already make every slot private to one thread, so shared
//     memory would only add a load and a store per FMA. The R * (D + 1)
//     shape survives as the workspace slice (B rows, then A).
//   - The build grid puts the table index fastest in blockIdx.x so the L CTAs
//     of a tile share K/V through L2 instead of reading HBM L times.
//   - Each reduce launch folds 8 tiles with a fixed pairwise tree, so the
//     whole reduction is still one binary tree over tiles but takes
//     ceil(log8(tiles)) launches instead of ceil(log2(tiles)).
//   - beta is read from device memory, so a trainable beta on the GPU never
//     needs a host sync (.item()) per call.
//   - Lane r computes phi[r] directly as a P-term product (lanes in parallel)
//     instead of one thread building all R values with a product tree.
#include "race_fwd.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "race_common.cuh"
#include "race_internal.cuh"

namespace race {
namespace {

using detail::bf16;
using detail::bucket_build_kernel;
using detail::ceil_div;
using detail::Dims;
using detail::dispatch_shape;
using detail::kDefaultDynamicSmemBytes;
using detail::kMaxGridY;
using detail::kMaxPlanes;
using detail::kMaxTables;
using detail::kThreads;
using detail::kWarps;
using detail::round_up4;
using detail::token_row_offset;

constexpr int kReduceFanIn = 8;
constexpr int kMaxReduceBlocks = 1 << 15;
// Token indices are int; tile_begin + tile size must not overflow.
constexpr int kMaxSeqLen =
    std::numeric_limits<int>::max() - std::max(kBuildTileTokens, kQueryTileTokens);

// ---------------------------------------------------------------------------
// Kernel 2: deterministic tree reduce over tiles
// ---------------------------------------------------------------------------

// One level of the tree: for every group of kReduceFanIn tiles spaced
// `stride` apart, starting at a multiple of kReduceFanIn * stride, sums them
// pairwise and writes the result to the group's first tile. Missing tiles
// (past num_tiles) contribute an exact 0, so the result equals a pairwise
// tree over the tiles in index order. Groups are disjoint, so in place is safe.
__global__ void __launch_bounds__(kThreads)
    tree_reduce_pass_kernel(float* __restrict__ workspace, int64_t tile_floats, int num_tiles,
                            int64_t stride, int64_t total_work) {
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         work < total_work; work += step) {
        const int64_t group = work / tile_floats;
        const int64_t element = work - group * tile_floats;
        const int64_t first_tile = group * kReduceFanIn * stride;

        float terms[kReduceFanIn];
#pragma unroll
        for (int i = 0; i < kReduceFanIn; ++i) {
            const int64_t tile = first_tile + i * stride;
            terms[i] = tile < num_tiles ? workspace[tile * tile_floats + element] : 0.0f;
        }
#pragma unroll
        for (int width = 1; width < kReduceFanIn; width *= 2) {
#pragma unroll
            for (int i = 0; i < kReduceFanIn; i += 2 * width) terms[i] += terms[i + width];
        }
        workspace[first_tile * tile_floats + element] = terms[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: query pass
// ---------------------------------------------------------------------------

// Dynamic shared memory of the query kernel, in floats. B needs no padding
// here: it is only read row-wise (lane-contiguous float4/float2), which is
// conflict-free. Regions are rounded to 4 floats to keep vector alignment.
template <int D, int P>
struct QuerySmem {
    static constexpr int kCorners = Dims<D, P>::kCorners;
    __host__ __device__ static int buckets(int) { return 0; }
    __host__ __device__ static int mass(int num_tables) { return num_tables * kCorners * D; }
    __host__ __device__ static int planes(int num_tables) {
        return mass(num_tables) + round_up4(num_tables * kCorners);
    }
    __host__ __device__ static int probs_stride(int num_tables) {
        return round_up4(num_tables * kCorners);
    }
    __host__ __device__ static int probs(int num_tables) {
        return planes(num_tables) + num_tables * P * D;
    }
    __host__ __device__ static int total_floats(int num_tables) {
        return probs(num_tables) + kWarps * probs_stride(num_tables);
    }
};

// Grid: x = query tiles, y = batch_heads. `totals` is workspace tile 0.
template <int D, int P>
__global__ void __launch_bounds__(kThreads)
    query_kernel(const bf16* __restrict__ queries, const float* __restrict__ planes,
                 const float* __restrict__ beta_ptr, const float* __restrict__ totals,
                 bf16* __restrict__ out, int seq_len, int num_tables) {
    using Layout = QuerySmem<D, P>;
    constexpr int kCorners = Dims<D, P>::kCorners;
    constexpr int kVec = Dims<D, P>::kVec;
    constexpr int kSliceFloats = Dims<D, P>::kSliceFloats;
    constexpr int kBucketFloats = kCorners * D;

    extern __shared__ __align__(16) float smem[];
    float* buckets_s = smem + Layout::buckets(num_tables);  // [L][R][D]
    float* mass_s = smem + Layout::mass(num_tables);        // [L][R]
    float* planes_s = smem + Layout::planes(num_tables);    // [L][P][D]

    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    float* probs_s = smem + Layout::probs(num_tables) + warp * Layout::probs_stride(num_tables);

    // The L slices of this head are contiguous in the workspace (bh-major).
    const float* head_totals = totals + static_cast<size_t>(bh) * num_tables * kSliceFloats;
    for (int i = threadIdx.x; i < num_tables * kSliceFloats; i += kThreads) {
        const int table = i / kSliceFloats;
        const int offset = i - table * kSliceFloats;
        const float value = head_totals[i];
        if (offset < kBucketFloats) {
            buckets_s[table * kBucketFloats + offset] = value;
        } else {
            mass_s[table * kCorners + offset - kBucketFloats] = value;
        }
    }
    for (int i = threadIdx.x; i < num_tables * P * D; i += kThreads) planes_s[i] = planes[i];
    const float beta = *beta_ptr;
    __syncthreads();

    const int tile_begin = blockIdx.x * kQueryTileTokens;
    const int tile_end = min(seq_len, tile_begin + kQueryTileTokens);

    // Warp-uniform loop: only __syncwarp inside, no block barriers.
    for (int token = tile_begin + warp; token < tile_end; token += kWarps) {
        const size_t row = token_row_offset(bh, token, seq_len, D) + lane * kVec;
        float query[kVec];
        load_bf16_vec<kVec>(queries + row, query);

        // Lane r computes phi_l[r] for every table and its share of Den.
        float den_partial = 0.0f;
        for (int table = 0; table < num_tables; ++table) {
            const float phi = corner_probability<D, P>(query, planes_s + table * P * D, beta, lane);
            if (lane < kCorners) {
                probs_s[table * kCorners + lane] = phi;
                den_partial = fmaf(phi, mass_s[table * kCorners + lane], den_partial);
            }
        }
        __syncwarp();

        // Lane owns output columns [lane * kVec, lane * kVec + kVec).
        float numer[kVec];
#pragma unroll
        for (int j = 0; j < kVec; ++j) numer[j] = 0.0f;
        for (int table = 0; table < num_tables; ++table) {
            const float* table_buckets = buckets_s + table * kBucketFloats + lane * kVec;
#pragma unroll
            for (int r = 0; r < kCorners; ++r) {
                const float phi = probs_s[table * kCorners + r];
                float bucket[kVec];
                load_f32_vec<kVec>(table_buckets + r * D, bucket);
#pragma unroll
                for (int j = 0; j < kVec; ++j) numer[j] = fmaf(phi, bucket[j], numer[j]);
            }
        }

        // Den > 0 in exact arithmetic; it is 0 only after fp32 underflow, when
        // the query's buckets hold no key mass. Such rows output 0. NaN in Den
        // propagates instead of being masked.
        const float den = warp_allreduce_sum(den_partial);
        const float inv_den = den == 0.0f ? 0.0f : 1.0f / den;
        float result[kVec];
#pragma unroll
        for (int j = 0; j < kVec; ++j) result[j] = numer[j] * inv_den;
        store_bf16_vec<kVec>(out + row, result);
        __syncwarp();  // probs_s is rewritten by the next token
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

bool is_supported(const ForwardShape& shape) {
    return (shape.head_dim == 64 || shape.head_dim == 128) && shape.num_planes >= 1 &&
           shape.num_planes <= kMaxPlanes && shape.num_tables >= 1 &&
           shape.num_tables <= kMaxTables && shape.seq_len >= 1 &&
           shape.seq_len <= kMaxSeqLen && shape.batch_heads >= 1 &&
           shape.batch_heads <= kMaxGridY;
}

int num_build_tiles(const ForwardShape& shape) {
    return ceil_div(shape.seq_len, kBuildTileTokens);
}

size_t workspace_slice_floats(const ForwardShape& shape) {
    return (static_cast<size_t>(1) << shape.num_planes) * (shape.head_dim + 1);
}

size_t workspace_floats(const ForwardShape& shape) {
    return static_cast<size_t>(num_build_tiles(shape)) * shape.batch_heads * shape.num_tables *
           workspace_slice_floats(shape);
}

cudaError_t race_bucket_build(const bf16* k, const bf16* v, const float* planes,
                              const float* beta, float* workspace, const ForwardShape& shape,
                              cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const dim3 grid(num_build_tiles(shape) * shape.num_tables, shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        bucket_build_kernel<D, P, false><<<grid, kThreads, 0, stream>>>(
            k, v, nullptr, planes, beta, workspace, shape.seq_len, shape.batch_heads,
            shape.num_tables);
        return cudaGetLastError();
    });
}

cudaError_t race_bucket_reduce(float* workspace, const ForwardShape& shape, cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const int num_tiles = num_build_tiles(shape);
    const int64_t tile_floats = static_cast<int64_t>(shape.batch_heads) * shape.num_tables *
                                static_cast<int64_t>(workspace_slice_floats(shape));
    for (int64_t stride = 1; stride < num_tiles; stride *= kReduceFanIn) {
        const int64_t span = stride * kReduceFanIn;
        const int64_t num_groups = (num_tiles + span - 1) / span;
        const int64_t total_work = num_groups * tile_floats;
        const int64_t blocks = std::min<int64_t>((total_work + kThreads - 1) / kThreads,
                                                 kMaxReduceBlocks);
        tree_reduce_pass_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            workspace, tile_floats, num_tiles, stride, total_work);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

cudaError_t race_query(const bf16* q, const float* planes, const float* beta,
                       const float* workspace, bf16* out, const ForwardShape& shape,
                       cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const dim3 grid(ceil_div(shape.seq_len, kQueryTileTokens), shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        auto* kernel = query_kernel<D, P>;
        const size_t smem_bytes =
            static_cast<size_t>(QuerySmem<D, P>::total_floats(shape.num_tables)) * sizeof(float);
        // Worst case (L = 4, P = 5, D = 128) is ~80 KB, above the 48 KB
        // default, so larger configurations opt in. The attribute is per
        // device, so it is set on every call rather than cached.
        if (smem_bytes > kDefaultDynamicSmemBytes) {
            const cudaError_t err = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
            if (err != cudaSuccess) return err;
        }
        kernel<<<grid, kThreads, smem_bytes, stream>>>(q, planes, beta, workspace, out,
                                                       shape.seq_len, shape.num_tables);
        return cudaGetLastError();
    });
}

cudaError_t race_forward(const bf16* q, const bf16* k, const bf16* v, const float* planes,
                         const float* beta, float* workspace, bf16* out, const ForwardShape& shape,
                         cudaStream_t stream) {
    cudaError_t err = race_bucket_build(k, v, planes, beta, workspace, shape, stream);
    if (err != cudaSuccess) return err;
    err = race_bucket_reduce(workspace, shape, stream);
    if (err != cudaSuccess) return err;
    return race_query(q, planes, beta, workspace, out, shape, stream);
}

}  // namespace race
