// Chunk-parallel causal RACE Attention forward, v2a: three kernels plus host
// launchers. Design and numbers: docs/causal_v2_design.md (sections 3-7).
//
// Math, per stream (one batch-head) with Phi(x) in R^S the L tables' corner
// probabilities, table-major (feature s = l * R + r):
//   A_i = sum_{j<=i} Phi_K,j,   B_i = sum_{j<=i} Phi_K,j (x) v_j
//   O_i = (Phi_Q,i . B_i) / (Phi_Q,i . A_i)
//
// Two levels of chunking (plan section 3.3):
//   - A tile is T_blk consecutive tokens of one stream. K1 sums each tile, K2
//     turns the tile sums into exclusive prefixes, so tile t starts from the
//     state after all earlier tiles.
//   - A sub-chunk is C tokens. One K3 CTA owns one tile and walks its
//     sub-chunks in order with the running state (A, B) in shared memory.
//     For a sub-chunk with rows Phi_Q, Phi_K, V (C each) and state (A, B)
//     from before the sub-chunk:
//         G   = tril(Phi_Q Phi_K^T)          (j <= i, diagonal included)
//         Den = Phi_Q A + rowsum(G)
//         Num = Phi_Q B + G V
//         O   = Num * (1 / Den)
//         A  += colsum(Phi_K),   B += Phi_K^T V
//     The carry is exclusive and the mask inclusive, so query i sees exactly
//     keys 0..i, like the reference's cumsum.
//
// Kernels:
//   K1 tile_sums_kernel: the non-causal bucket build (its stage_tokens and
//      accumulate_stage from src/noncausal/kernels/race_internal.cuh), with
//      the tile length as a runtime argument instead of kBuildTileTokens.
//   K2 tile_scan_kernel: one thread per workspace element of a tile (all
//      streams at once); each walks the tiles in index order.
//   K3 output_pass_kernel: one CTA per (tile, stream), 256 threads, dynamic
//      shared memory, fp32 CUDA cores. Phi is recomputed, never stored.
//
// Precision: bf16 in and out, fp32 everywhere in between, accurate tanhf and
// expf, one reciprocal per token, one bf16 rounding per output. No atomics.
// K1 and K3 compute Phi_K with different fp32 operation orders (warp
// butterfly vs one thread's serial dot product). That is harmless: each
// contribution enters A and B with the same Phi value, so Num and Den stay
// consistent and a constant V still comes back exactly.
//
// Differences from docs/causal_v2_design.md section 6.2, each for a concrete reason:
//   - Den is computed in the G step instead of a separate step: the 16
//     threads that hold one row of G add their G entries and a strided share
//     of Phi_Q . A, then a 16-lane shuffle reduction. This drops one barrier
//     (5 per sub-chunk instead of 6) and the 64-thread serial loop.
//   - C is 64 only when the Num micro-tile stays at 16 accumulators
//     (C * D <= 4096) and the L = 4 footprint fits the smallest opt-in
//     shared-memory limit of the targets (99 KiB on sm_89); otherwise 32.
//     So config A (D = 64, P = 4) uses C = 64 and config B uses C = 32.
//   - The state update walks the S rows in passes of 64 (D = 64) or 32
//     (D = 128) rows so its register tile stays at 16 accumulators for any
//     L; A is updated by threads 0..S-1 in the same phase.
#include "race_causal_fwd.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../../noncausal/kernels/race_common.cuh"
#include "../../noncausal/kernels/race_internal.cuh"

namespace race {
namespace causal {
namespace {

using detail::accumulate_stage;
using detail::bf16;
using detail::BuildOwnership;
using detail::Dims;
using detail::dispatch_planes;
using detail::kDefaultDynamicSmemBytes;
using detail::kMaxGridY;
using detail::kMaxPlanes;
using detail::kMaxTables;
using detail::kStageTokens;
using detail::kThreads;
using detail::kWarps;
using detail::stage_tokens;
using detail::workspace_slice_offset;

// Token indices are int; tile_begin + tile length must not overflow.
constexpr int kMaxSeqLen = std::numeric_limits<int>::max() - kMaxAutoTileTokens - 1;

// K3 threads form a 16 x 16 grid (row lane ty = tid / 16, column lane
// tx = tid % 16) for the register micro-tiled products.
constexpr int kGridSide = 16;
static_assert(kGridSide * kGridSide == kThreads, "K3 assumes a 16 x 16 thread grid");

// The smallest per-CTA opt-in shared memory among sm_80 (163 KiB), sm_89
// (99 KiB) and sm_90 (227 KiB). Used only to choose C, not to launch.
constexpr size_t kPortableSmemBytes = 99 * 1024;

// K2 loads this many tiles ahead of the running sum so several independent
// loads are in flight per thread (plan section 5, Little's law).
constexpr int kScanUnroll = 8;

// ---------------------------------------------------------------------------
// Output-pass shared-memory layout
// ---------------------------------------------------------------------------

__host__ __device__ constexpr size_t align16(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

// Byte offsets of the K3 regions. Every region starts 16-byte aligned.
//   x_q, x_k, x_v : bf16 [C][D + 2]   staged rows; (D + 2) / 2 words per row
//                                     is odd, so a warp reading one column of
//                                     32 rows hits 32 banks
//   gram          : fp32 [C][C + 1]   aliases x_q and x_k, which are dead
//                                     once Phi exists
//   planes        : fp32 [L][P][D]    read as warp broadcasts
//   phi_q, phi_k  : fp32 [C][S + 1]   S + 1 is odd (S is even)
//   state         : fp32 [S][D + 1]   B in columns 0..D-1, A in column D
//   inv_den       : fp32 [C]
struct OutputSmemLayout {
    size_t x_q, x_k, x_v, planes, phi_q, phi_k, state, inv_den, total;
};

__host__ __device__ constexpr OutputSmemLayout output_smem_layout(int head_dim, int num_planes,
                                                                  int chunk, int num_tables) {
    const size_t num_features = static_cast<size_t>(num_tables) << num_planes;
    const size_t x_bytes = align16(static_cast<size_t>(chunk) * (head_dim + 2) * sizeof(bf16));
    const size_t phi_bytes = align16(static_cast<size_t>(chunk) * (num_features + 1) * sizeof(float));
    OutputSmemLayout layout{};
    layout.x_q = 0;
    layout.x_k = x_bytes;
    layout.x_v = 2 * x_bytes;
    layout.planes = 3 * x_bytes;
    layout.phi_q = layout.planes +
                   align16(static_cast<size_t>(num_tables) * num_planes * head_dim * sizeof(float));
    layout.phi_k = layout.phi_q + phi_bytes;
    layout.state = layout.phi_k + phi_bytes;
    layout.inv_den = layout.state + align16(num_features * (head_dim + 1) * sizeof(float));
    layout.total = layout.inv_den + align16(static_cast<size_t>(chunk) * sizeof(float));
    return layout;
}

// Sub-chunk length for (D, P), see the header comment of this file.
__host__ __device__ constexpr int choose_sub_chunk(int head_dim, int num_planes) {
    return 64 * head_dim <= 4096 &&
                   output_smem_layout(head_dim, num_planes, 64, kMaxTables).total <=
                       kPortableSmemBytes
               ? 64
               : 32;
}

template <int D, int P, int C>
struct OutputPassTraits {
    static constexpr int kCorners = 1 << P;
    static constexpr int kSliceFloats = kCorners * (D + 1);
    static constexpr int kRowTiles = C / kGridSide;  // G / Num rows per thread: ty + 16 a
    static constexpr int kColTiles = D / kGridSide;  // Num / update columns per thread: tx + 16 b
    // State rows per thread in one update pass, so the update tile is
    // kUpdateRowTiles x kColTiles = 16 accumulators.
    static constexpr int kUpdateRowTiles = kGridSide / kColTiles;
    static constexpr int kUpdateRowsPerPass = kGridSide * kUpdateRowTiles;
    static constexpr int kXStride = D + 2;      // bf16 elements per staged row
    static constexpr int kStateStride = D + 1;  // floats per state row
    static constexpr int kGramStride = C + 1;   // floats per G row
    static constexpr int kVectorsPerRow = D / 8;  // 16-byte vectors per bf16 row
    static constexpr int kLoadItersPerTensor = C * kVectorsPerRow / kThreads;

    static_assert(C == 32 || C == 64, "sub-chunk must be 32 or 64 tokens");
    static_assert(C % kWarpSize == 0, "a warp of the Phi step must cover one (side, table)");
    static_assert((C * kVectorsPerRow) % kThreads == 0, "loads must split evenly over threads");
    static_assert(kRowTiles * kColTiles <= 16, "Num micro-tile is limited to 16 accumulators");
    static_assert(static_cast<size_t>(C) * (C + 1) * sizeof(float) <=
                      2 * static_cast<size_t>(C) * (D + 2) * sizeof(bf16),
                  "G must fit in the staged Q and K rows it aliases");
};

// ---------------------------------------------------------------------------
// K1: tile sums
// ---------------------------------------------------------------------------

// The non-causal bucket_build_kernel (race_internal.cuh) with kWeighted =
// false and the tile length as an argument; the staging, the fixed (r, c)
// ownership, the summation order and the workspace slice are identical.
// Grid: x = num_tiles * num_tables (table fastest, so the L CTAs of a tile
// share K and V through L2), y = batch_heads.
template <int D, int P>
__global__ void __launch_bounds__(kThreads)
    tile_sums_kernel(const bf16* __restrict__ keys, const bf16* __restrict__ values,
                     const float* __restrict__ planes, const float* __restrict__ beta_ptr,
                     float* __restrict__ workspace, int seq_len, int batch_heads, int num_tables,
                     int tile_tokens) {
    using Own = BuildOwnership<D, P>;
    constexpr int kCorners = Dims<D, P>::kCorners;
    constexpr int kRows = Own::kRowsPerThread;

    __shared__ __align__(16) float planes_s[P * D];
    __shared__ __align__(16) float probs_s[2][kStageTokens][kCorners];
    __shared__ __align__(16) float values_s[2][kStageTokens][D];
    __shared__ float warp_mass_s[kWarps][kWarpSize];

    const int table = blockIdx.x % num_tables;
    const int tile = blockIdx.x / num_tables;
    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    // tile_begin + min(...) equals min(seq_len, tile_begin + tile_tokens); in
    // this form ptxas does not spill a register in <64, 5> on sm_80 and sm_89.
    const int tile_begin = tile * tile_tokens;
    const int tile_end = tile_begin + min(seq_len - tile_begin, tile_tokens);
    const int num_stages = (tile_end - tile_begin + kStageTokens - 1) / kStageTokens;

    const float* table_planes = planes + static_cast<size_t>(table) * P * D;
    for (int i = threadIdx.x; i < P * D; i += kThreads) planes_s[i] = table_planes[i];
    const float beta = *beta_ptr;

    const bf16* key_rows = keys + static_cast<size_t>(bh) * seq_len * D;
    const bf16* value_rows = values + static_cast<size_t>(bh) * seq_len * D;

    const int column = threadIdx.x % D;
    const int row_group = threadIdx.x / D;
    const bool owns_rows = row_group < Own::kActiveRowGroups;
    const int first_row = row_group * kRows;

    float acc[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) acc[i] = 0.0f;
    float mass = 0.0f;  // lane r: sum of phi[r] over the tokens this warp staged

    __syncthreads();  // planes_s is ready
    stage_tokens<D, P, false>(key_rows, value_rows, nullptr, planes_s, beta, tile_begin, tile_end,
                              probs_s[0], values_s[0], warp, lane, mass);
    __syncthreads();

    // Invariant at the top of iteration s: buffer s & 1 holds stage s and no
    // thread still reads buffer (s + 1) & 1 (its last reader was iteration
    // s - 1, which ended with a barrier).
    for (int s = 0; s < num_stages; ++s) {
        const int buffer = s & 1;
        if (s + 1 < num_stages) {
            stage_tokens<D, P, false>(key_rows, value_rows, nullptr, planes_s, beta,
                                      tile_begin + (s + 1) * kStageTokens, tile_end,
                                      probs_s[buffer ^ 1], values_s[buffer ^ 1], warp, lane, mass);
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

// ---------------------------------------------------------------------------
// K2: exclusive scan over tiles
// ---------------------------------------------------------------------------

// Element e of every tile (e < tile_floats = BH * L * R * (D + 1), which spans
// all streams because a tile's slices are contiguous) is owned by one thread,
// which walks the tiles in index order: tile t receives the sum of tiles
// 0..t-1, and final_state[e] the sum of all tiles. Consecutive threads touch
// consecutive floats of a tile, so every load and store is coalesced. The
// loads of kScanUnroll tiles do not depend on the running sum, so they are
// issued together before the sequential adds.
__global__ void __launch_bounds__(kThreads)
    tile_scan_kernel(float* __restrict__ workspace, float* __restrict__ final_state,
                     int64_t tile_floats, int num_tiles) {
    const int64_t element = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (element >= tile_floats) return;  // no barriers in this kernel

    float* column = workspace + element;
    float running = 0.0f;
    for (int first = 0; first < num_tiles; first += kScanUnroll) {
        float totals[kScanUnroll];
#pragma unroll
        for (int u = 0; u < kScanUnroll; ++u) {
            const int tile = first + u;
            totals[u] = tile < num_tiles ? column[tile * tile_floats] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < kScanUnroll; ++u) {
            const int tile = first + u;
            if (tile < num_tiles) {
                column[tile * tile_floats] = running;
                running += totals[u];
            }
        }
    }
    if (final_state != nullptr) final_state[element] = running;
}

// ---------------------------------------------------------------------------
// K3: output pass
// ---------------------------------------------------------------------------

// Step 1. Copies the C rows of q, k and v starting at the sub-chunk into
// x_q, x_k, x_v. Rows at or past rows_valid (past the tile or the sequence)
// are zero-filled: V must be finite there because G and Phi_K are zero for
// those rows and 0 * NaN would still poison the sums.
// Each thread issues all of its 16-byte global loads before any store.
template <int D, int P, int C>
__device__ __forceinline__ void load_sub_chunk(const bf16* __restrict__ q_rows,
                                               const bf16* __restrict__ k_rows,
                                               const bf16* __restrict__ v_rows, int rows_valid,
                                               bf16* x_q, bf16* x_k, bf16* x_v) {
    using Traits = OutputPassTraits<D, P, C>;
    constexpr int kIters = Traits::kLoadItersPerTensor;
    constexpr int kVectorsPerRow = Traits::kVectorsPerRow;

    uint4 raw[3][kIters];
#pragma unroll
    for (int tensor = 0; tensor < 3; ++tensor) {
        const bf16* src = tensor == 0 ? q_rows : (tensor == 1 ? k_rows : v_rows);
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int vector = it * kThreads + threadIdx.x;
            const int row = vector / kVectorsPerRow;
            const int col = (vector % kVectorsPerRow) * 8;
            raw[tensor][it] = make_uint4(0u, 0u, 0u, 0u);
            if (row < rows_valid) {
                raw[tensor][it] = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * D + col);
            }
        }
    }
#pragma unroll
    for (int tensor = 0; tensor < 3; ++tensor) {
        bf16* dst = tensor == 0 ? x_q : (tensor == 1 ? x_k : x_v);
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int vector = it * kThreads + threadIdx.x;
            const int row = vector / kVectorsPerRow;
            const int col = (vector % kVectorsPerRow) * 8;
            // The row stride (D + 2 elements) keeps rows 4-byte aligned only,
            // so the 16 bytes go in as four 32-bit words.
            uint32_t* words = reinterpret_cast<uint32_t*>(dst + row * Traits::kXStride + col);
            words[0] = raw[tensor][it].x;
            words[1] = raw[tensor][it].y;
            words[2] = raw[tensor][it].z;
            words[3] = raw[tensor][it].w;
        }
    }
}

// Step 2. Phi for every (side, table, token) of the sub-chunk: P dot products
// of length D against the table's planes, tanh, a sigmoid pair per plane, then
// the R corner products. Pair index p maps to token = p % C, table = (p / C) %
// L, side = p / (C * L), so a warp is 32 consecutive tokens of one (side,
// table): plane reads are broadcasts and row reads hit 32 banks (odd word
// stride). Key rows at or past rows_valid get Phi = 0 so they add nothing to
// G or to the state.
template <int D, int P, int C>
__device__ __forceinline__ void compute_features(const bf16* x_q, const bf16* x_k,
                                                 const float* planes_s, float beta, int num_tables,
                                                 int rows_valid, int phi_stride, float* phi_q,
                                                 float* phi_k) {
    using Traits = OutputPassTraits<D, P, C>;
    constexpr int kCorners = Traits::kCorners;
    const int pairs_per_side = C * num_tables;

    for (int pair = threadIdx.x; pair < 2 * pairs_per_side; pair += kThreads) {
        const bool key_side = pair >= pairs_per_side;
        const int within_side = key_side ? pair - pairs_per_side : pair;
        const int table = within_side / C;
        const int token = within_side - table * C;
        const bf16* row = (key_side ? x_k : x_q) + token * Traits::kXStride;
        const float* table_planes = planes_s + table * P * D;

        float projection[P];
#pragma unroll
        for (int t = 0; t < P; ++t) projection[t] = 0.0f;
#pragma unroll 8
        for (int col = 0; col < D; col += 2) {
            const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(row + col));
#pragma unroll
            for (int t = 0; t < P; ++t) {
                const float2 w = *reinterpret_cast<const float2*>(table_planes + t * D + col);
                projection[t] = fmaf(x.x, w.x, projection[t]);
                projection[t] = fmaf(x.y, w.y, projection[t]);
            }
        }

        float prob_plus[P];
        float prob_minus[P];
#pragma unroll
        for (int t = 0; t < P; ++t) {
            sigmoid_pair(2.0f * beta * tanhf(projection[t]), prob_plus[t], prob_minus[t]);
        }

        const bool keep = !key_side || token < rows_valid;
        float* dst = (key_side ? phi_k : phi_q) + token * phi_stride + table * kCorners;
#pragma unroll
        for (int corner = 0; corner < kCorners; ++corner) {
            dst[corner] = keep ? bernoulli_product<P>(prob_plus, prob_minus, corner) : 0.0f;
        }
    }
}

// Step 3. G = tril(Phi_Q Phi_K^T) into gram, and 1 / Den into inv_den.
// Thread (ty, tx) owns G[ty + 16a][tx + 16b] for a, b < C / 16, so the 16
// threads with the same ty (one half-warp) hold whole rows. Each adds its
// masked G entries and the terms s = tx, tx + 16, ... of Phi_Q[i] . A; a
// xor-shuffle over the half-warp then gives every lane the same Den_i.
// Den == 0 (only after fp32 underflow) yields an output row of 0.
template <int D, int P, int C>
__device__ __forceinline__ void gram_and_denominator(const float* phi_q, const float* phi_k,
                                                     const float* state, int num_features,
                                                     int phi_stride, float* gram, float* inv_den) {
    using Traits = OutputPassTraits<D, P, C>;
    constexpr int kRowTiles = Traits::kRowTiles;
    const int ty = threadIdx.x / kGridSide;
    const int tx = threadIdx.x % kGridSide;

    float acc[kRowTiles][kRowTiles];
#pragma unroll
    for (int a = 0; a < kRowTiles; ++a) {
#pragma unroll
        for (int b = 0; b < kRowTiles; ++b) acc[a][b] = 0.0f;
    }
    for (int s = 0; s < num_features; ++s) {
        float query[kRowTiles];
        float key[kRowTiles];
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) query[a] = phi_q[(ty + kGridSide * a) * phi_stride + s];
#pragma unroll
        for (int b = 0; b < kRowTiles; ++b) key[b] = phi_k[(tx + kGridSide * b) * phi_stride + s];
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) {
#pragma unroll
            for (int b = 0; b < kRowTiles; ++b) acc[a][b] = fmaf(query[a], key[b], acc[a][b]);
        }
    }

    float den[kRowTiles];
#pragma unroll
    for (int a = 0; a < kRowTiles; ++a) {
        const int i = ty + kGridSide * a;
        den[a] = 0.0f;
#pragma unroll
        for (int b = 0; b < kRowTiles; ++b) {
            const int j = tx + kGridSide * b;
            const float g = j <= i ? acc[a][b] : 0.0f;
            gram[i * Traits::kGramStride + j] = g;
            den[a] += g;
        }
        for (int s = tx; s < num_features; s += kGridSide) {
            den[a] = fmaf(phi_q[i * phi_stride + s], state[s * Traits::kStateStride + D], den[a]);
        }
    }
    // Offsets below 16 stay inside the half-warp that owns the row.
#pragma unroll
    for (int a = 0; a < kRowTiles; ++a) {
#pragma unroll
        for (int offset = kGridSide / 2; offset > 0; offset >>= 1) {
            den[a] += __shfl_xor_sync(kFullWarpMask, den[a], offset);
        }
    }
    if (tx == 0) {
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) {
            inv_den[ty + kGridSide * a] = den[a] == 0.0f ? 0.0f : 1.0f / den[a];
        }
    }
}

// Steps 4-5. Num = [Phi_Q | G] [B ; V] for the thread's (ty + 16a, tx + 16b)
// micro-tile, k over S (carry, reading the state before this sub-chunk) and
// then over C (intra, G is already masked), in one set of accumulators. Rows
// of the sequence are scaled by 1 / Den, rounded to bf16 and stored: the 16
// threads of a half-warp write 16 consecutive bf16 of one row, a full 32-byte
// sector.
template <int D, int P, int C>
__device__ __forceinline__ void numerator_and_store(const float* phi_q, const float* state,
                                                    const float* gram, const bf16* x_v,
                                                    const float* inv_den, int num_features,
                                                    int phi_stride, int rows_valid,
                                                    bf16* __restrict__ out_rows) {
    using Traits = OutputPassTraits<D, P, C>;
    constexpr int kRowTiles = Traits::kRowTiles;
    constexpr int kColTiles = Traits::kColTiles;
    const int ty = threadIdx.x / kGridSide;
    const int tx = threadIdx.x % kGridSide;

    float acc[kRowTiles][kColTiles];
#pragma unroll
    for (int a = 0; a < kRowTiles; ++a) {
#pragma unroll
        for (int b = 0; b < kColTiles; ++b) acc[a][b] = 0.0f;
    }
    for (int s = 0; s < num_features; ++s) {
        float weight[kRowTiles];
        float carried[kColTiles];
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) weight[a] = phi_q[(ty + kGridSide * a) * phi_stride + s];
#pragma unroll
        for (int b = 0; b < kColTiles; ++b) carried[b] = state[s * Traits::kStateStride + tx + kGridSide * b];
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) {
#pragma unroll
            for (int b = 0; b < kColTiles; ++b) acc[a][b] = fmaf(weight[a], carried[b], acc[a][b]);
        }
    }
#pragma unroll 4
    for (int j = 0; j < C; ++j) {
        float weight[kRowTiles];
        float value[kColTiles];
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) weight[a] = gram[(ty + kGridSide * a) * Traits::kGramStride + j];
#pragma unroll
        for (int b = 0; b < kColTiles; ++b) {
            value[b] = __bfloat162float(x_v[j * Traits::kXStride + tx + kGridSide * b]);
        }
#pragma unroll
        for (int a = 0; a < kRowTiles; ++a) {
#pragma unroll
            for (int b = 0; b < kColTiles; ++b) acc[a][b] = fmaf(weight[a], value[b], acc[a][b]);
        }
    }

#pragma unroll
    for (int a = 0; a < kRowTiles; ++a) {
        const int i = ty + kGridSide * a;
        if (i < rows_valid) {
            const float scale = inv_den[i];
#pragma unroll
            for (int b = 0; b < kColTiles; ++b) {
                out_rows[static_cast<size_t>(i) * D + tx + kGridSide * b] =
                    __float2bfloat16_rn(acc[a][b] * scale);
            }
        }
    }
}

// Step 6. B += Phi_K^T V and A += colsum(Phi_K). The sub-chunk's partial sum
// is formed first and then added, so the running state sees one add per
// sub-chunk. B is covered in passes of kUpdateRowsPerPass rows, thread
// (ty, tx) owning rows s0 + ty + 16a and columns tx + 16b; rows s >= S (only
// when S is not a multiple of the pass) read Phi as 0 and are not written.
// Threads 0..S-1 update A (column D), which no B owner touches.
template <int D, int P, int C>
__device__ __forceinline__ void update_state(const float* phi_k, const bf16* x_v,
                                             int num_features, int phi_stride, float* state) {
    using Traits = OutputPassTraits<D, P, C>;
    constexpr int kRows = Traits::kUpdateRowTiles;
    constexpr int kColTiles = Traits::kColTiles;
    const int ty = threadIdx.x / kGridSide;
    const int tx = threadIdx.x % kGridSide;

    for (int first_row = 0; first_row < num_features; first_row += Traits::kUpdateRowsPerPass) {
        float acc[kRows][kColTiles];
#pragma unroll
        for (int a = 0; a < kRows; ++a) {
#pragma unroll
            for (int b = 0; b < kColTiles; ++b) acc[a][b] = 0.0f;
        }
#pragma unroll 4
        for (int j = 0; j < C; ++j) {
            float weight[kRows];
            float value[kColTiles];
#pragma unroll
            for (int a = 0; a < kRows; ++a) {
                const int s = first_row + ty + kGridSide * a;
                weight[a] = s < num_features ? phi_k[j * phi_stride + s] : 0.0f;
            }
#pragma unroll
            for (int b = 0; b < kColTiles; ++b) {
                value[b] = __bfloat162float(x_v[j * Traits::kXStride + tx + kGridSide * b]);
            }
#pragma unroll
            for (int a = 0; a < kRows; ++a) {
#pragma unroll
                for (int b = 0; b < kColTiles; ++b) acc[a][b] = fmaf(weight[a], value[b], acc[a][b]);
            }
        }
#pragma unroll
        for (int a = 0; a < kRows; ++a) {
            const int s = first_row + ty + kGridSide * a;
            if (s < num_features) {
#pragma unroll
                for (int b = 0; b < kColTiles; ++b) {
                    state[s * Traits::kStateStride + tx + kGridSide * b] += acc[a][b];
                }
            }
        }
    }

    if (threadIdx.x < num_features) {
        const int s = threadIdx.x;
        float mass = 0.0f;
        for (int j = 0; j < C; ++j) mass += phi_k[j * phi_stride + s];
        state[s * Traits::kStateStride + D] += mass;
    }
}

// Grid: x = tile, y = stream (batch-head). The workspace holds exclusive
// prefixes (after K2). All loops and barriers are uniform across the CTA.
// Register cap: 3 CTAs per SM (at most 80 registers). At the benchmarked
// shapes (L = 4, P = 4-5, 78-131 KiB of shared memory) shared memory already
// limits residency to 1-2 CTAs per SM; smaller shapes can fit a third. A
// 64-register cap would add 4-32 bytes of spills in the C = 64
// instantiations without adding a resident CTA.
template <int D, int P, int C>
__global__ void __launch_bounds__(kThreads, 3)
    output_pass_kernel(const bf16* __restrict__ queries, const bf16* __restrict__ keys,
                       const bf16* __restrict__ values, const float* __restrict__ planes,
                       const float* __restrict__ beta_ptr, const float* __restrict__ workspace,
                       bf16* __restrict__ out, int seq_len, int batch_heads, int num_tables,
                       int tile_tokens) {
    using Traits = OutputPassTraits<D, P, C>;
    constexpr int kCorners = Traits::kCorners;
    constexpr int kSliceFloats = Traits::kSliceFloats;
    constexpr int kBucketFloats = kCorners * D;

    extern __shared__ __align__(16) unsigned char smem[];
    const OutputSmemLayout layout = output_smem_layout(D, P, C, num_tables);
    bf16* x_q = reinterpret_cast<bf16*>(smem + layout.x_q);
    bf16* x_k = reinterpret_cast<bf16*>(smem + layout.x_k);
    bf16* x_v = reinterpret_cast<bf16*>(smem + layout.x_v);
    float* gram = reinterpret_cast<float*>(smem + layout.x_q);  // aliases x_q and x_k
    float* planes_s = reinterpret_cast<float*>(smem + layout.planes);
    float* phi_q = reinterpret_cast<float*>(smem + layout.phi_q);
    float* phi_k = reinterpret_cast<float*>(smem + layout.phi_k);
    float* state = reinterpret_cast<float*>(smem + layout.state);
    float* inv_den = reinterpret_cast<float*>(smem + layout.inv_den);

    const int num_features = num_tables * kCorners;
    const int phi_stride = num_features + 1;
    const int tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tile_begin = tile * tile_tokens;
    const int tile_end = min(seq_len, tile_begin + tile_tokens);

    // Prologue: the exclusive prefix of this (tile, stream) and the planes.
    // The L slices of one (tile, stream) are contiguous in the workspace;
    // slice offset r * D + c goes to state[l * R + r][c] and R * D + r (A) to
    // state[l * R + r][D].
    const float* prefix = workspace + (static_cast<size_t>(tile) * batch_heads + bh) *
                                          num_tables * kSliceFloats;
    for (int i = threadIdx.x; i < num_tables * kSliceFloats; i += kThreads) {
        const int table = i / kSliceFloats;
        const int offset = i - table * kSliceFloats;
        const int row = offset < kBucketFloats ? offset / D : offset - kBucketFloats;
        const int col = offset < kBucketFloats ? offset - row * D : D;
        state[(table * kCorners + row) * Traits::kStateStride + col] = prefix[i];
    }
    for (int i = threadIdx.x; i < num_tables * P * D; i += kThreads) planes_s[i] = planes[i];
    const float beta = *beta_ptr;

    const size_t stream_offset = static_cast<size_t>(bh) * seq_len * D;
    // Barrier invariant: each phase reads only what an earlier phase wrote
    // before the barrier in between, and a region is rewritten only after the
    // barrier that follows its last read. The prologue's state and planes
    // writes are covered by the first barrier (nothing reads them in step 1).
    for (int chunk_begin = tile_begin; chunk_begin < tile_end; chunk_begin += C) {
        const int rows_valid = min(C, tile_end - chunk_begin);
        const size_t chunk_offset = stream_offset + static_cast<size_t>(chunk_begin) * D;

        load_sub_chunk<D, P, C>(queries + chunk_offset, keys + chunk_offset, values + chunk_offset,
                                rows_valid, x_q, x_k, x_v);
        __syncthreads();  // x_* ready (and, first time, state and planes)
        compute_features<D, P, C>(x_q, x_k, planes_s, beta, num_tables, rows_valid, phi_stride,
                                  phi_q, phi_k);
        __syncthreads();  // phi ready; x_q and x_k are dead, gram may overwrite them
        gram_and_denominator<D, P, C>(phi_q, phi_k, state, num_features, phi_stride, gram, inv_den);
        __syncthreads();  // gram and inv_den ready
        numerator_and_store<D, P, C>(phi_q, state, gram, x_v, inv_den, num_features, phi_stride,
                                     rows_valid, out + chunk_offset);
        __syncthreads();  // every read of the state before this sub-chunk is done
        update_state<D, P, C>(phi_k, x_v, num_features, phi_stride, state);
        __syncthreads();  // state updated; x_v and phi_k free for the next sub-chunk
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Calls fn(integral_constant<D>, integral_constant<P>) for the runtime shape.
template <typename Fn>
cudaError_t dispatch_shape(const CausalShape& shape, Fn&& fn) {
    switch (shape.head_dim) {
        case 64: return dispatch_planes<64>(shape.num_planes, fn);
        case 128: return dispatch_planes<128>(shape.num_planes, fn);
        default: return cudaErrorInvalidValue;
    }
}

int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t pow2_floor(int64_t x) {
    int64_t p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

// Every output_pass_kernel<D, P, C> has this signature.
using OutputPassKernel = void (*)(const bf16*, const bf16*, const bf16*, const float*,
                                  const float*, const float*, bf16*, int, int, int, int);

// The K3 kernel for (D, P) with its dynamic shared memory opted in. The
// attribute is per device, so it is set on every call rather than cached.
template <int D, int P>
cudaError_t prepare_output_pass(int num_tables, OutputPassKernel* kernel, size_t* smem_bytes) {
    constexpr int C = choose_sub_chunk(D, P);
    *kernel = output_pass_kernel<D, P, C>;
    *smem_bytes = output_smem_layout(D, P, C, num_tables).total;

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    int limit = 0;
    err = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (err != cudaSuccess) return err;
    if (*smem_bytes > static_cast<size_t>(limit)) return cudaErrorInvalidConfiguration;

    if (*smem_bytes > kDefaultDynamicSmemBytes) {
        err = cudaFuncSetAttribute(*kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(*smem_bytes));
        if (err != cudaSuccess) return err;
    }
    // Ask for the largest shared-memory carveout so 2 CTAs fit where the
    // footprint allows it (a hint; the driver may round).
    return cudaFuncSetAttribute(*kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                cudaSharedmemCarveoutMaxShared);
}

// Default tile length: the smallest power of two >= C for which the state
// traffic 16 * S * (D + 1) / T_blk bytes per token is at most 5% of the
// 12 * D bytes of Q/K/V/O traffic, capped at kMaxAutoTileTokens. This gives
// 2048 for config A and 4096 for config B (plan section 3.3).
int default_tile_tokens(const CausalShape& shape) {
    const int64_t num_features = static_cast<int64_t>(shape.num_tables) << shape.num_planes;
    const int64_t state_bytes_x16 = 16 * num_features * (shape.head_dim + 1);
    int tile = sub_chunk_tokens(shape.head_dim, shape.num_planes);
    // 16 S (D + 1) / T <= 0.05 * 12 D   <=>   T * 3 * D >= 80 * S * (D + 1)
    while (tile < kMaxAutoTileTokens &&
           static_cast<int64_t>(tile) * 3 * shape.head_dim < 5 * state_bytes_x16) {
        tile *= 2;
    }
    return tile;
}

}  // namespace

bool is_supported(const CausalShape& shape) {
    return (shape.head_dim == 64 || shape.head_dim == 128) && shape.num_planes >= 1 &&
           shape.num_planes <= kMaxPlanes && shape.num_tables >= 1 &&
           shape.num_tables <= kMaxTables && shape.seq_len >= 1 &&
           shape.seq_len <= kMaxSeqLen && shape.batch_heads >= 1 &&
           shape.batch_heads <= kMaxGridY;
}

int sub_chunk_tokens(int head_dim, int num_planes) { return choose_sub_chunk(head_dim, num_planes); }

size_t output_pass_smem_bytes(const CausalShape& shape) {
    if (!is_supported(shape)) return 0;
    const int chunk = sub_chunk_tokens(shape.head_dim, shape.num_planes);
    return output_smem_layout(shape.head_dim, shape.num_planes, chunk, shape.num_tables).total;
}

cudaError_t race_causal_output_fits(const CausalShape& shape, bool* fits, int* limit_bytes) {
    *fits = false;
    *limit_bytes = 0;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    err = cudaDeviceGetAttribute(limit_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (err != cudaSuccess) return err;
    const size_t needed = output_pass_smem_bytes(shape);
    *fits = needed > 0 && needed <= static_cast<size_t>(*limit_bytes);
    return cudaSuccess;
}

bool is_valid_tile_tokens(const CausalShape& shape, int tile_tokens) {
    const int chunk = sub_chunk_tokens(shape.head_dim, shape.num_planes);
    return is_supported(shape) && tile_tokens > 0 && tile_tokens % chunk == 0 &&
           tile_tokens <= std::numeric_limits<int>::max() - shape.seq_len;
}

int num_tiles(const CausalShape& shape, int tile_tokens) {
    return static_cast<int>(ceil_div64(shape.seq_len, tile_tokens));
}

size_t slice_floats(const CausalShape& shape) {
    return (static_cast<size_t>(1) << shape.num_planes) * (shape.head_dim + 1);
}

size_t workspace_floats(const CausalShape& shape, int tile_tokens) {
    return static_cast<size_t>(num_tiles(shape, tile_tokens)) * final_state_floats(shape);
}

size_t final_state_floats(const CausalShape& shape) {
    return static_cast<size_t>(shape.batch_heads) * shape.num_tables * slice_floats(shape);
}

cudaError_t race_causal_select_tile_tokens(const CausalShape& shape, int* tile_tokens) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    int num_sms = 0;
    err = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess) return err;

    int ctas_per_sm = 0;
    err = dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        OutputPassKernel kernel = nullptr;
        size_t smem_bytes = 0;
        const cudaError_t prep = prepare_output_pass<D, P>(shape.num_tables, &kernel, &smem_bytes);
        if (prep != cudaSuccess) return prep;
        return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, kThreads,
                                                             smem_bytes);
    });
    if (err != cudaSuccess) return err;

    // Enough tiles for about 4 waves of output-pass CTAs, in powers of two.
    const int chunk = sub_chunk_tokens(shape.head_dim, shape.num_planes);
    const int64_t tokens = static_cast<int64_t>(shape.batch_heads) * shape.seq_len;
    const int64_t target = tokens / (4 * static_cast<int64_t>(num_sms) * std::max(ctas_per_sm, 1));
    const int64_t shrunk = std::max<int64_t>(pow2_floor(std::max<int64_t>(target, 1)), chunk);
    *tile_tokens = static_cast<int>(std::min<int64_t>(shrunk, default_tile_tokens(shape)));
    return cudaSuccess;
}

cudaError_t race_causal_tile_sums(const bf16* k, const bf16* v, const float* planes,
                                  const float* beta, float* workspace, const CausalShape& shape,
                                  int tile_tokens, cudaStream_t stream) {
    if (!is_valid_tile_tokens(shape, tile_tokens)) return cudaErrorInvalidValue;
    const dim3 grid(num_tiles(shape, tile_tokens) * shape.num_tables, shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        tile_sums_kernel<D, P><<<grid, kThreads, 0, stream>>>(
            k, v, planes, beta, workspace, shape.seq_len, shape.batch_heads, shape.num_tables,
            tile_tokens);
        return cudaGetLastError();
    });
}

cudaError_t race_causal_tile_scan(float* workspace, float* final_state, const CausalShape& shape,
                                  int tile_tokens, cudaStream_t stream) {
    if (!is_valid_tile_tokens(shape, tile_tokens)) return cudaErrorInvalidValue;
    const int64_t tile_floats = static_cast<int64_t>(final_state_floats(shape));
    const int64_t blocks = ceil_div64(tile_floats, kThreads);
    tile_scan_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        workspace, final_state, tile_floats, num_tiles(shape, tile_tokens));
    return cudaGetLastError();
}

cudaError_t race_causal_output(const bf16* q, const bf16* k, const bf16* v, const float* planes,
                               const float* beta, const float* workspace, bf16* out,
                               const CausalShape& shape, int tile_tokens, cudaStream_t stream) {
    if (!is_valid_tile_tokens(shape, tile_tokens)) return cudaErrorInvalidValue;
    const dim3 grid(num_tiles(shape, tile_tokens), shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        OutputPassKernel kernel = nullptr;
        size_t smem_bytes = 0;
        const cudaError_t err = prepare_output_pass<D, P>(shape.num_tables, &kernel, &smem_bytes);
        if (err != cudaSuccess) return err;
        kernel<<<grid, kThreads, smem_bytes, stream>>>(q, k, v, planes, beta, workspace, out,
                                                       shape.seq_len, shape.batch_heads,
                                                       shape.num_tables, tile_tokens);
        return cudaGetLastError();
    });
}

cudaError_t race_causal_forward(const bf16* q, const bf16* k, const bf16* v, const float* planes,
                                const float* beta, float* workspace, float* final_state, bf16* out,
                                const CausalShape& shape, int tile_tokens, cudaStream_t stream) {
    cudaError_t err = race_causal_tile_sums(k, v, planes, beta, workspace, shape, tile_tokens, stream);
    if (err != cudaSuccess) return err;
    err = race_causal_tile_scan(workspace, final_state, shape, tile_tokens, stream);
    if (err != cudaSuccess) return err;
    return race_causal_output(q, k, v, planes, beta, workspace, out, shape, tile_tokens, stream);
}

}  // namespace causal
}  // namespace race
