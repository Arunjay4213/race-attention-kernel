// Tensor-core variant of the non-causal RACE forward: bucket build and query
// pass with the three GEMM-shaped stages on bf16 wmma (m16n16k16, fp32
// accumulate). The tree reduce between them is the fp32 path's
// (race_bucket_reduce in race_fwd.cu), and so is the workspace layout.
//
// Per stage of kTcStageTokens = 32 tokens, both kernels run the same hashing:
//   1. projection  Z = X W^T        [32 x D] x [D x NP]    (tensor cores)
//   2. hash        u = tanh(Z), (sigmoid(2 beta u), sigmoid(-2 beta u))
//   3. corners     Phi[token][l * R + r] = prod_t (...), rounded to bf16
// where NP = L * P rounded up to 16 and all L tables are stacked, so one CTA
// hashes every table and K, Q and V are read from HBM once. Then
//   bucket build: B += Phi^T V   [LRp x 32] x [32 x D]   (tensor cores),
//                 A += Phi^T 1   (CUDA cores, same rounded Phi),
//   query pass:   Num = Phi_Q B  [32 x LRp] x [LRp x D]  (tensor cores),
//                 Den = Phi_Q A  (CUDA cores, same rounded Phi_Q),
//                 O = Num / Den, bf16,
// with LRp = L * R rounded up to 16 (the stacked corner index is l * R + r).
//
// Precision, stage by stage (tests/numerics.py turns this into the bound):
//   - X is bf16 already, so it enters the MMA exactly. W is fp32 and is split
//     into hi = bf16(W) and lo = bf16(W - hi) with two MMAs per k-step, so
//     the planes carry |W - hi - lo| <= 2^-16 |W| and the dot products are
//     accumulated in fp32. One bf16 W would change the planes by up to
//     2^-8 |W| and the hash with them.
//   - Phi is rounded to bf16 (relative error <= 2^-8) for the fragments of
//     both the build and the query. A and Den are summed from the same
//     rounded values, so Num and Den always use identical weights: the
//     rounding perturbs the weights of a convex combination of the v_j
//     instead of scaling O.
//   - The reduced B is split into hi + lo for the query fragments
//     (relative error <= 2^-16); A stays fp32.
//   - V is bf16 already and enters exactly.
//
// Rules the kernels follow:
//   - Every wmma call is made by a full warp; branches around wmma calls
//     depend only on the warp index. Barriers sit at block-uniform points.
//   - Fragment pointers are 32-byte aligned and every leading dimension is a
//     multiple of 16 bytes: bf16 rows are padded by 8 elements (16 bytes),
//     fp32 rows by 4, and every shared-memory region starts at a multiple of
//     128 bytes.
//   - Constant operands stay in registers for the whole CTA: the plane
//     fragments in both kernels and the B fragments in the query pass.
//   - No atomics; every sum has a fixed order, so results are bitwise
//     reproducible.
#include "race_fwd_tc.h"

#include <mma.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "race_common.cuh"
#include "race_internal.cuh"

namespace race {
namespace {

namespace wmma = nvcuda::wmma;

using detail::bf16;
using detail::dispatch_shape;
using detail::kThreads;
using detail::kWarps;
using detail::token_row_offset;
using detail::workspace_slice_offset;

// wmma tile edge: m = n = k = 16.
constexpr int kFrag = 16;
// Tokens per stage. Two row tiles, so the projection and the query output
// split into 8 fragments' worth of work for the 8 warps (see TcDims).
constexpr int kTcStageTokens = 32;
constexpr int kStageRowTiles = kTcStageTokens / kFrag;
// Row padding: 16 bytes for bf16 rows, 16 bytes for fp32 rows. Keeps every
// leading dimension a multiple of 16 bytes and shifts consecutive rows by 4
// banks, which spreads the 16 row reads of a fragment load.
constexpr int kBf16RowPad = 8;
constexpr int kF32RowPad = 4;
constexpr int kSmemRegionAlign = 128;
constexpr int kStaticSmemLimit = 48 * 1024;
// An explicit minimum of one resident block per SM. With only a thread count
// in __launch_bounds__, ptxas applies its own occupancy heuristic and, under
// the torch build flags, caps a few instantiations at 64 registers with 4 to
// 24 bytes of spills on sm_80, sm_89 and sm_90. Residency is decided by
// shared memory and the register-resident fragments anyway (2 CTAs per SM at
// P = 4, L = 4, D = 128).
constexpr int kMinBlocksPerSm = 1;

static_assert(kThreads == 256 && kWarps == 8, "the warp mappings below assume 8 warps");
static_assert(kBuildTileTokens % kTcStageTokens == 0 && kQueryTileTokens % kTcStageTokens == 0,
              "tiles must split into whole stages");

__host__ __device__ constexpr int round_up(int n, int multiple) {
    return (n + multiple - 1) / multiple * multiple;
}

__host__ __device__ constexpr int next_pow2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

__host__ __device__ constexpr int max_of(int a, int b) { return a > b ? a : b; }

__host__ __device__ constexpr int ceil_div_c(int a, int b) { return (a + b - 1) / b; }

using FragRowA = wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, bf16, wmma::row_major>;
using FragColA = wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, bf16, wmma::col_major>;
using FragRowB = wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, bf16, wmma::row_major>;
using FragColB = wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, bf16, wmma::col_major>;
using FragAcc = wmma::fragment<wmma::accumulator, kFrag, kFrag, kFrag, float>;

// Compile-time shape of one (D, P, L) instantiation and the warp mappings.
template <int D, int P, int L>
struct TcDims {
    static_assert(D == 64 || D == 128, "head_dim must be 64 or 128");
    static_assert(P >= 1 && P <= detail::kMaxPlanes, "num_planes must be 1..5");
    static_assert(L >= 1 && L <= detail::kMaxTables, "num_tables must be 1..4");

    static constexpr int kCorners = 1 << P;                  // R
    static constexpr int kStackedCorners = L * kCorners;     // L * R
    static constexpr int kCornerTiles = round_up(kStackedCorners, kFrag) / kFrag;
    // Columns of a Phi row. A power of two >= 16 and >= L * R, so the corner
    // step can give every thread one fixed column (kThreads is a multiple of
    // it); columns >= L * R hold 0.
    static constexpr int kCornerSlots = next_pow2(max_of(kStackedCorners, kFrag));
    static constexpr int kPhiStride = kCornerSlots + kBf16RowPad;
    static constexpr int kPhiRowsPerPass = kThreads / kCornerSlots;
    static constexpr int kPhiPasses = kTcStageTokens / kPhiRowsPerPass;

    static constexpr int kProjections = L * P;  // stacked plane index l * P + t
    static constexpr int kProjColTiles = round_up(kProjections, kFrag) / kFrag;
    static constexpr int kProjCols = kProjColTiles * kFrag;  // NP

    static constexpr int kDimTiles = D / kFrag;
    static constexpr int kRowStride = D + kBf16RowPad;  // bf16 rows of X, V and staged W, B
    static constexpr int kF32RowStride = D + kF32RowPad;

    // Projection: kStageRowTiles * kProjColTiles output fragments (2 or 4),
    // each split over kProjParts contiguous ranges of k-steps so all 8 warps
    // work. Warp w computes fragment w % kProjFrags over k-part
    // w / kProjFrags and stores its partial; the hash step adds the parts in
    // part order.
    static constexpr int kProjFrags = kStageRowTiles * kProjColTiles;
    static constexpr int kProjParts = kWarps / kProjFrags;
    static constexpr int kProjStepsPerPart = kDimTiles / kProjParts;

    // Bucket build and query output: warp w owns column tile w % kDimTiles
    // and the row tiles (w / kDimTiles) + kRowGroups * i. For D = 128 that is
    // one column tile and every row tile; for D = 64 two warps share a column
    // tile and take alternate row tiles.
    static constexpr int kRowGroups = kWarps / kDimTiles;
    static constexpr int kBuildAccPerWarp = ceil_div_c(kCornerTiles, kRowGroups);
    static constexpr int kQueryAccPerWarp = kStageRowTiles / kRowGroups;
    static constexpr int kSliceFloats = detail::Dims<D, P>::kSliceFloats;  // R * (D + 1)

    static_assert(kWarps % kProjFrags == 0 && kDimTiles % kProjParts == 0,
                  "projection work must split evenly over the warps");
    static_assert(kWarps % kDimTiles == 0 && kStageRowTiles % kRowGroups == 0,
                  "output tiles must split evenly over the warps");
    static_assert(kThreads % kCornerSlots == 0 && kTcStageTokens % kPhiRowsPerPass == 0,
                  "the corner step must cover the stage exactly");
};

// Byte offsets of the shared-memory regions, each 128-byte aligned.
//
// Bucket build:
//   keys    [32][D + 8] bf16          the stage's keys (single buffer)
//   values  [2][32][D + 8] bf16       double-buffered values
//   work    union of                  (lifetimes do not overlap, see the loop)
//             projection partials [kProjParts][32][NP] fp32
//             Phi                 [32][kPhiStride] bf16
//             split planes        [2][16][D + 8] bf16 (prologue)
//             B staging           [16 * kRowGroups][D + 4] fp32 (epilogue)
//             mass partials       [kThreads] fp32 (epilogue)
//   probs   [32][L * P] float2        (sigmoid(2 beta u), sigmoid(-2 beta u))
template <int D, int P, int L>
struct BuildTcSmem {
    using T = TcDims<D, P, L>;
    static constexpr int kKeys = 0;
    static constexpr int kKeysBytes = kTcStageTokens * T::kRowStride * 2;
    static constexpr int kValues = round_up(kKeys + kKeysBytes, kSmemRegionAlign);
    static constexpr int kValuesBufferElems = kTcStageTokens * T::kRowStride;
    static constexpr int kValuesBytes = 2 * kValuesBufferElems * 2;
    static constexpr int kWork = round_up(kValues + kValuesBytes, kSmemRegionAlign);
    static constexpr int kWorkBytes =
        max_of(max_of(T::kProjParts * kTcStageTokens * T::kProjCols * 4,
                      kTcStageTokens * T::kPhiStride * 2),
               max_of(max_of(2 * kFrag * T::kRowStride * 2,
                             kFrag * T::kRowGroups * T::kF32RowStride * 4),
                      kThreads * 4));
    static constexpr int kProbs = round_up(kWork + kWorkBytes, kSmemRegionAlign);
    static constexpr int kProbsBytes = kTcStageTokens * T::kProjections * 8;
    static constexpr int kBytes = round_up(kProbs + kProbsBytes, kSmemRegionAlign);
    static_assert(kBytes <= kStaticSmemLimit, "bucket build exceeds static shared memory");
};

// Query pass:
//   queries [32][D + 8] bf16          the stage's queries (single buffer)
//   work    union of
//             projection partials [kProjParts][32][NP] fp32
//             Phi_Q               [32][kPhiStride] bf16
//             split planes / B    [2][16][D + 8] bf16 (prologue)
//   numer   [32][D + 4] fp32          Num of the stage, read back row-wise
//   probs   [32][L * P] float2
//   mass    [kCornerSlots] fp32       A, zero past L * R
//   inv_den [32] fp32
template <int D, int P, int L>
struct QueryTcSmem {
    using T = TcDims<D, P, L>;
    static constexpr int kQueries = 0;
    static constexpr int kQueriesBytes = kTcStageTokens * T::kRowStride * 2;
    static constexpr int kWork = round_up(kQueries + kQueriesBytes, kSmemRegionAlign);
    static constexpr int kWorkBytes =
        max_of(max_of(T::kProjParts * kTcStageTokens * T::kProjCols * 4,
                      kTcStageTokens * T::kPhiStride * 2),
               2 * kFrag * T::kRowStride * 2);
    static constexpr int kNumer = round_up(kWork + kWorkBytes, kSmemRegionAlign);
    static constexpr int kNumerBytes = kTcStageTokens * T::kF32RowStride * 4;
    static constexpr int kProbs = round_up(kNumer + kNumerBytes, kSmemRegionAlign);
    static constexpr int kProbsBytes = kTcStageTokens * T::kProjections * 8;
    static constexpr int kMass = round_up(kProbs + kProbsBytes, kSmemRegionAlign);
    static constexpr int kMassBytes = T::kCornerSlots * 4;
    static constexpr int kInvDen = round_up(kMass + kMassBytes, kSmemRegionAlign);
    static constexpr int kInvDenBytes = kTcStageTokens * 4;
    static constexpr int kBytes = round_up(kInvDen + kInvDenBytes, kSmemRegionAlign);
    static_assert(kBytes <= kStaticSmemLimit, "query pass exceeds static shared memory");
};

// ---------------------------------------------------------------------------
// Asynchronous row copies
// ---------------------------------------------------------------------------

// 16-byte global -> shared copy that bypasses registers (sm_80 cp.async).
// With valid = false nothing is read and the 16 bytes are zero-filled; src
// must still be a valid address.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    const int src_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src),
                 "r"(src_bytes)
                 : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Issues the copies of the stage's 32 rows (tokens stage_begin ..) into
// rows_s, row stride D + 8. Rows at or past token_end are zero-filled, so
// tail tokens hash to finite values and contribute exact zeros to V.
// Requires stage_begin < token_end (a stage never starts past the end).
template <int D>
__device__ __forceinline__ void copy_stage_rows_async(const bf16* __restrict__ rows, bf16* rows_s,
                                                      int stage_begin, int token_end) {
    constexpr int kChunksPerRow = D * 2 / 16;
    constexpr int kRowStride = D + kBf16RowPad;
    for (int idx = threadIdx.x; idx < kTcStageTokens * kChunksPerRow; idx += kThreads) {
        const int row = idx / kChunksPerRow;
        const int chunk = idx % kChunksPerRow;
        const int token = stage_begin + row;
        const bool valid = token < token_end;
        const bf16* src = rows + static_cast<size_t>(valid ? token : stage_begin) * D + chunk * 8;
        cp_async_16(rows_s + row * kRowStride + chunk * 8, src, valid);
    }
}

// ---------------------------------------------------------------------------
// Operand preparation (prologues)
// ---------------------------------------------------------------------------

// Writes rows [first_row, first_row + 16) of a [num_rows x D] fp32 matrix to
// shared memory as hi = bf16(x) and lo = bf16(x - hi), row stride D + 8.
// Rows >= num_rows are zero. x - hi is exact in fp32, so hi + lo is x with a
// relative error of at most 2^-16. row_ptr(g) returns row g's first element.
template <int D, typename RowPtrFn>
__device__ __forceinline__ void split_rows_to_bf16(RowPtrFn&& row_ptr, int first_row,
                                                   int num_rows, bf16* hi_s, bf16* lo_s) {
    constexpr int kRowStride = D + kBf16RowPad;
    for (int idx = threadIdx.x; idx < kFrag * D; idx += kThreads) {
        const int row = idx / D;
        const int col = idx % D;
        const int source_row = first_row + row;
        const float x = source_row < num_rows ? row_ptr(source_row)[col] : 0.0f;
        const bf16 hi = __float2bfloat16_rn(x);
        hi_s[row * kRowStride + col] = hi;
        lo_s[row * kRowStride + col] = __float2bfloat16_rn(x - __bfloat162float(hi));
    }
}

// The plane fragments of one warp for the whole CTA: W^T as the col-major B
// operand of Z = X W^T, for this warp's column tile and k-part.
template <int D, int P, int L>
struct PlaneFragments {
    FragColB hi[TcDims<D, P, L>::kProjStepsPerPart];
    FragColB lo[TcDims<D, P, L>::kProjStepsPerPart];
};

// Loads the warp's plane fragments. planes is [L * P][D] fp32 (the [L][P][D]
// tensor, stacked plane index l * P + t); rows past L * P are zero. staging
// must hold 2 * 16 * (D + 8) bf16 and is free again when this returns.
// Called by all threads (contains barriers).
template <int D, int P, int L>
__device__ __forceinline__ void load_plane_fragments(const float* __restrict__ planes,
                                                     bf16* staging, int warp,
                                                     PlaneFragments<D, P, L>& frags) {
    using T = TcDims<D, P, L>;
    const int frag = warp % T::kProjFrags;
    const int col_tile = frag / kStageRowTiles;
    const int part = warp / T::kProjFrags;
    bf16* hi_s = staging;
    bf16* lo_s = staging + kFrag * T::kRowStride;
#pragma unroll
    for (int tile = 0; tile < T::kProjColTiles; ++tile) {
        split_rows_to_bf16<D>([&](int row) { return planes + static_cast<size_t>(row) * D; },
                              tile * kFrag, T::kProjections, hi_s, lo_s);
        __syncthreads();
        if (col_tile == tile) {  // warp-uniform
#pragma unroll
            for (int j = 0; j < T::kProjStepsPerPart; ++j) {
                // Element (k = d, n = plane) of W^T is staged at [plane][d],
                // which is W^T in column-major order with ldm = D + 8.
                const int k_step = part * T::kProjStepsPerPart + j;
                wmma::load_matrix_sync(frags.hi[j], hi_s + k_step * kFrag, T::kRowStride);
                wmma::load_matrix_sync(frags.lo[j], lo_s + k_step * kFrag, T::kRowStride);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Hashing of one stage, shared by both kernels
// ---------------------------------------------------------------------------

// Z = X W^T for the stage's 32 rows. Warp w computes rows of row tile
// (w % kProjFrags) % 2, columns of column tile (w % kProjFrags) / 2, over the
// k-steps of part w / kProjFrags, and stores that partial fragment to
// partials[part][32][NP]. hi and lo go into the same accumulator, hi first.
template <int D, int P, int L>
__device__ __forceinline__ void project_stage(const bf16* rows_s,
                                              const PlaneFragments<D, P, L>& planes, float* partials,
                                              int warp) {
    using T = TcDims<D, P, L>;
    const int frag = warp % T::kProjFrags;
    const int row_tile = frag % kStageRowTiles;
    const int col_tile = frag / kStageRowTiles;
    const int part = warp / T::kProjFrags;

    FragAcc z;
    wmma::fill_fragment(z, 0.0f);
#pragma unroll
    for (int j = 0; j < T::kProjStepsPerPart; ++j) {
        const int k_step = part * T::kProjStepsPerPart + j;
        FragRowA x;
        wmma::load_matrix_sync(x, rows_s + row_tile * kFrag * T::kRowStride + k_step * kFrag,
                               T::kRowStride);
        wmma::mma_sync(z, x, planes.hi[j], z);
        wmma::mma_sync(z, x, planes.lo[j], z);
    }
    float* out = partials + part * kTcStageTokens * T::kProjCols +
                 row_tile * kFrag * T::kProjCols + col_tile * kFrag;
    wmma::store_matrix_sync(out, z, T::kProjCols, wmma::mem_row_major);
}

// probs[token][l * P + t] = (sigmoid(2 beta u), sigmoid(-2 beta u)) with
// u = tanh(z) and z the sum of the k-part partials in part order.
template <int D, int P, int L>
__device__ __forceinline__ void hash_stage(const float* partials, float2* probs, float beta) {
    using T = TcDims<D, P, L>;
    for (int entry = threadIdx.x; entry < kTcStageTokens * T::kProjections; entry += kThreads) {
        const int token = entry / T::kProjections;
        const int plane = entry % T::kProjections;
        float z = 0.0f;
#pragma unroll
        for (int part = 0; part < T::kProjParts; ++part) {
            z += partials[(part * kTcStageTokens + token) * T::kProjCols + plane];
        }
        float prob_plus, prob_minus;
        sigmoid_pair(2.0f * beta * tanhf(z), prob_plus, prob_minus);
        probs[entry] = make_float2(prob_plus, prob_minus);
    }
}

// phi[token][slot] = bf16 corner probability of stacked corner slot = l * R + r
// for tokens < valid_rows, and 0 for padding slots (>= L * R) and for rows at
// or past valid_rows (tail tokens). Thread t always handles slot
// t % kCornerSlots, for tokens t / kCornerSlots + kPhiRowsPerPass * i, so
// with kAccumulateMass it adds its rounded values to a single running mass.
template <int D, int P, int L, bool kAccumulateMass>
__device__ __forceinline__ void corner_stage(const float2* probs, bf16* phi, int valid_rows,
                                             float& mass) {
    using T = TcDims<D, P, L>;
    const int slot = threadIdx.x % T::kCornerSlots;
    const int first_row = threadIdx.x / T::kCornerSlots;
    const bool live_slot = slot < T::kStackedCorners;
    const int table = slot / T::kCorners;
    const int corner = slot % T::kCorners;
#pragma unroll
    for (int pass = 0; pass < T::kPhiPasses; ++pass) {
        const int token = first_row + pass * T::kPhiRowsPerPass;
        float value = 0.0f;
        if (live_slot && token < valid_rows) {
            float prob_plus[P];
            float prob_minus[P];
#pragma unroll
            for (int t = 0; t < P; ++t) {
                const float2 pair = probs[token * T::kProjections + table * P + t];
                prob_plus[t] = pair.x;
                prob_minus[t] = pair.y;
            }
            value = bernoulli_product<P>(prob_plus, prob_minus, corner);
        }
        const bf16 rounded = __float2bfloat16_rn(value);
        phi[token * T::kPhiStride + slot] = rounded;
        if constexpr (kAccumulateMass) mass += __bfloat162float(rounded);
    }
}

// ---------------------------------------------------------------------------
// Kernel 1 (tensor cores): bucket build
// ---------------------------------------------------------------------------

// Grid: x = build tiles, y = batch_heads. One CTA hashes every table of its
// 2048 keys, so K and V are read once. Writes the same per-(tile, head, table)
// slices as bucket_build_kernel.
template <int D, int P, int L>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm)
    bucket_build_tc_kernel(const bf16* __restrict__ keys, const bf16* __restrict__ values,
                           const float* __restrict__ planes, const float* __restrict__ beta_ptr,
                           float* __restrict__ workspace, int seq_len, int batch_heads) {
    using T = TcDims<D, P, L>;
    using Layout = BuildTcSmem<D, P, L>;
    __shared__ __align__(kSmemRegionAlign) unsigned char smem[Layout::kBytes];
    bf16* keys_s = reinterpret_cast<bf16*>(smem + Layout::kKeys);
    bf16* values_s = reinterpret_cast<bf16*>(smem + Layout::kValues);
    unsigned char* work = smem + Layout::kWork;
    float* partials_s = reinterpret_cast<float*>(work);
    bf16* phi_s = reinterpret_cast<bf16*>(work);
    float2* probs_s = reinterpret_cast<float2*>(smem + Layout::kProbs);

    const int tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int tile_begin = tile * kBuildTileTokens;
    const int tile_end = min(seq_len, tile_begin + kBuildTileTokens);
    const int num_stages = ceil_div_c(tile_end - tile_begin, kTcStageTokens);

    const bf16* key_rows = keys + token_row_offset(bh, 0, seq_len, D);
    const bf16* value_rows = values + token_row_offset(bh, 0, seq_len, D);

    // Stage 0 is in flight while the planes are split.
    copy_stage_rows_async<D>(key_rows, keys_s, tile_begin, tile_end);
    copy_stage_rows_async<D>(value_rows, values_s, tile_begin, tile_end);
    cp_async_commit();

    PlaneFragments<D, P, L> plane_frags;
    load_plane_fragments<D, P, L>(planes, reinterpret_cast<bf16*>(work), warp, plane_frags);
    const float beta = *beta_ptr;

    // Warp w accumulates B rows of row tiles row_group + kRowGroups * i and
    // columns [16 col_tile, 16 col_tile + 16) for the whole tile.
    const int col_tile = warp % T::kDimTiles;
    const int row_group = warp / T::kDimTiles;
    FragAcc acc[T::kBuildAccPerWarp];
#pragma unroll
    for (int i = 0; i < T::kBuildAccPerWarp; ++i) wmma::fill_fragment(acc[i], 0.0f);
    float mass = 0.0f;  // slot threadIdx.x % kCornerSlots, summed over this thread's tokens

    // Invariant at the top of iteration s: the copies of stage s into keys_s
    // and values buffer s & 1 were issued, and every read of the previous
    // stage has happened before the barrier below. Region lifetimes within an
    // iteration: keys_s is read only by the projection (so stage s + 1 may
    // overwrite it after the second barrier), the work region holds the
    // projection partials until the third barrier and Phi after it, and
    // values buffer (s + 1) & 1 was last read in iteration s - 1.
    for (int s = 0; s < num_stages; ++s) {
        const int stage_begin = tile_begin + s * kTcStageTokens;
        const bf16* stage_values = values_s + (s & 1) * Layout::kValuesBufferElems;
        cp_async_wait_all();
        __syncthreads();

        project_stage<D, P, L>(keys_s, plane_frags, partials_s, warp);
        __syncthreads();

        if (s + 1 < num_stages) {
            const int next_begin = stage_begin + kTcStageTokens;
            copy_stage_rows_async<D>(key_rows, keys_s, next_begin, tile_end);
            copy_stage_rows_async<D>(value_rows,
                                     values_s + ((s + 1) & 1) * Layout::kValuesBufferElems,
                                     next_begin, tile_end);
        }
        cp_async_commit();

        hash_stage<D, P, L>(partials_s, probs_s, beta);
        __syncthreads();

        corner_stage<D, P, L, true>(probs_s, phi_s, tile_end - stage_begin, mass);
        __syncthreads();

        // B += Phi^T V over the stage's two k-steps. Phi is stored [token][slot]
        // row-major, which is Phi^T [slot][token] in column-major order.
#pragma unroll
        for (int k_step = 0; k_step < kStageRowTiles; ++k_step) {
            FragRowB value_frag;
            wmma::load_matrix_sync(value_frag,
                                   stage_values + k_step * kFrag * T::kRowStride + col_tile * kFrag,
                                   T::kRowStride);
#pragma unroll
            for (int i = 0; i < T::kBuildAccPerWarp; ++i) {
                const int row_tile = row_group + T::kRowGroups * i;
                if (row_tile < T::kCornerTiles) {  // warp-uniform
                    FragColA phi_frag;
                    wmma::load_matrix_sync(phi_frag,
                                           phi_s + k_step * kFrag * T::kPhiStride + row_tile * kFrag,
                                           T::kPhiStride);
                    wmma::mma_sync(acc[i], phi_frag, value_frag, acc[i]);
                }
            }
        }
    }
    __syncthreads();  // the work region is free: every MMA read of Phi is done

    // A: thread t holds slot t % kCornerSlots summed over its tokens; the
    // kPhiRowsPerPass threads of a slot are added in thread order.
    float* mass_partials_s = reinterpret_cast<float*>(work);
    mass_partials_s[threadIdx.x] = mass;
    __syncthreads();
    if (threadIdx.x < T::kStackedCorners) {
        const int slot = threadIdx.x;
        float total = 0.0f;
#pragma unroll
        for (int group = 0; group < T::kPhiRowsPerPass; ++group) {
            total += mass_partials_s[group * T::kCornerSlots + slot];
        }
        float* slice = workspace + workspace_slice_offset(tile, bh, slot / T::kCorners,
                                                          batch_heads, L, T::kSliceFloats);
        slice[T::kCorners * D + slot % T::kCorners] = total;
    }
    __syncthreads();

    // B: round i stages row tiles kRowGroups * i .. kRowGroups * i + kRowGroups - 1
    // (one per row group) through shared memory, then all threads copy the
    // rows to their table slices. A 16-row tile can span tables when R < 16,
    // and slices are not 32-byte aligned in general, so fragments are not
    // stored to the workspace directly.
    float* staging_s = reinterpret_cast<float*>(work);
    constexpr int kRoundRows = kFrag * T::kRowGroups;
#pragma unroll
    for (int i = 0; i < T::kBuildAccPerWarp; ++i) {
        const int row_tile = row_group + T::kRowGroups * i;
        if (row_tile < T::kCornerTiles) {  // warp-uniform
            wmma::store_matrix_sync(staging_s + row_group * kFrag * T::kF32RowStride + col_tile * kFrag,
                                    acc[i], T::kF32RowStride, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < kRoundRows * D; idx += kThreads) {
            const int row = idx / D;
            const int col = idx % D;
            const int slot = kRoundRows * i + row;
            if (slot < T::kStackedCorners) {
                float* slice = workspace + workspace_slice_offset(tile, bh, slot / T::kCorners,
                                                                  batch_heads, L, T::kSliceFloats);
                slice[(slot % T::kCorners) * D + col] = staging_s[row * T::kF32RowStride + col];
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 3 (tensor cores): query pass
// ---------------------------------------------------------------------------

// The B fragments of one warp for the whole CTA: B split into hi and lo as
// the row-major B operand of Num = Phi_Q B, for the warp's column tile and
// every k-step (16 stacked corners each).
template <int D, int P, int L>
struct BucketFragments {
    FragRowB hi[TcDims<D, P, L>::kCornerTiles];
    FragRowB lo[TcDims<D, P, L>::kCornerTiles];
};

// Rounds 8 fp32 values to bf16 and stores them with one 16-byte store.
__device__ __forceinline__ void store_bf16x8(bf16* dst, const float (&src)[8]) {
    uint4 raw;
    const __nv_bfloat162 pairs[4] = {
        __floats2bfloat162_rn(src[0], src[1]), __floats2bfloat162_rn(src[2], src[3]),
        __floats2bfloat162_rn(src[4], src[5]), __floats2bfloat162_rn(src[6], src[7])};
    memcpy(&raw, pairs, sizeof(raw));
    *reinterpret_cast<uint4*>(dst) = raw;
}

// Grid: x = query tiles, y = batch_heads. `totals` is workspace tile 0.
template <int D, int P, int L>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm)
    query_tc_kernel(const bf16* __restrict__ queries, const float* __restrict__ planes,
                    const float* __restrict__ beta_ptr, const float* __restrict__ totals,
                    bf16* __restrict__ out, int seq_len) {
    using T = TcDims<D, P, L>;
    using Layout = QueryTcSmem<D, P, L>;
    __shared__ __align__(kSmemRegionAlign) unsigned char smem[Layout::kBytes];
    bf16* queries_s = reinterpret_cast<bf16*>(smem + Layout::kQueries);
    unsigned char* work = smem + Layout::kWork;
    float* partials_s = reinterpret_cast<float*>(work);
    bf16* phi_s = reinterpret_cast<bf16*>(work);
    float* numer_s = reinterpret_cast<float*>(smem + Layout::kNumer);
    float2* probs_s = reinterpret_cast<float2*>(smem + Layout::kProbs);
    float* mass_s = reinterpret_cast<float*>(smem + Layout::kMass);
    float* inv_den_s = reinterpret_cast<float*>(smem + Layout::kInvDen);

    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int tile_begin = blockIdx.x * kQueryTileTokens;
    const int tile_end = min(seq_len, tile_begin + kQueryTileTokens);
    const int num_stages = ceil_div_c(tile_end - tile_begin, kTcStageTokens);

    const bf16* query_rows = queries + token_row_offset(bh, 0, seq_len, D);
    bf16* out_rows = out + token_row_offset(bh, 0, seq_len, D);

    // Stage 0 is in flight while the constant operands are prepared.
    copy_stage_rows_async<D>(query_rows, queries_s, tile_begin, tile_end);
    cp_async_commit();

    bf16* split_hi_s = reinterpret_cast<bf16*>(work);
    bf16* split_lo_s = split_hi_s + kFrag * T::kRowStride;
    PlaneFragments<D, P, L> plane_frags;
    load_plane_fragments<D, P, L>(planes, split_hi_s, warp, plane_frags);

    // The L slices of this head are contiguous in the workspace (bh-major);
    // stacked corner slot = l * R + r is row r of table l's B.
    const float* head_totals = totals + static_cast<size_t>(bh) * L * T::kSliceFloats;
    const auto bucket_row = [&](int slot) {
        return head_totals + (slot / T::kCorners) * T::kSliceFloats + (slot % T::kCorners) * D;
    };
    const int col_tile = warp % T::kDimTiles;
    const int row_group = warp / T::kDimTiles;
    BucketFragments<D, P, L> bucket_frags;
#pragma unroll
    for (int k_step = 0; k_step < T::kCornerTiles; ++k_step) {
        split_rows_to_bf16<D>(bucket_row, k_step * kFrag, T::kStackedCorners, split_hi_s,
                              split_lo_s);
        __syncthreads();
        wmma::load_matrix_sync(bucket_frags.hi[k_step], split_hi_s + col_tile * kFrag,
                               T::kRowStride);
        wmma::load_matrix_sync(bucket_frags.lo[k_step], split_lo_s + col_tile * kFrag,
                               T::kRowStride);
        __syncthreads();
    }
    // Read after the first barrier of the stage loop.
    for (int slot = threadIdx.x; slot < T::kCornerSlots; slot += kThreads) {
        mass_s[slot] = slot < T::kStackedCorners
                           ? head_totals[(slot / T::kCorners) * T::kSliceFloats +
                                         T::kCorners * D + slot % T::kCorners]
                           : 0.0f;
    }
    const float beta = *beta_ptr;

    // Same stage invariant as the bucket build: at the top of iteration s the
    // copies of stage s into queries_s were issued and every read of the
    // previous stage (including the output reads of numer_s and inv_den_s)
    // happens before the barrier below.
    for (int s = 0; s < num_stages; ++s) {
        const int stage_begin = tile_begin + s * kTcStageTokens;
        cp_async_wait_all();
        __syncthreads();

        project_stage<D, P, L>(queries_s, plane_frags, partials_s, warp);
        __syncthreads();

        if (s + 1 < num_stages) {
            copy_stage_rows_async<D>(query_rows, queries_s, stage_begin + kTcStageTokens, tile_end);
        }
        cp_async_commit();

        hash_stage<D, P, L>(partials_s, probs_s, beta);
        __syncthreads();

        float unused_mass = 0.0f;
        corner_stage<D, P, L, false>(probs_s, phi_s, tile_end - stage_begin, unused_mass);
        __syncthreads();

        // Num = Phi_Q (B_hi + B_lo) for this warp's output fragments.
#pragma unroll
        for (int i = 0; i < T::kQueryAccPerWarp; ++i) {
            const int row_tile = row_group + T::kRowGroups * i;
            FragAcc numer;
            wmma::fill_fragment(numer, 0.0f);
#pragma unroll
            for (int k_step = 0; k_step < T::kCornerTiles; ++k_step) {
                FragRowA phi_frag;
                wmma::load_matrix_sync(phi_frag,
                                       phi_s + row_tile * kFrag * T::kPhiStride + k_step * kFrag,
                                       T::kPhiStride);
                wmma::mma_sync(numer, phi_frag, bucket_frags.hi[k_step], numer);
                wmma::mma_sync(numer, phi_frag, bucket_frags.lo[k_step], numer);
            }
            wmma::store_matrix_sync(numer_s + row_tile * kFrag * T::kF32RowStride + col_tile * kFrag,
                                    numer, T::kF32RowStride, wmma::mem_row_major);
        }

        // Den = Phi_Q A from the same rounded Phi_Q: warp w handles 4 tokens,
        // lane l the slots l, l + 32, ..., then a butterfly over the lanes.
        constexpr int kTokensPerWarp = kTcStageTokens / kWarps;
#pragma unroll
        for (int t = 0; t < kTokensPerWarp; ++t) {
            const int row = warp * kTokensPerWarp + t;
            float partial = 0.0f;
#pragma unroll
            for (int slot = lane; slot < T::kCornerSlots; slot += kWarpSize) {
                partial = fmaf(__bfloat162float(phi_s[row * T::kPhiStride + slot]), mass_s[slot],
                               partial);
            }
            // Den > 0 in exact arithmetic; it is 0 only after fp32 underflow,
            // when the query's buckets hold no key mass. Such rows output 0,
            // as in the fp32 path. NaN propagates.
            const float den = warp_allreduce_sum(partial);
            if (lane == 0) inv_den_s[row] = den == 0.0f ? 0.0f : 1.0f / den;
        }
        __syncthreads();

        // O = Num / Den as bf16, 8 columns (16 bytes) per thread and step.
        constexpr int kChunksPerRow = D / 8;
        for (int idx = threadIdx.x; idx < kTcStageTokens * kChunksPerRow; idx += kThreads) {
            const int row = idx / kChunksPerRow;
            const int chunk = idx % kChunksPerRow;
            const int token = stage_begin + row;
            if (token < tile_end) {
                float result[8];
                load_f32_vec<8>(numer_s + row * T::kF32RowStride + chunk * 8, result);
                const float inv_den = inv_den_s[row];
#pragma unroll
                for (int j = 0; j < 8; ++j) result[j] *= inv_den;
                store_bf16x8(out_rows + static_cast<size_t>(token) * D + chunk * 8, result);
            }
        }
    }
}

// Calls fn(integral_constant<D>, integral_constant<P>, integral_constant<L>)
// for the runtime shape.
template <typename Fn>
cudaError_t dispatch_shape_tables(const ForwardShape& shape, Fn&& fn) {
    return dispatch_shape(shape, [&](auto dim, auto planes_count) -> cudaError_t {
        switch (shape.num_tables) {
            case 1: return fn(dim, planes_count, std::integral_constant<int, 1>{});
            case 2: return fn(dim, planes_count, std::integral_constant<int, 2>{});
            case 3: return fn(dim, planes_count, std::integral_constant<int, 3>{});
            case 4: return fn(dim, planes_count, std::integral_constant<int, 4>{});
            default: return cudaErrorInvalidValue;
        }
    });
}

bool is_tc_aligned(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % kTensorCoreAlignmentBytes == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

cudaError_t race_bucket_build_tc(const bf16* k, const bf16* v, const float* planes,
                                 const float* beta, float* workspace, const ForwardShape& shape,
                                 cudaStream_t stream) {
    if (!is_supported(shape) || !is_tc_aligned(k) || !is_tc_aligned(v)) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(num_build_tiles(shape), shape.batch_heads);
    return dispatch_shape_tables(shape, [&](auto dim, auto planes_count, auto tables) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        constexpr int L = decltype(tables)::value;
        bucket_build_tc_kernel<D, P, L><<<grid, kThreads, 0, stream>>>(
            k, v, planes, beta, workspace, shape.seq_len, shape.batch_heads);
        return cudaGetLastError();
    });
}

cudaError_t race_query_tc(const bf16* q, const float* planes, const float* beta,
                          const float* workspace, bf16* out, const ForwardShape& shape,
                          cudaStream_t stream) {
    if (!is_supported(shape) || !is_tc_aligned(q) || !is_tc_aligned(out)) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(ceil_div_c(shape.seq_len, kQueryTileTokens), shape.batch_heads);
    return dispatch_shape_tables(shape, [&](auto dim, auto planes_count, auto tables) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        constexpr int L = decltype(tables)::value;
        query_tc_kernel<D, P, L><<<grid, kThreads, 0, stream>>>(q, planes, beta, workspace, out,
                                                                shape.seq_len);
        return cudaGetLastError();
    });
}

cudaError_t race_forward_tc(const bf16* q, const bf16* k, const bf16* v, const float* planes,
                            const float* beta, float* workspace, bf16* out,
                            const ForwardShape& shape, cudaStream_t stream) {
    cudaError_t err = race_bucket_build_tc(k, v, planes, beta, workspace, shape, stream);
    if (err != cudaSuccess) return err;
    err = race_bucket_reduce(workspace, shape, stream);
    if (err != cudaSuccess) return err;
    return race_query_tc(q, planes, beta, workspace, out, shape, stream);
}

}  // namespace race
