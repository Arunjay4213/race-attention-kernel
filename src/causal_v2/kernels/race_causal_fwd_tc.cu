// Chunk-parallel causal RACE Attention forward, v2b: the tile-sum (K1) and
// output (K3) kernels with their matrix products on bf16 tensor cores
// (nvcuda::wmma, m16n16k16, fp32 accumulation). K2 (the tile scan), the
// workspace layout and the public launchers are shared with v2a in
// race_causal_fwd.cu. Design: docs/causal_v2_design.md sections 4 and 6.3.
//
// Per sub-chunk of C tokens, with Phi_Q, Phi_K (C x S), V (C x D) and the
// state (A, B) from before the sub-chunk, the output pass computes
//     U        = X W_hi^T + X W_lo^T         tensor cores, X = Q and K rows
//     Phi      = product-form corners of tanh(U), rounded to bf16
//     G        = Phi_Q Phi_K^T               tensor cores, 16 x 16 tiles on or
//                                            below the diagonal only
//     G_hi, G_lo = bf16 hi / lo of tril(G)   mask and split on CUDA cores
//     Num      = Phi_Q (B_hi + B_lo) + (G_hi + G_lo) V    tensor cores
//     Den      = Phi_Q . A + rowsum(G_hi + G_lo)          CUDA cores
//     O        = Num * (1 / Den), rounded to bf16
//     B       += Phi_K^T V                   tensor cores, fp32 master in
//                                            accumulator fragments
//     A       += colsum(Phi_K)               CUDA cores
// The tile-sum kernel runs the K side of the same loop (projection, Phi_K,
// the B and A updates) from a zero state and writes the tile's (A, B).
//
// Precision (design section 6.3). X is exact in bf16; W is split into
// W_hi = bf16(W) and W_lo = bf16(W - W_hi), so W_hi + W_lo carries W to
// 2^-16 (a single bf16 W gives up to 1.7% row error at beta = 1). Phi is
// rounded to bf16 (unit roundoff 2^-8); that is the accepted v2b error. G is
// split into hi + lo like W (V is exact in bf16, so G V costs two MMAs, not
// three): with G rounded to bf16 alone, as the design planned, the first rows
// of a sequence, whose queries see few keys, reached 2.74 F on an A10G
// (d = 64, P = 4, L = 4, T = 2047, beta = 1/8; an fp64 emulation of the
// same rounding points gives the same 2.74 F, with G's rounding the largest
// term), above the 2.5 F bound; with the split the emulation gives 1.00 F.
// Every weight that multiplies a value in Num is the same number that Den
// sums: Den adds G_hi + G_lo (exact in fp32: 16 significant bits), and A
// sums the same bf16 Phi_K that multiplies V in B. With
// RACE_CAUSAL_PRECISE_CARRY the carry uses B_hi + B_lo, so the output stays
// a convex combination of values to fp32 accuracy and a constant V comes
// back exactly. W is split in each CTA's prologue: it is at most
// L P D = 2560 floats, read from L2.
//
// L is a template parameter here (it is a runtime argument in v2a): with it
// every stride and fragment loop bound is a compile-time constant, so the
// fragment loops unroll and the address arithmetic folds. In a runtime-L
// build, integer address arithmetic was ~40% of the instructions of the Num
// phase. wmma fragment loads from shared memory compile to generic 32-bit
// loads (not ldmatrix), and a row-major matrix_b or col-major matrix_a
// operand costs extra movmatrix transposes, so the layouts below keep the
// most-loaded operands in the layout the instruction wants.
//
// Shared memory of the output pass (bf16 unless noted; every region starts
// 32-byte aligned, which wmma requires of fragment pointers):
//   x_q, x_k   [C][XS]           staged Q and K rows, overwritten in place by
//                                Phi_Q and Phi_K (XS = max(D, S16) + 8, S16 = S
//                                rounded up to 16, feature padding zero)
//   x_v        [C][D + 8]        staged V rows
//   planes_hi, planes_lo [LP16][D + 8]  W_hi and W_lo, rows past L P zero
//   gram_hi, gram_lo [C/16 (C/16 + 1) / 2][16][24]  the lower-triangle
//                                tiles of G_hi and G_lo, tile ti (ti + 1) / 2
//                                + tj holding rows 16 ti.., keys 16 tj..
//   carry_hi, carry_lo [D][S16 + 8]    bf16 copies of B, transposed (column c
//                                of B is row c), so the carry reads them as
//                                col-major matrix_b without transposes
//                                (carry_lo only with PRECISE_CARRY)
//   mass       fp32 [S16]        A
//   scratch    fp32 [warps][16][16]  one 16 x 16 tile per warp; in the
//                                projection step the same bytes hold the
//                                U tiles, fp32 [sides][C][LP16 + 4]
// Every bf16 stride is 8 mod 16 elements: a multiple of 16 bytes (the wmma
// ldm rule) whose word count is an odd multiple of 4, so 8 rows read at the
// same column fall in 8 distinct 4-bank groups.
//
// Fragment layouts are opaque, so element-wise work (tanh and corners, the
// mask and rounding of G, the division, the hi/lo split) goes through the
// warp's own scratch: store_matrix_sync, __syncwarp, lanes work on it,
// __syncwarp. In that step lane l owns scratch row l / 2, columns
// 8 (l % 2) .. + 7, so it reads two float4 and writes one 16-byte vector
// (the carry copies read a column segment instead, see store_carry_copies).
//
// Launch shape (TcLaunch). Each K3 CTA walks its sub-chunks through
// barrier-separated phases whose bottlenecks differ (the Phi phase is
// CUDA-core work, the Num phase tensor-core work), so a second resident CTA
// overlaps its phases with the first one's; that was worth more than a
// larger CTA when measured on an A10G (K3 3.06 ms with 2 CTAs of 8 warps at
// C = 32 against 3.61 ms with 1 CTA of 16 warps at C = 64, d = 64, P = 4,
// L = 4, T = 2^18, 8 streams; before the G split). So the output pass uses
// 2 CTAs of 8 warps per SM where that footprint fits the smallest target
// (sm_86 and sm_89: 100 KiB of shared memory per SM, 1 KiB reserved per
// CTA), else 1 CTA of 16 warps, and C = 64 only where it still fits that
// residency; for every (d, P) that is C = 32 (8 warps for d = 64, P <= 4;
// 16 warps otherwise). The tile-sum kernel has a small footprint and uses 8
// warps, or 16 where a warp would otherwise hold more than 4 state tiles
// (d = 128, P = 5). Both cap registers at 128 per thread (512 threads per SM
// from one kernel).
//
// Warp mapping (w = warp index, W warps). Every wmma call is made by whole
// warps under warp-uniform conditions; no barrier is under divergence.
//   projection       : unit = (side, 16-row tile), 2 C / 16 units (C / 16 in
//                      K1, K side only) dealt to warps 0, 1, ...; each writes
//                      its U tile, then a barrier
//   Phi              : thread p takes (side p / (C L), table (p / C) % L,
//                      token p % C), so a warp is 32 consecutive tokens of one
//                      (side, table)
//   G                : the C/16 (C/16 + 1) / 2 lower-triangle tiles dealt
//                      round-robin
//   Num + output     : (C / 16) (D / 16) tiles dealt round-robin in rounds of
//                      W; with two rounds the row tiles of the odd round are
//                      taken in reverse, so each warp's causal intra work is
//                      balanced (C = 64, D = 64, W = 8: warp w gets row tiles
//                      w / 4 and 3 - w / 4). A warp's tiles share one column
//                      tile, so each carry and V fragment is loaded once per
//                      step. The two lanes of a row also form Den for it.
//   state update     : warp w owns state tiles w, w + W, ... of the
//                      (S16 / 16) x (D / 16) grid for the whole CTA; the fp32
//                      master of B stays in those accumulator fragments, and
//                      they share one column tile (one V fragment per step)
//   A update         : a power-of-two group of consecutive threads per
//                      feature, butterfly-combined
// Barriers per sub-chunk: 6 in the output pass (after the rows are staged,
// after U, after Phi, after G, after Num, after the update), 4 in K1. The
// next sub-chunk's rows are loaded into registers right after the first
// barrier, so their latency overlaps the sub-chunk's compute (except where
// registers are short, see TcTraits::prefetch_rows).
#include <mma.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "race_causal_internal.cuh"

namespace race {
namespace causal {
namespace tc {
namespace {

namespace wmma = nvcuda::wmma;
using detail::bf16;
using detail::kMaxTables;
using detail::workspace_slice_offset;

constexpr int kFrag = 16;  // wmma M = N = K
// Floats per row of a warp's 16 x 16 scratch tile (a multiple of 4, the wmma
// ldm rule for fp32). Unpadded: the lane mapping of the element-wise steps
// (row lane / 2, 8 columns) has 2-way bank conflicts with a padded stride of
// 20 as well, and the 2 KiB saved over 8 warps is what lets the output pass
// of d = 64 keep 2 CTAs per SM with the G split.
constexpr int kScratchStride = kFrag;
// bf16 elements per row of a compact G tile: 8 mod 16, so the 8 rows of a
// fragment load fall in 8 distinct 4-bank groups.
constexpr int kGramTileStride = kFrag + 8;
// Threads per SM that each kernel's register cap is sized for (128 registers
// per thread): 2 CTAs of 8 warps or 1 CTA of 16 warps.
constexpr int kThreadsPerSm = 512;
// The smallest shared memory per SM among the targets (sm_86 and sm_89) and
// the part the driver reserves per resident CTA.
constexpr size_t kPortableSmemPerSm = 100 * 1024;
constexpr size_t kReservedSmemPerCta = 1024;

using FragRowA = wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, bf16, wmma::row_major>;
using FragColA = wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, bf16, wmma::col_major>;
using FragRowB = wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, bf16, wmma::row_major>;
using FragColB = wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, bf16, wmma::col_major>;
using FragAcc = wmma::fragment<wmma::accumulator, kFrag, kFrag, kFrag, float>;

constexpr bool kPreciseCarry = RACE_CAUSAL_PRECISE_CARRY != 0;

__host__ __device__ constexpr int round_up16(int n) { return (n + 15) & ~15; }
__host__ __device__ constexpr size_t align32(size_t bytes) { return (bytes + 31) & ~size_t{31}; }
__host__ __device__ constexpr int max_int(int a, int b) { return a > b ? a : b; }
__host__ __device__ constexpr int min_int(int a, int b) { return a < b ? a : b; }
__host__ __device__ constexpr int ceil_div_int(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ constexpr int pow2_floor_int(int x) {
    int p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

// ---------------------------------------------------------------------------
// Shared-memory layout and launch shape
// ---------------------------------------------------------------------------

// Byte offsets of the shared-memory regions (see the header comment).
// Regions a kernel does not use have size 0: the tile-sum kernel
// (emit_output = false) has no x_q, G tiles or carry copies.
struct TcSmemLayout {
    size_t x_q, x_k, x_v, planes_hi, planes_lo, gram_hi, gram_lo, carry_hi, carry_lo, mass, scratch, total;
};

__host__ __device__ constexpr TcSmemLayout tc_smem_layout(int head_dim, int num_planes,
                                                          int num_tables, int chunk, int warps,
                                                          bool emit_output, bool precise_carry) {
    const size_t bf16_bytes = sizeof(bf16);
    const int features_padded = round_up16(num_tables << num_planes);
    const int plane_rows_padded = round_up16(num_tables * num_planes);
    const size_t x_stride = static_cast<size_t>(max_int(head_dim, features_padded)) + 8;
    const size_t value_stride = static_cast<size_t>(head_dim) + 8;
    const size_t x_bytes = chunk * x_stride * bf16_bytes;
    const size_t planes_bytes = plane_rows_padded * value_stride * bf16_bytes;
    const size_t carry_bytes = static_cast<size_t>(head_dim) * (features_padded + 8) * bf16_bytes;

    TcSmemLayout layout{};
    size_t cursor = 0;
    layout.x_q = cursor;
    cursor += emit_output ? align32(x_bytes) : 0;
    layout.x_k = cursor;
    cursor += align32(x_bytes);
    layout.x_v = cursor;
    cursor += align32(chunk * value_stride * bf16_bytes);
    layout.planes_hi = cursor;
    cursor += align32(planes_bytes);
    layout.planes_lo = cursor;
    cursor += align32(planes_bytes);
    const int gram_tiles = (chunk / kFrag) * (chunk / kFrag + 1) / 2;
    const size_t gram_bytes = static_cast<size_t>(gram_tiles) * kFrag * kGramTileStride * bf16_bytes;
    layout.gram_hi = cursor;
    cursor += emit_output ? align32(gram_bytes) : 0;
    layout.gram_lo = cursor;
    cursor += emit_output ? align32(gram_bytes) : 0;
    layout.carry_hi = cursor;
    cursor += emit_output ? align32(carry_bytes) : 0;
    layout.carry_lo = cursor;
    cursor += emit_output && precise_carry ? align32(carry_bytes) : 0;
    layout.mass = cursor;
    cursor += align32(static_cast<size_t>(features_padded) * sizeof(float));
    layout.scratch = cursor;
    cursor += align32(static_cast<size_t>(warps) * kFrag * kScratchStride * sizeof(float));
    layout.total = cursor;
    return layout;
}

// Warps per CTA and sub-chunk length of one kernel.
struct TcLaunch {
    int warps;
    int chunk;
};

// The output pass's launch shape for (D, P), from its L = 4 footprint (see
// the header comment): the first of (8 warps, C = 64), (8, 32) whose
// footprint lets 2 CTAs share the smallest SM, else the first of (16, 64),
// (16, 32) that fits one CTA there. It does not depend on L or on the GPU,
// so a shape gets the same summation order everywhere.
__host__ __device__ constexpr TcLaunch output_pass_launch(int head_dim, int num_planes) {
    const size_t two_ctas = kPortableSmemPerSm / 2 - kReservedSmemPerCta;
    for (int chunk = 64; chunk >= 32; chunk /= 2) {
        if (tc_smem_layout(head_dim, num_planes, kMaxTables, chunk, 8, true, kPreciseCarry).total <= two_ctas) {
            return TcLaunch{8, chunk};
        }
    }
    const bool fits_64 = tc_smem_layout(head_dim, num_planes, kMaxTables, 64, 16, true, kPreciseCarry).total <=
                         kPortableSmemBytes;
    return TcLaunch{16, fits_64 ? 64 : 32};
}

// The tile-sum kernel: the output pass's C, and 8 warps unless that gives a
// warp more than 4 state tiles at L = 4 (32 accumulator registers; d = 128,
// P = 5 would need 8 and spill at 128 registers), then 16. Its footprint
// has no Q rows, G or carry copies, so it fits one CTA everywhere and two
// where the rows are small.
__host__ __device__ constexpr TcLaunch tile_sums_launch(int head_dim, int num_planes) {
    const int state_tiles = round_up16(kMaxTables << num_planes) / kFrag * (head_dim / kFrag);
    return TcLaunch{state_tiles <= 4 * 8 ? 8 : 16, output_pass_launch(head_dim, num_planes).chunk};
}

// Sizes of one kernel instantiation (see the header comment).
template <int D_, int P_, int L_, int C_, int kWarps_>
struct TcTraits {
    static constexpr int D = D_;
    static constexpr int P = P_;
    static constexpr int L = L_;
    static constexpr int C = C_;
    static constexpr int kWarps = kWarps_;
    static constexpr int kThreads = kWarps * kWarpSize;
    static constexpr int kMinBlocks = kThreadsPerSm / kThreads;  // for __launch_bounds__

    static constexpr int kCorners = 1 << P;
    static constexpr int kFeatures = L * kCorners;                    // S
    static constexpr int kFeaturesPadded = round_up16(kFeatures);     // S16
    static constexpr int kFeatureTiles = kFeaturesPadded / kFrag;
    static constexpr int kPlaneRows = L * P;
    static constexpr int kPlaneRowsPadded = round_up16(kPlaneRows);  // LP16
    static constexpr int kPlaneTiles = kPlaneRowsPadded / kFrag;
    static constexpr int kRowTiles = C / kFrag;  // 16-token tiles per sub-chunk
    static constexpr int kColTiles = D / kFrag;  // 16-column tiles of V, B, O
    static constexpr int kXStride = max_int(D, kFeaturesPadded) + 8;
    static constexpr int kValueStride = D + 8;                // x_v and planes rows
    static constexpr int kCarryStride = kFeaturesPadded + 8;  // carry_hi / lo rows
    static constexpr int kUStride = kPlaneRowsPadded + 4;  // floats per U row
    static constexpr int kStateTiles = kFeatureTiles * kColTiles;
    static constexpr int kStateTilesPerWarp = ceil_div_int(kStateTiles, kWarps);
    static constexpr int kOutputTiles = kRowTiles * kColTiles;
    static constexpr int kOutputRounds = ceil_div_int(kOutputTiles, kWarps);
    static constexpr int kGramTiles = kRowTiles * (kRowTiles + 1) / 2;
    static constexpr int kGramTileElems = kFrag * kGramTileStride;
    static constexpr int kGramRounds = ceil_div_int(kGramTiles, kWarps);
    static constexpr int kMassGroup = min_int(pow2_floor_int(kThreads / kFeaturesPadded), kWarpSize);
    // Row staging: vector it * kThreads + tid of a tensor is row
    // vector / (D / 8), columns 8 (vector % (D / 8)) .. + 7; threads past the
    // C D / 8 vectors stage nothing.
    static constexpr int kVectorsPerRow = D / 8;
    static constexpr int kVectors = C * kVectorsPerRow;
    static constexpr int kLoadIters = ceil_div_int(kVectors, kThreads);

    // Whether a kernel staging `tensors` token tensors holds the next
    // sub-chunk's rows in registers during the current one. The resident
    // state tiles (8 registers each) and the staged rows (4 per vector) must
    // stay within 32 registers; beyond that the prefetch spills at the
    // 128-register cap (d = 128, P = 5, L >= 3 in the output pass), so those
    // rows are loaded at the top of each sub-chunk instead.
    __host__ __device__ static constexpr bool prefetch_rows(int tensors) {
        return kStateTilesPerWarp * 8 + tensors * kLoadIters * 4 <= 32;
    }

    static_assert(D == 64 || D == 128, "head_dim must be 64 or 128");
    static_assert(C == 32 || C == 64, "sub-chunk must be 32 or 64 tokens");
    static_assert(L >= 1 && L <= kMaxTables, "1..4 tables");
    static_assert(kWarps == 8 || kWarps == 16, "8 or 16 warps");
    static_assert(kPlaneTiles <= 2, "L P <= 20 fits two 16-column tiles");
    static_assert(kWarps % kColTiles == 0, "a warp's output and state tiles share one column tile");
    static_assert(C % kMassGroup == 0, "the A update splits the tokens evenly over a group");
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// 8 floats rounded to bf16 and packed into one 16-byte vector.
__device__ __forceinline__ uint4 pack_bf16x8(const float (&values)[8]) {
    uint32_t words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 pair = __floats2bfloat162_rn(values[2 * i], values[2 * i + 1]);
        memcpy(&words[i], &pair, sizeof(pair));
    }
    return make_uint4(words[0], words[1], words[2], words[3]);
}

// Stores 8 bf16 to shared memory (dst 16-byte aligned). The compiler proves
// the alignment from the aligned shared base, so this is one STS.128.
__device__ __forceinline__ void store_bf16x8(bf16* dst, const float (&values)[8]) {
    *reinterpret_cast<uint4*>(dst) = pack_bf16x8(values);
}

// Stores 8 bf16 of output to global memory (dst 16-byte aligned). A plain
// uint4 store built from four words is split into four 32-bit stores when
// the compiler cannot prove the alignment of a global pointer, and the
// partial sectors then cost 4x the DRAM writes (measured on an A10G:
// 1.12 GB written for a 268 MB output); __stcs is one st.global.cs.v4, and
// the streaming hint fits an output that is written once and not re-read.
__device__ __forceinline__ void store_output_bf16x8(bf16* dst, const float (&values)[8]) {
    __stcs(reinterpret_cast<uint4*>(dst), pack_bf16x8(values));
}

// The 8 floats of a lane's share of a 16 x 16 scratch tile: row lane / 2,
// columns 8 (lane % 2) .. + 7. Both float4 accesses are 16-byte aligned
// because the scratch stride is a multiple of 4 floats.
__device__ __forceinline__ void load_scratch_row8(const float* scratch, int lane, float (&values)[8]) {
    const float* src = scratch + (lane / 2) * kScratchStride + (lane % 2) * 8;
    const float4 low = *reinterpret_cast<const float4*>(src);
    const float4 high = *reinterpret_cast<const float4*>(src + 4);
    values[0] = low.x;
    values[1] = low.y;
    values[2] = low.z;
    values[3] = low.w;
    values[4] = high.x;
    values[5] = high.y;
    values[6] = high.z;
    values[7] = high.w;
}

__device__ __forceinline__ void store_scratch_row8(float* scratch, int lane, const float (&values)[8]) {
    float* dst = scratch + (lane / 2) * kScratchStride + (lane % 2) * 8;
    *reinterpret_cast<float4*>(dst) = make_float4(values[0], values[1], values[2], values[3]);
    *reinterpret_cast<float4*>(dst + 4) = make_float4(values[4], values[5], values[6], values[7]);
}

// Adds fragment `other` into `acc` element by element. Two accumulators of
// the same shape hold element (i, j) at the same x[] index, so this is
// layout-independent.
__device__ __forceinline__ void add_fragment(FragAcc& acc, const FragAcc& other) {
#pragma unroll
    for (int i = 0; i < acc.num_elements; ++i) acc.x[i] += other.x[i];
}

// Stores the R corner probabilities of one (token, table) as bf16 at dst
// (2 R contiguous bytes, aligned to 16 bytes or to 2 R if smaller: the row
// stride is a multiple of 16 bytes and the table offset is t R elements);
// keep = false stores zeros. Corner r has plane t positive iff bit
// (P - 1 - t) of r is set. A product tree over planes 0 .. P - 2 gives the
// R / 2 prefixes (level t doubles the list, entry r going to 2 r for plane t
// negative and 2 r + 1 for positive), and the last plane is expanded while
// the values are packed, so at most R / 2 + 8 corner values are live. Every
// corner is the product ((p_0 p_1) p_2) ... in plane order, bitwise equal
// to bernoulli_product, with about 2 R multiplies instead of R P. The loops
// have constant trip counts (the tree's inner one is guarded), so they
// unroll fully and the arrays stay in registers; a trip count that depended
// on t left them in local memory.
template <int P>
__device__ __forceinline__ void store_corners(bf16* dst, const float (&prob_plus)[P],
                                              const float (&prob_minus)[P], bool keep) {
    constexpr int kCorners = 1 << P;
    constexpr int kPrefixes = kCorners / 2;
    float prefix[kPrefixes];
    prefix[0] = 1.0f;  // P = 1: the last plane alone; 1.0f * p is exact
#pragma unroll
    for (int t = 0; t < P - 1; ++t) {
#pragma unroll
        for (int r = kPrefixes / 2 - 1; r >= 0; --r) {  // descending: entry r is read before 2 r is written
            if (r < (1 << t)) {
                const float parent = t == 0 ? 1.0f : prefix[r];
                prefix[2 * r + 1] = t == 0 ? prob_plus[0] : parent * prob_plus[t];
                prefix[2 * r] = t == 0 ? prob_minus[0] : parent * prob_minus[t];
            }
        }
    }
    constexpr int kPerStore = kCorners >= 8 ? 8 : kCorners;  // bf16 values per store
#pragma unroll
    for (int base = 0; base < kCorners; base += kPerStore) {
        uint32_t words[kPerStore / 2];
#pragma unroll
        for (int w = 0; w < kPerStore / 2; ++w) {
            const float head = prefix[base / 2 + w];
            const float negative = keep ? (P == 1 ? prob_minus[P - 1] : head * prob_minus[P - 1]) : 0.0f;
            const float positive = keep ? (P == 1 ? prob_plus[P - 1] : head * prob_plus[P - 1]) : 0.0f;
            const __nv_bfloat162 pair = __floats2bfloat162_rn(negative, positive);
            memcpy(&words[w], &pair, sizeof(pair));
        }
        if constexpr (kPerStore == 8) {
            *reinterpret_cast<uint4*>(dst + base) = make_uint4(words[0], words[1], words[2], words[3]);
        } else if constexpr (kPerStore == 4) {
            *reinterpret_cast<uint2*>(dst + base) = make_uint2(words[0], words[1]);
        } else {
            *reinterpret_cast<uint32_t*>(dst + base) = words[0];
        }
    }
}

// Maps state feature s (row of B) to its workspace offsets inside the L
// slices of one (tile, stream): slice l = s / R holds B[r][c] at r D + c and
// A[r] at R D + r, with r = s % R.
template <int D, int P>
__device__ __forceinline__ int workspace_row_offset(int s) {
    constexpr int kCorners = 1 << P;
    return (s >> P) * kCorners * (D + 1) + (s & (kCorners - 1)) * D;
}

template <int D, int P>
__device__ __forceinline__ int workspace_mass_offset(int s) {
    constexpr int kCorners = 1 << P;
    return (s >> P) * kCorners * (D + 1) + kCorners * D + (s & (kCorners - 1));
}

// ---------------------------------------------------------------------------
// Sub-chunk rows: global -> registers -> shared memory
// ---------------------------------------------------------------------------

// Loads the C rows of kTensors token tensors starting at element `offset`
// into registers in the row-staging order of the traits, so a warp reads
// whole 128-byte row segments. Rows at or past rows_valid are zero: V must
// be finite there because 0 * NaN would poison B, and X = 0 keeps Phi finite.
template <class Traits, int kTensors>
__device__ __forceinline__ void fetch_rows(const bf16* const (&sources)[kTensors], size_t offset,
                                           int rows_valid, uint4 (&staged)[kTensors][Traits::kLoadIters]) {
#pragma unroll
    for (int t = 0; t < kTensors; ++t) {
#pragma unroll
        for (int it = 0; it < Traits::kLoadIters; ++it) {
            const int vector = it * Traits::kThreads + threadIdx.x;
            const int row = vector / Traits::kVectorsPerRow;
            const int col = (vector % Traits::kVectorsPerRow) * 8;
            staged[t][it] = make_uint4(0u, 0u, 0u, 0u);
            if (vector < Traits::kVectors && row < rows_valid) {
                staged[t][it] = *reinterpret_cast<const uint4*>(
                    sources[t] + offset + static_cast<size_t>(row) * Traits::D + col);
            }
        }
    }
}

// Stores the staged rows into shared memory with row stride strides[t]
// elements (a multiple of 8, so every 16-byte vector store is aligned).
template <class Traits, int kTensors>
__device__ __forceinline__ void store_rows(const uint4 (&staged)[kTensors][Traits::kLoadIters],
                                           bf16* const (&destinations)[kTensors],
                                           const int (&strides)[kTensors]) {
#pragma unroll
    for (int t = 0; t < kTensors; ++t) {
#pragma unroll
        for (int it = 0; it < Traits::kLoadIters; ++it) {
            const int vector = it * Traits::kThreads + threadIdx.x;
            const int row = vector / Traits::kVectorsPerRow;
            const int col = (vector % Traits::kVectorsPerRow) * 8;
            if (vector < Traits::kVectors) {
                *reinterpret_cast<uint4*>(destinations[t] + row * strides[t] + col) = staged[t][it];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// State pieces
// ---------------------------------------------------------------------------

// W (fp32 [L P][D], plane t of table l in row l P + t) split into bf16 hi
// and lo rows of stride D + 8; rows L P .. LP16 - 1 are zero, so the padded
// projection columns come out as 0 and are never read.
template <class Traits>
__device__ __forceinline__ void load_split_planes(const float* __restrict__ planes, bf16* planes_hi,
                                                  bf16* planes_lo) {
    constexpr int D = Traits::D;
    for (int i = threadIdx.x; i < Traits::kPlaneRowsPadded * D; i += Traits::kThreads) {
        const int row = i / D;
        const int col = i - row * D;
        const float w = row < Traits::kPlaneRows ? planes[i] : 0.0f;
        const bf16 high = __float2bfloat16_rn(w);
        planes_hi[row * Traits::kValueStride + col] = high;
        planes_lo[row * Traits::kValueStride + col] = __float2bfloat16_rn(w - __bfloat162float(high));
    }
}

// Writes the bf16 carry copies of state tile (fs, tc): hi = bf16(B) and, with
// the precise carry, lo = bf16(B - hi) (B - hi is exact in fp32, so hi + lo
// carries ~16 significant bits). The copies are transposed: lane
// (c = lane % 16, half = lane / 16) reads B[16 fs + 8 half .. + 7][16 tc + c],
// a column segment of the row-major scratch tile, and writes it as one
// 16-byte vector into carry row 16 tc + c. A column-major store_matrix_sync
// would let the lanes read rows instead, but with the unpadded scratch
// stride its 8 x 4 element pattern is a 4-way bank conflict; the row-major
// store and the strided reads (2-way) made the output pass 6% faster on an
// A10G (3.04 against 3.23 ms, d = 64, P = 4, L = 4, T = 2^18, 8 streams).
template <class Traits, bool kWithLow>
__device__ __forceinline__ void store_carry_copies(const FragAcc& tile, int fs, int tc, bf16* carry_hi,
                                                   bf16* carry_lo, float* scratch, int lane) {
    wmma::store_matrix_sync(scratch, tile, kScratchStride, wmma::mem_row_major);
    __syncwarp();
    const int column = lane % kFrag;
    const int half = lane / kFrag;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) values[i] = scratch[(half * 8 + i) * kScratchStride + column];
    float high[8];
    float low[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        high[i] = __bfloat162float(__float2bfloat16_rn(values[i]));
        low[i] = values[i] - high[i];
    }
    const int offset = (tc * kFrag + column) * Traits::kCarryStride + fs * kFrag + half * 8;
    store_bf16x8(carry_hi + offset, high);  // exact: high[] already holds bf16 values
    if constexpr (kWithLow) store_bf16x8(carry_lo + offset, low);
    __syncwarp();  // scratch reads are done before the warp's next store to it
}

// ---------------------------------------------------------------------------
// Sub-chunk phases
// ---------------------------------------------------------------------------

// Projection units: unit u = (side u / (C / 16), row tile u % (C / 16))
// computes U = X W_hi^T + X W_lo^T for its 16 rows into U row
// side * C + token of u_tiles (fp32, stride LP16 + 4), with the hi and lo
// products as two independent accumulator chains added at the end. Units are
// dealt to warps 0, 1, ...; the other warps have no projection work.
// x[side] are the staged rows (stride XS); side kSides - 1 is the key side.
template <class Traits, int kSides>
__device__ __forceinline__ void project_rows(const bf16* const (&x)[kSides], const bf16* planes_hi,
                                             const bf16* planes_lo, float* u_tiles, int warp) {
    static_assert(kSides * Traits::C * Traits::kUStride <= Traits::kWarps * kFrag * kScratchStride,
                  "the U tiles fit in the scratch region");
    for (int unit = warp; unit < kSides * Traits::kRowTiles; unit += Traits::kWarps) {
        const int side = unit / Traits::kRowTiles;
        const int row_tile = unit - side * Traits::kRowTiles;
        const bf16* rows = (side == 0 ? x[0] : x[kSides - 1]) + row_tile * kFrag * Traits::kXStride;

        FragAcc high[Traits::kPlaneTiles];
        FragAcc low[Traits::kPlaneTiles];
#pragma unroll
        for (int pt = 0; pt < Traits::kPlaneTiles; ++pt) {
            wmma::fill_fragment(high[pt], 0.0f);
            wmma::fill_fragment(low[pt], 0.0f);
        }
#pragma unroll
        for (int kt = 0; kt < Traits::kColTiles; ++kt) {
            FragRowA x_tile;
            wmma::load_matrix_sync(x_tile, rows + kt * kFrag, Traits::kXStride);
#pragma unroll
            for (int pt = 0; pt < Traits::kPlaneTiles; ++pt) {
                // planes [plane][dim] read as col_major is the (dim x plane) operand W^T.
                const int offset = pt * kFrag * Traits::kValueStride + kt * kFrag;
                FragColB w_hi;
                FragColB w_lo;
                wmma::load_matrix_sync(w_hi, planes_hi + offset, Traits::kValueStride);
                wmma::load_matrix_sync(w_lo, planes_lo + offset, Traits::kValueStride);
                wmma::mma_sync(high[pt], x_tile, w_hi, high[pt]);
                wmma::mma_sync(low[pt], x_tile, w_lo, low[pt]);
            }
        }
        float* u_rows = u_tiles + unit * kFrag * Traits::kUStride;
#pragma unroll
        for (int pt = 0; pt < Traits::kPlaneTiles; ++pt) {
            add_fragment(high[pt], low[pt]);
            wmma::store_matrix_sync(u_rows + pt * kFrag, high[pt], Traits::kUStride, wmma::mem_row_major);
        }
    }
}

// Phi for every (side, token, table) of the sub-chunk from the U tiles,
// written as bf16 over the staged rows of that side (stride XS). The rows
// are dead: every projection read them before the barrier that precedes this
// step. Thread p takes side p / (C L), table (p / C) % L, token p % C. Key
// rows at or past rows_valid get Phi = 0 so they add nothing to G or the
// state, and feature columns S .. S16 - 1 are zeroed for the padded tiles.
// beta is read here (an L1 hit) rather than held in a register across the
// sub-chunk loop, which is one register fewer at the 128-register cap.
template <class Traits, int kSides>
__device__ __forceinline__ void compute_features(bf16* const (&phi)[kSides], const float* u_tiles,
                                                 const float* __restrict__ beta_ptr, int rows_valid) {
    const float beta = *beta_ptr;
    constexpr int P = Traits::P;
    constexpr int C = Traits::C;
    constexpr int kCorners = Traits::kCorners;
    constexpr int kPairsPerSide = C * Traits::L;
    for (int pair = threadIdx.x; pair < kSides * kPairsPerSide; pair += Traits::kThreads) {
        const int side = pair / kPairsPerSide;
        const int table = (pair - side * kPairsPerSide) / C;
        const int token = pair % C;
        const float* u = u_tiles + (side * C + token) * Traits::kUStride + table * P;
        float prob_plus[P];
        float prob_minus[P];
#pragma unroll
        for (int t = 0; t < P; ++t) sigmoid_pair(2.0f * beta * tanhf(u[t]), prob_plus[t], prob_minus[t]);
        const bool keep = side != kSides - 1 || token < rows_valid;
        bf16* rows = side == 0 ? phi[0] : phi[kSides - 1];
        store_corners<P>(rows + token * Traits::kXStride + table * kCorners, prob_plus, prob_minus, keep);
    }
    if constexpr (Traits::kFeaturesPadded > Traits::kFeatures) {
        constexpr int kPad = Traits::kFeaturesPadded - Traits::kFeatures;
        for (int e = threadIdx.x; e < kSides * C * kPad; e += Traits::kThreads) {
            const int side = e / (C * kPad);
            const int token = (e - side * C * kPad) / kPad;
            const int col = Traits::kFeatures + e % kPad;
            (side == 0 ? phi[0] : phi[kSides - 1])[token * Traits::kXStride + col] = __float2bfloat16_rn(0.0f);
        }
    }
}

// Lower-triangle tile index t = ti (ti + 1) / 2 + tj -> (ti, tj), tj <= ti.
__device__ __forceinline__ void gram_tile_coords(int t, int& ti, int& tj) {
    ti = 0;
    while ((ti + 1) * (ti + 2) / 2 <= t) ++ti;
    tj = t - ti * (ti + 1) / 2;
}

// G tiles on or below the diagonal: tril(Phi_Q Phi_K^T) split into
// G_hi = bf16(G) and G_lo = bf16(G - G_hi), written as compact tile t of
// gram_hi and gram_lo. Masked entries (and keys past the end, whose Phi_K is
// 0) are exact zeros in both. Tile t goes to warp t % W; a warp's tiles
// accumulate as independent chains.
template <class Traits>
__device__ __forceinline__ void gram_tiles(const bf16* phi_q, const bf16* phi_k, bf16* gram_hi,
                                           bf16* gram_lo, float* scratch, int warp, int lane) {
    constexpr int kRounds = Traits::kGramRounds;
    int tile_row[kRounds];
    int tile_col[kRounds];
    FragAcc acc[kRounds];
#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
        gram_tile_coords(warp + round * Traits::kWarps, tile_row[round], tile_col[round]);
        wmma::fill_fragment(acc[round], 0.0f);
    }
#pragma unroll
    for (int ft = 0; ft < Traits::kFeatureTiles; ++ft) {
#pragma unroll
        for (int round = 0; round < kRounds; ++round) {
            if (warp + round * Traits::kWarps < Traits::kGramTiles) {  // warp-uniform
                FragRowA query;
                FragColB key;  // Phi_K [key][feature] read as col_major is Phi_K^T
                wmma::load_matrix_sync(query, phi_q + tile_row[round] * kFrag * Traits::kXStride + ft * kFrag,
                                       Traits::kXStride);
                wmma::load_matrix_sync(key, phi_k + tile_col[round] * kFrag * Traits::kXStride + ft * kFrag,
                                       Traits::kXStride);
                wmma::mma_sync(acc[round], query, key, acc[round]);
            }
        }
    }
#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
        if (warp + round * Traits::kWarps >= Traits::kGramTiles) break;  // warp-uniform
        wmma::store_matrix_sync(scratch, acc[round], kScratchStride, wmma::mem_row_major);
        __syncwarp();
        const int row = lane / 2;
        const int col = (lane % 2) * 8;
        const int diagonal_shift = (tile_row[round] - tile_col[round]) * kFrag;  // i - j = row - col + shift
        float values[8];
        load_scratch_row8(scratch, lane, values);
        float high[8];
        float low[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float g = col + c <= row + diagonal_shift ? values[c] : 0.0f;
            high[c] = __bfloat162float(__float2bfloat16_rn(g));
            low[c] = g - high[c];  // exact in fp32
        }
        const int offset = (warp + round * Traits::kWarps) * Traits::kGramTileElems + row * kGramTileStride + col;
        store_bf16x8(gram_hi + offset, high);
        store_bf16x8(gram_lo + offset, low);
        __syncwarp();
    }
}

// Output tile of round k for warp w (see the header comment): linear tile
// w + W k in row-major (row tile, column tile) order, with the rows of odd
// rounds mirrored. Because W is a multiple of D / 16, the column tile is
// w % (D / 16) in every round. Warps whose round-0 tile is past the last
// tile (C = 32, D = 64, W = 16: warps 8..15) have no tile.
template <class Traits>
__device__ __forceinline__ int output_row_tile(int warp, int round) {
    constexpr int kRowsPerRound = Traits::kWarps / Traits::kColTiles;
    const int ti = (warp + round * Traits::kWarps) / Traits::kColTiles;
    const int first = round * kRowsPerRound;
    return round % 2 == 1 ? 2 * first + kRowsPerRound - 1 - ti : ti;
}

// Den_i = Phi_Q[i] . A + sum_{j <= i} (G_hi + G_lo)[i][j] for row i of row
// tile ti, formed by the two lanes of the row: lane half h sums features
// h S16 / 2 .. and columns 8 h .. 8 h + 7 of every G tile of the row, and
// the two halves are added (both lanes add the same two operands, so both
// hold the same Den). Keys up to the end of the diagonal tile are summed;
// the masked ones are exact zeros. G_hi + G_lo is exact in fp32, so Den sums
// exactly the weights the tensor cores apply to V. Every warp with a tile in
// row tile ti forms the same value the same way.
template <class Traits>
__device__ __forceinline__ float row_denominator(const bf16* phi_q, const float* mass, const bf16* gram_hi,
                                                 const bf16* gram_lo, int ti, int lane) {
    constexpr int kHalfFeatures = Traits::kFeaturesPadded / 2;  // a multiple of 8
    const int i = ti * kFrag + lane / 2;
    const int half = lane % 2;
    float partial = 0.0f;
    const bf16* query = phi_q + i * Traits::kXStride + half * kHalfFeatures;
    const float* masses = mass + half * kHalfFeatures;
#pragma unroll
    for (int s = 0; s < kHalfFeatures; s += 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(query + s);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float4 mass_low = *reinterpret_cast<const float4*>(masses + s);
        const float4 mass_high = *reinterpret_cast<const float4*>(masses + s + 4);
        const float weights[8] = {mass_low.x,  mass_low.y,  mass_low.z,  mass_low.w,
                                  mass_high.x, mass_high.y, mass_high.z, mass_high.w};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 q = __bfloat1622float2(pairs[p]);
            partial = fmaf(q.x, weights[2 * p], partial);
            partial = fmaf(q.y, weights[2 * p + 1], partial);
        }
    }
    const int first_tile = ti * (ti + 1) / 2;
    const int row_offset = (lane / 2) * kGramTileStride + half * 8;
    for (int tj = 0; tj <= ti; ++tj) {
        const int offset = (first_tile + tj) * Traits::kGramTileElems + row_offset;
        const uint4 raw_high = *reinterpret_cast<const uint4*>(gram_hi + offset);
        const uint4 raw_low = *reinterpret_cast<const uint4*>(gram_lo + offset);
        const __nv_bfloat162* high = reinterpret_cast<const __nv_bfloat162*>(&raw_high);
        const __nv_bfloat162* low = reinterpret_cast<const __nv_bfloat162*>(&raw_low);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 h = __bfloat1622float2(high[p]);
            const float2 l = __bfloat1622float2(low[p]);
            partial += h.x + l.x;
            partial += h.y + l.y;
        }
    }
    return partial + __shfl_xor_sync(kFullWarpMask, partial, 1);
}

// Num = Phi_Q (B_hi [+ B_lo]) + (G_hi + G_lo) V for this warp's output tiles, then
// O = Num / Den for the rows of the sequence, rounded to bf16 and stored as
// 16-byte vectors (two lanes cover one 32-byte sector of a row). The warp's
// rounds (and the hi and lo carries) are independent accumulator chains;
// they share each carry and V fragment load.
template <class Traits, bool kWithLow>
__device__ __forceinline__ void output_tiles(const bf16* phi_q, const bf16* carry_hi,
                                             const bf16* carry_lo, const bf16* gram_hi,
                                             const bf16* gram_lo, const bf16* x_v,
                                             const float* mass, int rows_valid,
                                             bf16* __restrict__ out_rows, float* scratch, int warp,
                                             int lane) {
    constexpr int kRounds = Traits::kOutputRounds;
    static_assert(kRounds <= 2, "the odd-round mirror balances at most two rounds");
    static_assert(kRounds == 1 || Traits::kOutputTiles % Traits::kWarps == 0,
                  "with two rounds every warp has a tile in both");
    if (warp >= Traits::kOutputTiles) return;  // warp-uniform; only when there are fewer tiles than warps
    const int tc = warp % Traits::kColTiles;

    int ti[kRounds];
    FragAcc acc[kRounds];
    FragAcc acc_low[kWithLow ? kRounds : 1];
#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
        ti[round] = output_row_tile<Traits>(warp, round);
        wmma::fill_fragment(acc[round], 0.0f);
        if constexpr (kWithLow) wmma::fill_fragment(acc_low[round], 0.0f);
    }
#pragma unroll
    for (int ft = 0; ft < Traits::kFeatureTiles; ++ft) {
        // carry [col][feature] read as col_major is the (feature x col) operand B.
        const int carry_offset = tc * kFrag * Traits::kCarryStride + ft * kFrag;
        FragColB high;
        wmma::load_matrix_sync(high, carry_hi + carry_offset, Traits::kCarryStride);
        FragColB low;
        if constexpr (kWithLow) wmma::load_matrix_sync(low, carry_lo + carry_offset, Traits::kCarryStride);
#pragma unroll
        for (int round = 0; round < kRounds; ++round) {
            FragRowA query;
            wmma::load_matrix_sync(query, phi_q + ti[round] * kFrag * Traits::kXStride + ft * kFrag,
                                   Traits::kXStride);
            wmma::mma_sync(acc[round], query, high, acc[round]);
            if constexpr (kWithLow) wmma::mma_sync(acc_low[round], query, low, acc_low[round]);
        }
    }
    int last_row_tile = 0;
#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
        if constexpr (kWithLow) add_fragment(acc[round], acc_low[round]);
        last_row_tile = max(last_row_tile, ti[round]);
    }
#pragma unroll
    for (int tj = 0; tj < Traits::kRowTiles; ++tj) {
        if (tj <= last_row_tile) {  // warp-uniform
            FragRowB values;
            wmma::load_matrix_sync(values, x_v + tj * kFrag * Traits::kValueStride + tc * kFrag,
                                   Traits::kValueStride);
#pragma unroll
            for (int round = 0; round < kRounds; ++round) {
                if (tj <= ti[round]) {  // warp-uniform
                    const int tile = ti[round] * (ti[round] + 1) / 2 + tj;
                    FragRowA weights;
                    wmma::load_matrix_sync(weights, gram_hi + tile * Traits::kGramTileElems, kGramTileStride);
                    wmma::mma_sync(acc[round], weights, values, acc[round]);
                    wmma::load_matrix_sync(weights, gram_lo + tile * Traits::kGramTileElems, kGramTileStride);
                    wmma::mma_sync(acc[round], weights, values, acc[round]);
                }
            }
        }
    }

#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
        const float den = row_denominator<Traits>(phi_q, mass, gram_hi, gram_lo, ti[round], lane);
        wmma::store_matrix_sync(scratch, acc[round], kScratchStride, wmma::mem_row_major);
        __syncwarp();
        const int i = ti[round] * kFrag + lane / 2;
        if (i < rows_valid) {
            const float inv_den = den == 0.0f ? 0.0f : 1.0f / den;
            float values[8];
            load_scratch_row8(scratch, lane, values);
#pragma unroll
            for (int c = 0; c < 8; ++c) values[c] *= inv_den;
            store_output_bf16x8(out_rows + static_cast<size_t>(i) * Traits::D + tc * kFrag + (lane % 2) * 8,
                                values);
        }
        __syncwarp();
    }
}

// B += Phi_K^T V into this warp's state fragments: fragment f is state tile
// idx = w + W f of the (S16 / 16) x (D / 16) grid, feature tile idx / (D / 16)
// and column tile w % (D / 16) for every f, so one V fragment per step
// serves all of them. With kEmit the updated tiles are also written as the
// carry copies the next sub-chunk reads.
template <class Traits, bool kEmit, bool kWithLow>
__device__ __forceinline__ void update_state(FragAcc (&state)[Traits::kStateTilesPerWarp], const bf16* phi_k,
                                             const bf16* x_v, bf16* carry_hi, bf16* carry_lo,
                                             float* scratch, int warp, int lane) {
    const int tc = warp % Traits::kColTiles;
#pragma unroll
    for (int kt = 0; kt < Traits::kRowTiles; ++kt) {
        FragRowB values;
        wmma::load_matrix_sync(values, x_v + kt * kFrag * Traits::kValueStride + tc * kFrag,
                               Traits::kValueStride);
#pragma unroll
        for (int f = 0; f < Traits::kStateTilesPerWarp; ++f) {
            const int idx = warp + f * Traits::kWarps;
            if (idx < Traits::kStateTiles) {  // warp-uniform
                FragColA keys;  // Phi_K [token][feature] read as col_major is Phi_K^T
                wmma::load_matrix_sync(keys, phi_k + kt * kFrag * Traits::kXStride + (idx / Traits::kColTiles) * kFrag,
                                       Traits::kXStride);
                wmma::mma_sync(state[f], keys, values, state[f]);
            }
        }
    }
    if constexpr (kEmit) {
#pragma unroll
        for (int f = 0; f < Traits::kStateTilesPerWarp; ++f) {
            const int idx = warp + f * Traits::kWarps;
            if (idx < Traits::kStateTiles) {
                store_carry_copies<Traits, kWithLow>(state[f], idx / Traits::kColTiles, tc, carry_hi, carry_lo,
                                                     scratch, lane);
            }
        }
    }
}

// A[s] += sum over the sub-chunk's tokens of bf16 Phi_K[j][s]. Thread tid
// sums tokens part, part + G, ... of feature tid / G (G = kMassGroup
// consecutive lanes per feature), then a butterfly over the G lanes gives
// each of them the same total. All threads take part in the shuffles.
template <class Traits>
__device__ __forceinline__ void accumulate_mass(const bf16* phi_k, float* mass) {
    constexpr int kGroup = Traits::kMassGroup;
    const int s = threadIdx.x / kGroup;
    const int part = threadIdx.x % kGroup;
    float sum = 0.0f;
    if (s < Traits::kFeaturesPadded) {
#pragma unroll
        for (int j = part; j < Traits::C; j += kGroup) sum += __bfloat162float(phi_k[j * Traits::kXStride + s]);
    }
#pragma unroll
    for (int offset = kGroup / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(kFullWarpMask, sum, offset);
    }
    if (part == 0 && s < Traits::kFeaturesPadded) mass[s] += sum;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// K1 (v2b): the tile's A and B from its own tokens. Grid: x = tile,
// y = stream. Writes the tile's L slices of the workspace. Workspace rows of
// B are not 32-byte aligned in general (R (D + 1) floats per slice), so the
// fragments go out through scratch with 8-byte stores.
template <class Traits>
__global__ void __launch_bounds__(Traits::kThreads, Traits::kMinBlocks)
    tile_sums_tc_kernel(const bf16* __restrict__ keys, const bf16* __restrict__ values,
                        const float* __restrict__ planes, const float* __restrict__ beta_ptr,
                        float* __restrict__ workspace, int seq_len, int batch_heads, int tile_tokens) {
    constexpr int D = Traits::D;
    constexpr int P = Traits::P;
    constexpr int C = Traits::C;
    constexpr TcSmemLayout kLayout = tc_smem_layout(D, P, Traits::L, C, Traits::kWarps, false, false);
    extern __shared__ __align__(128) unsigned char smem[];
    bf16* x_k = reinterpret_cast<bf16*>(smem + kLayout.x_k);
    bf16* x_v = reinterpret_cast<bf16*>(smem + kLayout.x_v);
    bf16* planes_hi = reinterpret_cast<bf16*>(smem + kLayout.planes_hi);
    bf16* planes_lo = reinterpret_cast<bf16*>(smem + kLayout.planes_lo);
    float* mass = reinterpret_cast<float*>(smem + kLayout.mass);
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    float* u_tiles = reinterpret_cast<float*>(smem + kLayout.scratch);
    float* scratch = u_tiles + warp * kFrag * kScratchStride;
    const bf16* const staged_keys[1] = {x_k};
    bf16* const phi_k[1] = {x_k};

    const int tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tile_begin = tile * tile_tokens;
    const int tile_end = tile_begin + min(seq_len - tile_begin, tile_tokens);
    const size_t stream_offset = static_cast<size_t>(bh) * seq_len * D;
    const bf16* const sources[2] = {keys, values};
    bf16* const destinations[2] = {x_k, x_v};
    const int strides[2] = {Traits::kXStride, Traits::kValueStride};

    constexpr bool kPrefetch = Traits::prefetch_rows(2);
    uint4 staged[2][Traits::kLoadIters];
    if constexpr (kPrefetch) {
        fetch_rows<Traits, 2>(sources, stream_offset + static_cast<size_t>(tile_begin) * D,
                              min(C, tile_end - tile_begin), staged);
    }

    load_split_planes<Traits>(planes, planes_hi, planes_lo);
    for (int s = threadIdx.x; s < Traits::kFeaturesPadded; s += Traits::kThreads) mass[s] = 0.0f;
    FragAcc state[Traits::kStateTilesPerWarp];
#pragma unroll
    for (int f = 0; f < Traits::kStateTilesPerWarp; ++f) wmma::fill_fragment(state[f], 0.0f);

    // Barrier invariant as in the output pass: a region is rewritten only
    // after the barrier that follows its last read.
    for (int chunk_begin = tile_begin; chunk_begin < tile_end; chunk_begin += C) {
        const int rows_valid = min(C, tile_end - chunk_begin);
        if constexpr (!kPrefetch) {
            fetch_rows<Traits, 2>(sources, stream_offset + static_cast<size_t>(chunk_begin) * D, rows_valid,
                                  staged);
        }
        store_rows<Traits, 2>(staged, destinations, strides);
        __syncthreads();  // rows staged (first time: planes and mass too)
        if (kPrefetch && chunk_begin + C < tile_end) {
            fetch_rows<Traits, 2>(sources, stream_offset + static_cast<size_t>(chunk_begin + C) * D,
                                  min(C, tile_end - chunk_begin - C), staged);
        }
        project_rows<Traits, 1>(staged_keys, planes_hi, planes_lo, u_tiles, warp);
        __syncthreads();  // U ready; the staged K rows are dead
        compute_features<Traits, 1>(phi_k, u_tiles, beta_ptr, rows_valid);
        __syncthreads();  // Phi_K ready; the U tiles are dead
        update_state<Traits, false, false>(state, x_k, x_v, nullptr, nullptr, scratch, warp, lane);
        accumulate_mass<Traits>(x_k, mass);
        __syncthreads();  // x_k, x_v free; mass final after the last sub-chunk
    }

    float* slices = workspace + workspace_slice_offset(tile, bh, 0, batch_heads, Traits::L, Traits::kCorners * (D + 1));
    const int tc = warp % Traits::kColTiles;
#pragma unroll
    for (int f = 0; f < Traits::kStateTilesPerWarp; ++f) {
        const int idx = warp + f * Traits::kWarps;
        if (idx < Traits::kStateTiles) {
            wmma::store_matrix_sync(scratch, state[f], kScratchStride, wmma::mem_row_major);
            __syncwarp();
            const int s = (idx / Traits::kColTiles) * kFrag + lane / 2;
            if (s < Traits::kFeatures) {
                float row[8];
                load_scratch_row8(scratch, lane, row);
                // Slices hold an even number of floats and D is even, so this
                // address is 8-byte aligned.
                float* dst = slices + workspace_row_offset<D, P>(s) + tc * kFrag + (lane % 2) * 8;
#pragma unroll
                for (int c = 0; c < 8; c += 2) *reinterpret_cast<float2*>(dst + c) = make_float2(row[c], row[c + 1]);
            }
            __syncwarp();
        }
    }
    for (int s = threadIdx.x; s < Traits::kFeatures; s += Traits::kThreads) {
        slices[workspace_mass_offset<D, P>(s)] = mass[s];
    }
}

// K3 (v2b): the outputs of one tile. Grid: x = tile, y = stream. The
// workspace holds exclusive prefixes (after K2). All loops and barriers are
// uniform across the CTA.
template <class Traits>
__global__ void __launch_bounds__(Traits::kThreads, Traits::kMinBlocks)
    output_pass_tc_kernel(const bf16* __restrict__ queries, const bf16* __restrict__ keys,
                          const bf16* __restrict__ values, const float* __restrict__ planes,
                          const float* __restrict__ beta_ptr, const float* __restrict__ workspace,
                          bf16* __restrict__ out, int seq_len, int batch_heads, int tile_tokens) {
    constexpr int D = Traits::D;
    constexpr int P = Traits::P;
    constexpr int C = Traits::C;
    constexpr TcSmemLayout kLayout = tc_smem_layout(D, P, Traits::L, C, Traits::kWarps, true, kPreciseCarry);
    extern __shared__ __align__(128) unsigned char smem[];
    bf16* x_q = reinterpret_cast<bf16*>(smem + kLayout.x_q);
    bf16* x_k = reinterpret_cast<bf16*>(smem + kLayout.x_k);
    bf16* x_v = reinterpret_cast<bf16*>(smem + kLayout.x_v);
    bf16* planes_hi = reinterpret_cast<bf16*>(smem + kLayout.planes_hi);
    bf16* planes_lo = reinterpret_cast<bf16*>(smem + kLayout.planes_lo);
    bf16* gram_hi = reinterpret_cast<bf16*>(smem + kLayout.gram_hi);
    bf16* gram_lo = reinterpret_cast<bf16*>(smem + kLayout.gram_lo);
    bf16* carry_hi = reinterpret_cast<bf16*>(smem + kLayout.carry_hi);
    bf16* carry_lo = reinterpret_cast<bf16*>(smem + kLayout.carry_lo);  // unused without the precise carry
    float* mass = reinterpret_cast<float*>(smem + kLayout.mass);
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    float* u_tiles = reinterpret_cast<float*>(smem + kLayout.scratch);
    float* scratch = u_tiles + warp * kFrag * kScratchStride;
    const bf16* const staged_rows[2] = {x_q, x_k};
    bf16* const phi[2] = {x_q, x_k};

    const int tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tile_begin = tile * tile_tokens;
    const int tile_end = tile_begin + min(seq_len - tile_begin, tile_tokens);
    const size_t stream_offset = static_cast<size_t>(bh) * seq_len * D;
    const bf16* const sources[3] = {queries, keys, values};
    bf16* const destinations[3] = {x_q, x_k, x_v};
    const int strides[3] = {Traits::kXStride, Traits::kXStride, Traits::kValueStride};

    constexpr bool kPrefetch = Traits::prefetch_rows(3);
    uint4 staged[3][Traits::kLoadIters];
    if constexpr (kPrefetch) {
        fetch_rows<Traits, 3>(sources, stream_offset + static_cast<size_t>(tile_begin) * D,
                              min(C, tile_end - tile_begin), staged);
    }

    // Prologue: W split, and this (tile, stream)'s exclusive prefix into the
    // state fragments (through scratch, row-major), the carry copies and A.
    load_split_planes<Traits>(planes, planes_hi, planes_lo);
    const float* prefix =
        workspace + workspace_slice_offset(tile, bh, 0, batch_heads, Traits::L, Traits::kCorners * (D + 1));
    const int tc = warp % Traits::kColTiles;
    FragAcc state[Traits::kStateTilesPerWarp];
#pragma unroll
    for (int f = 0; f < Traits::kStateTilesPerWarp; ++f) {
        const int idx = warp + f * Traits::kWarps;
        if (idx < Traits::kStateTiles) {
            const int fs = idx / Traits::kColTiles;
            const int s = fs * kFrag + lane / 2;
            float row[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (s < Traits::kFeatures) {
                const float* src = prefix + workspace_row_offset<D, P>(s) + tc * kFrag + (lane % 2) * 8;
#pragma unroll
                for (int c = 0; c < 8; c += 2) {
                    const float2 pair = *reinterpret_cast<const float2*>(src + c);
                    row[c] = pair.x;
                    row[c + 1] = pair.y;
                }
            }
            store_scratch_row8(scratch, lane, row);
            __syncwarp();
            wmma::load_matrix_sync(state[f], scratch, kScratchStride, wmma::mem_row_major);
            __syncwarp();
            store_carry_copies<Traits, kPreciseCarry>(state[f], fs, tc, carry_hi, carry_lo, scratch, lane);
        }
    }
    for (int s = threadIdx.x; s < Traits::kFeaturesPadded; s += Traits::kThreads) {
        mass[s] = s < Traits::kFeatures ? prefix[workspace_mass_offset<D, P>(s)] : 0.0f;
    }

    // Barrier invariant: each phase reads only what an earlier phase wrote
    // before the barrier in between, and a region is rewritten only after the
    // barrier that follows its last read. The prologue's writes are covered
    // by the first barrier.
    for (int chunk_begin = tile_begin; chunk_begin < tile_end; chunk_begin += C) {
        const int rows_valid = min(C, tile_end - chunk_begin);
        if constexpr (!kPrefetch) {
            fetch_rows<Traits, 3>(sources, stream_offset + static_cast<size_t>(chunk_begin) * D, rows_valid,
                                  staged);
        }
        store_rows<Traits, 3>(staged, destinations, strides);
        __syncthreads();  // (1) rows staged
        if (kPrefetch && chunk_begin + C < tile_end) {
            fetch_rows<Traits, 3>(sources, stream_offset + static_cast<size_t>(chunk_begin + C) * D,
                                  min(C, tile_end - chunk_begin - C), staged);
        }
        project_rows<Traits, 2>(staged_rows, planes_hi, planes_lo, u_tiles, warp);
        __syncthreads();  // (2) U ready; the staged Q and K rows are dead
        compute_features<Traits, 2>(phi, u_tiles, beta_ptr, rows_valid);
        __syncthreads();  // (3) Phi_Q, Phi_K ready; the U tiles are dead
        gram_tiles<Traits>(x_q, x_k, gram_hi, gram_lo, scratch, warp, lane);
        __syncthreads();  // (4) G tiles ready
        output_tiles<Traits, kPreciseCarry>(x_q, carry_hi, carry_lo, gram_hi, gram_lo, x_v, mass, rows_valid,
                                            out + stream_offset + static_cast<size_t>(chunk_begin) * D, scratch,
                                            warp, lane);
        __syncthreads();  // (5) every read of the carry copies and of A is done
        update_state<Traits, true, kPreciseCarry>(state, x_k, x_v, carry_hi, carry_lo, scratch, warp, lane);
        accumulate_mass<Traits>(x_k, mass);
        __syncthreads();  // (6) carry copies and A updated; x_* free for the next rows
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Calls fn(integral_constant<D>, integral_constant<P>, integral_constant<L>).
template <typename Fn>
cudaError_t dispatch_tc_shape(const CausalShape& shape, Fn&& fn) {
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

template <int D, int P, int L>
using OutputPassTraits = TcTraits<D, P, L, output_pass_launch(D, P).chunk, output_pass_launch(D, P).warps>;
template <int D, int P, int L>
using TileSumsTraits = TcTraits<D, P, L, tile_sums_launch(D, P).chunk, tile_sums_launch(D, P).warps>;

using TileSumsKernel = void (*)(const bf16*, const bf16*, const float*, const float*, float*, int,
                                int, int);
using OutputPassKernel = void (*)(const bf16*, const bf16*, const bf16*, const float*, const float*,
                                  const float*, bf16*, int, int, int);

// A kernel with its shared memory opted in, and its launch shape.
template <typename Kernel>
struct PreparedKernel {
    Kernel kernel = nullptr;
    size_t smem_bytes = 0;
    int threads = 0;
};

template <int D, int P, int L>
cudaError_t prepare_output_pass(PreparedKernel<OutputPassKernel>* prepared) {
    using Traits = OutputPassTraits<D, P, L>;
    prepared->kernel = output_pass_tc_kernel<Traits>;
    prepared->smem_bytes = tc_smem_layout(D, P, L, Traits::C, Traits::kWarps, true, kPreciseCarry).total;
    prepared->threads = Traits::kThreads;
    return opt_in_shared_memory(prepared->kernel, prepared->smem_bytes);
}

template <int D, int P, int L>
cudaError_t prepare_tile_sums(PreparedKernel<TileSumsKernel>* prepared) {
    using Traits = TileSumsTraits<D, P, L>;
    prepared->kernel = tile_sums_tc_kernel<Traits>;
    prepared->smem_bytes = tc_smem_layout(D, P, L, Traits::C, Traits::kWarps, false, false).total;
    prepared->threads = Traits::kThreads;
    return opt_in_shared_memory(prepared->kernel, prepared->smem_bytes);
}

}  // namespace

int sub_chunk_tokens(int head_dim, int num_planes) { return output_pass_launch(head_dim, num_planes).chunk; }

size_t output_pass_smem_bytes(const CausalShape& shape) {
    const TcLaunch launch = output_pass_launch(shape.head_dim, shape.num_planes);
    return tc_smem_layout(shape.head_dim, shape.num_planes, shape.num_tables, launch.chunk, launch.warps,
                          true, kPreciseCarry)
        .total;
}

cudaError_t output_pass_occupancy(const CausalShape& shape, int* ctas_per_sm) {
    return dispatch_tc_shape(shape, [&](auto dim, auto planes_count, auto tables) {
        PreparedKernel<OutputPassKernel> prepared;
        const cudaError_t err =
            prepare_output_pass<decltype(dim)::value, decltype(planes_count)::value, decltype(tables)::value>(
                &prepared);
        if (err != cudaSuccess) return err;
        return cudaOccupancyMaxActiveBlocksPerMultiprocessor(ctas_per_sm, prepared.kernel, prepared.threads,
                                                             prepared.smem_bytes);
    });
}

cudaError_t tile_sums(const bf16* k, const bf16* v, const float* planes, const float* beta,
                      float* workspace, const CausalShape& shape, int tile_tokens,
                      cudaStream_t stream) {
    const dim3 grid(num_tiles(shape, tile_tokens), shape.batch_heads);
    return dispatch_tc_shape(shape, [&](auto dim, auto planes_count, auto tables) {
        PreparedKernel<TileSumsKernel> prepared;
        const cudaError_t err =
            prepare_tile_sums<decltype(dim)::value, decltype(planes_count)::value, decltype(tables)::value>(
                &prepared);
        if (err != cudaSuccess) return err;
        prepared.kernel<<<grid, prepared.threads, prepared.smem_bytes, stream>>>(
            k, v, planes, beta, workspace, shape.seq_len, shape.batch_heads, tile_tokens);
        return cudaGetLastError();
    });
}

cudaError_t output(const bf16* q, const bf16* k, const bf16* v, const float* planes,
                   const float* beta, const float* workspace, bf16* out, const CausalShape& shape,
                   int tile_tokens, cudaStream_t stream) {
    const dim3 grid(num_tiles(shape, tile_tokens), shape.batch_heads);
    return dispatch_tc_shape(shape, [&](auto dim, auto planes_count, auto tables) {
        PreparedKernel<OutputPassKernel> prepared;
        const cudaError_t err =
            prepare_output_pass<decltype(dim)::value, decltype(planes_count)::value, decltype(tables)::value>(
                &prepared);
        if (err != cudaSuccess) return err;
        prepared.kernel<<<grid, prepared.threads, prepared.smem_bytes, stream>>>(
            q, k, v, planes, beta, workspace, out, shape.seq_len, shape.batch_heads, tile_tokens);
        return cudaGetLastError();
    });
}

}  // namespace tc
}  // namespace causal
}  // namespace race
