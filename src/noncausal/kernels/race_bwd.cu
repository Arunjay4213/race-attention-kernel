// Non-causal RACE Attention backward pass: four kernels plus host launchers.
// The derivation, with every symbol used below, is in BACKWARD_DERIVATION.md.
//
// Per head, with g_i = dO_i and the forward's reduced A_l, B_l:
//   query side   y_l[r] = B_l[r] . g_i,  Den_i = sum phi A,  gNum_i = sum phi y
//                1 / Den_i and dDen_i = -gNum_i / Den_i^2
//                dphi_q[r] = y_l[r] / Den_i + A_l[r] dDen_i
//   bucket grads dB_l[r] = sum_i phi_l(q_i)[r] g_i / Den_i,  dA_l[r] = sum_i phi_l(q_i)[r] dDen_i
//   key side     dv_j = sum_l sum_r phi_l(k_j)[r] dB_l[r]
//                dphi_k[r] = dB_l[r] . v_j + dA_l[r]
//   both sides   h_t = sum_r phi[r] dphi[r] c_{r,t}   (c from corner_coefficient)
//                dx += beta h_t (1 - u_t^2) w_{l,t},  d beta += u_t h_t
//
// Kernels:
//   1. query_grad_kernel: one CTA per (query tile, head), one warp per query
//      token, A and B in shared memory like the forward query kernel. Three
//      passes over the tables per token: hash, then y and Den, gNum (which
//      sum over all tables), and only then dphi and the chain rule. Writes
//      dq, the per-token pair (1 / Den, dDen), and one beta partial per CTA.
//   2. bucket_build_kernel<D, P, true> (race_internal.cuh): the forward's
//      bucket build over (q_i, g_i) with per-token weights, so dB and dA
//      reuse its register-resident ownership and staging unchanged.
//   3. race_bucket_reduce (race_fwd.cu): the same deterministic tree reduce.
//   4. key_grad_kernel: one CTA per (key tile, head), one warp per key token,
//      dA and dB in shared memory. Writes dk, dv and one beta partial per CTA.
//      Also two passes per token (hash every table, then the gradients), so
//      the key row is dead during the gradient sweep.
//   5. beta_grad_reduce_kernel: one CTA sums all beta partials in a fixed order.
//
// Why dq and dB/dA are separate kernels: dB_l needs 1 / Den_i and dDen_i,
// which need every table, while the build owns one table per CTA to keep its
// accumulators in registers (L * R * D / 256 = up to 64 per thread for all
// tables at once). Splitting costs a second read of q and dO (the L build
// CTAs of a tile share it through L2) and an 8-byte weight per token.
//
// Nothing is saved from the forward except its reduced A, B (L * R * (D + 1)
// floats per head): O is never needed because gNum = g . Num is formed from
// the y values that dphi_q needs anyway, and Den is recomputed from A.
//
// Per-token dot products over D against every corner (y and dB . v) are
// reduce-scatters (corner_reduce_scatter): each lane dots its D / 32 columns
// against all R rows, and R - 1 + (5 - P) shuffles leave lane r with the sum
// for corner r & (R - 1). All per-corner values are held in this corner
// layout, so lanes r >= R carry copies and sums over corners are butterflies
// over the low P lane bits (corner_allreduce_sum).
//
// Precision: bf16 in and out, fp32 everywhere in between, accurate tanhf and
// expf. No atomics anywhere.
#include "race_bwd.h"

#include <algorithm>
#include <cstdint>

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
using detail::kThreads;
using detail::kWarps;
using detail::round_up4;
using detail::token_row_offset;

// Register budget of the two per-token kernels, stated as a launch bound:
// 4 CTAs of 256 threads fill the 64K-register file at 64 registers per
// thread. Without it ptxas picks its own occupancy target (48 registers on
// sm_90 for several instantiations) and spills a few bytes to reach it; with
// it, every instantiation compiles to at most 64 registers and no spills.
// Shared memory, not registers, is what limits residency for large L * R.
constexpr int kMinBlocksPerSm = 4;

// Dynamic shared memory of the two per-token kernels, in floats: the reduced
// buckets of one head ([L][R][D], then [L][R] masses), all planes [L][P][D],
// then kWarps per-warp scratch regions of `scratch_stride` floats. Same
// layout and reasoning as the forward's QuerySmem; every region starts at a
// multiple of 4 floats for the float4 reads.
template <int D, int P>
struct TotalsSmem {
    static constexpr int kCorners = Dims<D, P>::kCorners;
    __host__ __device__ static int buckets(int) { return 0; }
    __host__ __device__ static int mass(int num_tables) { return num_tables * kCorners * D; }
    __host__ __device__ static int planes(int num_tables) {
        return mass(num_tables) + round_up4(num_tables * kCorners);
    }
    __host__ __device__ static int scratch(int num_tables) {
        return planes(num_tables) + num_tables * P * D;
    }
    __host__ __device__ static int total_floats(int num_tables, int scratch_stride) {
        return scratch(num_tables) + kWarps * scratch_stride;
    }
};

// Per-warp scratch of the two per-token kernels, in floats, rewritten for
// every token. The first pass over the tables fills probs and hash, later
// passes read them:
//   probs        [L][R]     phi_l[r], written by lanes 0..R-1
//   hash         [L][3][P]  u_t, sigmoid(2 beta u_t), sigmoid(-2 beta u_t),
//                           written by lanes 0..P-1
//   grad_z       [L][P]     dL/dz_{l,t} for z = w_{l,t} . x, written by lanes 0..P-1
//   corner_grads [L][R]     y_l[r] = B_l[r] . g (query side only)
// Keeping u and the sigmoids here means the later passes evaluate no
// transcendentals and no token row stays live across the table loops, and
// staging dL/dz here lets the row gradient sum_l sum_t dL/dz_{l,t} w_{l,t}
// run as a separate short loop instead of holding its accumulators and plane
// slices across the P butterflies. Together with reciprocal() below this
// keeps every instantiation within 64 registers and free of spills.
template <int D, int P, bool kWithCornerGrads>
struct TokenScratch {
    static constexpr int kCorners = Dims<D, P>::kCorners;
    __host__ __device__ static int probs(int) { return 0; }
    __host__ __device__ static int hash(int num_tables) {
        return round_up4(num_tables * kCorners);
    }
    __host__ __device__ static int grad_z(int num_tables) {
        return hash(num_tables) + round_up4(num_tables * 3 * P);
    }
    __host__ __device__ static int corner_grads(int num_tables) {
        return grad_z(num_tables) + round_up4(num_tables * P);
    }
    __host__ __device__ static int stride(int num_tables) {
        return corner_grads(num_tables) +
               (kWithCornerGrads ? round_up4(num_tables * kCorners) : 0);
    }
};

// 1 / x for 0 < x < 2^126, accurate to 1 ulp, without a subroutine call.
// NaN propagates; 0 and inf are outside the contract (both give NaN). The
// IEEE division 1.0f / x compiles to an inline fast path plus a call for
// out-of-range operands, and a call forces the live registers around it into
// local memory (this made several instantiations spill). This is that fast
// path, the hardware approximation refined by one Newton step with an exact
// fma residual, with subnormal x first scaled by 2^24 into the range where
// the approximation is accurate. A result above FLT_MAX overflows to inf, as
// the division would.
__device__ __forceinline__ float reciprocal(float x) {
    constexpr float kScale = 16777216.0f;          // 2^24
    constexpr float kMinNormal = 1.17549435e-38f;  // FLT_MIN
    const bool subnormal = x < kMinNormal;
    const float scaled = subnormal ? x * kScale : x;
    const float approx = __fdividef(1.0f, scaled);
    const float refined = fmaf(approx, fmaf(-scaled, approx, 1.0f), approx);
    return subnormal ? refined * kScale : refined;
}

// sigmoid_pair (race_common.cuh) with reciprocal() in place of the division:
// the same formulas, so the values agree with the forward's to about 1 ulp.
__device__ __forceinline__ void sigmoid_pair_no_call(float z, float& prob_plus,
                                                     float& prob_minus) {
    const float e = expf(-fabsf(z));
    const float large = reciprocal(1.0f + e);
    const float small = e * large;
    prob_plus = z >= 0.0f ? large : small;
    prob_minus = z >= 0.0f ? small : large;
}

// First pass for one table: hashes the token row, stores u, the sigmoids and
// phi into the warp's scratch, and returns phi[corner]. The arithmetic is
// that of corner_probability (same dot products, butterfly, tanhf, product
// order) except for the reciprocal in the sigmoids, so u is bitwise the
// forward's and phi agrees to a few ulp. Each plane's values go to shared
// memory as soon as they exist and phi is built as a running product, so
// little state is live across the next plane's transcendentals. Must be
// called by all 32 lanes.
template <int D, int P>
__device__ __forceinline__ float hash_to_scratch(const float (&row)[D / kWarpSize],
                                                 const float* table_planes, float beta,
                                                 int lane, int corner, float* table_probs,
                                                 float* table_hash) {
    constexpr int kVec = D / kWarpSize;
    float phi = 1.0f;
#pragma unroll
    for (int t = 0; t < P; ++t) {
        float plane_slice[kVec];
        load_f32_vec<kVec>(table_planes + t * D + lane * kVec, plane_slice);
        float partial = 0.0f;
#pragma unroll
        for (int j = 0; j < kVec; ++j) partial = fmaf(row[j], plane_slice[j], partial);
        const float projection = tanhf(warp_allreduce_sum(partial));
        float prob_plus;
        float prob_minus;
        sigmoid_pair_no_call(2.0f * beta * projection, prob_plus, prob_minus);
        if (lane == t) {
            table_hash[t] = projection;
            table_hash[P + t] = prob_plus;
            table_hash[2 * P + t] = prob_minus;
        }
        const bool plus = (corner >> (P - 1 - t)) & 1;
        phi *= plus ? prob_plus : prob_minus;
    }
    if (lane < (1 << P)) table_probs[lane] = phi;
    return phi;
}

// Copies one head's reduced bucket sums (the L contiguous workspace slices
// starting at `head_totals`) and all planes into shared memory. Block-strided;
// the caller synchronizes before reading.
template <int D, int P>
__device__ __forceinline__ void load_totals(const float* __restrict__ head_totals,
                                            const float* __restrict__ planes, int num_tables,
                                            float* buckets_s, float* mass_s, float* planes_s) {
    constexpr int kCorners = Dims<D, P>::kCorners;
    constexpr int kSliceFloats = Dims<D, P>::kSliceFloats;
    constexpr int kBucketFloats = kCorners * D;
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
}

// Pulls one table's corner gradients back through phi and tanh, with u and
// the sigmoids read from the table's hash scratch ([3][P]):
//   h_t = sum_r phi[r] dphi[r] c_{r,t}
//   dL/dz_t = beta h_t (1 - u_t^2)   -> table_grad_z[t], written by lane t
//   beta_grad += u_t h_t
// `weighted_grad` is phi[corner] * dphi[corner] in corner layout. h_t, and so
// beta_grad and dL/dz_t, is bitwise identical on every lane. 1 - u^2 is one
// fma: its absolute error stays below 2^-23 even where tanh saturates, which
// is small against its natural scale of 1. Must be called by all 32 lanes.
template <int P>
__device__ __forceinline__ void hash_grad_to_scratch(float weighted_grad, const float* table_hash,
                                                     int corner, float beta, int lane,
                                                     float* table_grad_z, float& beta_grad) {
#pragma unroll
    for (int t = 0; t < P; ++t) {
        const float projection = table_hash[t];
        const float coef =
            corner_coefficient<P>(table_hash[P + t], table_hash[2 * P + t], corner, t);
        const float h = corner_allreduce_sum<P>(weighted_grad * coef);
        beta_grad = fmaf(projection, h, beta_grad);
        if (lane == t) table_grad_z[t] = beta * h * fmaf(-projection, projection, 1.0f);
    }
}

// Row gradient from the staged dL/dz: grad_row[j] = sum_l sum_t
// dL/dz_{l,t} w_{l,t}[lane * kVec + j], accumulated table-major, plane-minor.
template <int D, int P>
__device__ __forceinline__ void projection_grads_to_row(const float* grad_z_s,
                                                        const float* planes_s, int num_tables,
                                                        int lane,
                                                        float (&grad_row)[D / kWarpSize]) {
    constexpr int kVec = D / kWarpSize;
#pragma unroll
    for (int j = 0; j < kVec; ++j) grad_row[j] = 0.0f;
#pragma unroll 1
    for (int table = 0; table < num_tables; ++table) {
#pragma unroll
        for (int t = 0; t < P; ++t) {
            const float grad_z = grad_z_s[table * P + t];
            float plane_slice[kVec];
            load_f32_vec<kVec>(planes_s + (table * P + t) * D + lane * kVec, plane_slice);
#pragma unroll
            for (int j = 0; j < kVec; ++j) grad_row[j] = fmaf(grad_z, plane_slice[j], grad_row[j]);
        }
    }
}

// Sums the per-warp beta gradients of a CTA in warp order and writes the
// CTA's partial. Must be called by every thread of the CTA (it has a barrier);
// `beta_grad` is identical on all lanes of a warp.
__device__ __forceinline__ void write_beta_partial(float beta_grad, float* warp_beta_s,
                                                   float* __restrict__ beta_partial) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (lane == 0) warp_beta_s[warp] = beta_grad;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) total += warp_beta_s[w];
        *beta_partial = total;
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: query side
// ---------------------------------------------------------------------------

// Grid: x = query tiles, y = batch_heads. `totals` is the forward's reduced
// A, B. beta_partials receives one value per CTA at [bh][tile].
template <int D, int P>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm)
    query_grad_kernel(const bf16* __restrict__ queries, const bf16* __restrict__ grad_out,
                      const float* __restrict__ planes, const float* __restrict__ beta_ptr,
                      const float* __restrict__ totals, bf16* __restrict__ grad_queries,
                      float2* __restrict__ token_weights, float* __restrict__ beta_partials,
                      int seq_len, int num_tables) {
    using Layout = TotalsSmem<D, P>;
    using Scratch = TokenScratch<D, P, true>;
    constexpr int kCorners = Dims<D, P>::kCorners;
    constexpr int kVec = Dims<D, P>::kVec;
    constexpr int kBucketFloats = kCorners * D;
    constexpr int kHashFloats = 3 * P;

    extern __shared__ __align__(16) float smem[];
    __shared__ float warp_beta_s[kWarps];
    float* buckets_s = smem + Layout::buckets(num_tables);  // [L][R][D]
    float* mass_s = smem + Layout::mass(num_tables);        // [L][R]
    float* planes_s = smem + Layout::planes(num_tables);    // [L][P][D]

    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int corner = lane & (kCorners - 1);
    float* scratch = smem + Layout::scratch(num_tables) + warp * Scratch::stride(num_tables);
    float* probs_s = scratch + Scratch::probs(num_tables);                // [L][R]
    float* hash_s = scratch + Scratch::hash(num_tables);                  // [L][3][P]
    float* grad_z_s = scratch + Scratch::grad_z(num_tables);              // [L][P]
    float* corner_grads_s = scratch + Scratch::corner_grads(num_tables);  // [L][R]

    load_totals<D, P>(totals + static_cast<size_t>(bh) * num_tables * Dims<D, P>::kSliceFloats,
                      planes, num_tables, buckets_s, mass_s, planes_s);
    const float beta = *beta_ptr;
    __syncthreads();

    const int tile_begin = blockIdx.x * kQueryTileTokens;
    const int tile_end = min(seq_len, tile_begin + kQueryTileTokens);
    float beta_grad = 0.0f;  // identical on all lanes of the warp

    // Warp-uniform loop: only __syncwarp inside, no block barriers.
    for (int token = tile_begin + warp; token < tile_end; token += kWarps) {
        const size_t row = token_row_offset(bh, token, seq_len, D) + lane * kVec;
        // Pass 1: hash every table; the query row is not needed afterwards.
        // dO is loaded only after it, so that it is not live across the
        // hashing (see TokenScratch).
        {
            float query[kVec];
            load_bf16_vec<kVec>(queries + row, query);
#pragma unroll 1
            for (int table = 0; table < num_tables; ++table) {
                hash_to_scratch<D, P>(query, planes_s + table * P * D, beta, lane, corner,
                                      probs_s + table * kCorners, hash_s + table * kHashFloats);
            }
        }
        float grad[kVec];
        load_bf16_vec<kVec>(grad_out + row, grad);

        // Pass 2: y_l[r] = B_l[r] . g and the lane's shares of Den and gNum.
        // Each lane reads back only the phi it wrote itself (or, for lanes
        // >= R, the copy lane `corner` wrote, which the __syncwarp orders).
        __syncwarp();
        float den_partial = 0.0f;
        float grad_num_partial = 0.0f;
#pragma unroll 1
        for (int table = 0; table < num_tables; ++table) {
            const float* table_buckets = buckets_s + table * kBucketFloats + lane * kVec;
            const float corner_grad = corner_reduce_scatter<P>(lane, [&](int r) {
                float bucket[kVec];
                load_f32_vec<kVec>(table_buckets + r * D, bucket);
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < kVec; ++j) dot = fmaf(bucket[j], grad[j], dot);
                return dot;
            });
            const int slot = table * kCorners + corner;
            if (lane < kCorners) corner_grads_s[slot] = corner_grad;
            const float phi = probs_s[slot];
            den_partial = fmaf(phi, mass_s[slot], den_partial);
            grad_num_partial = fmaf(phi, corner_grad, grad_num_partial);
        }

        // Den = 0 only after fp32 underflow; the forward outputs 0 for such a
        // row, which does not depend on any input, so its gradient is 0 too.
        // dDen is formed as (gNum / Den) / Den: gNum / Den = g . O is bounded
        // by |g| max|v|, so only a true gradient overflow can overflow here.
        const float den = corner_allreduce_sum<P>(den_partial);
        const float grad_num = corner_allreduce_sum<P>(grad_num_partial);
        const float inv_den = den == 0.0f ? 0.0f : reciprocal(den);
        const float grad_den = -(grad_num * inv_den) * inv_den;
        __syncwarp();  // corner_grads_s written by lanes 0..R-1 is read by all lanes

        // Pass 3: dphi = y / Den + A dDen, then back through phi and tanh.
#pragma unroll 1
        for (int table = 0; table < num_tables; ++table) {
            const int slot = table * kCorners + corner;
            const float grad_phi = fmaf(corner_grads_s[slot], inv_den, mass_s[slot] * grad_den);
            hash_grad_to_scratch<P>(probs_s[slot] * grad_phi, hash_s + table * kHashFloats, corner,
                                    beta, lane, grad_z_s + table * P, beta_grad);
        }
        __syncwarp();  // grad_z_s written by lanes 0..P-1 is read by all lanes

        float grad_query[kVec];
        projection_grads_to_row<D, P>(grad_z_s, planes_s, num_tables, lane, grad_query);
        store_bf16_vec<kVec>(grad_queries + row, grad_query);
        if (lane == 0) {
            token_weights[static_cast<size_t>(bh) * seq_len + token] =
                make_float2(inv_den, grad_den);
        }
        __syncwarp();  // scratch is rewritten by the next token
    }

    write_beta_partial(beta_grad, warp_beta_s,
                       beta_partials + static_cast<size_t>(bh) * gridDim.x + blockIdx.x);
}

// ---------------------------------------------------------------------------
// Kernel 4: key side
// ---------------------------------------------------------------------------

// Grid: x = key tiles, y = batch_heads. `grad_totals` is the reduced dA, dB
// (tile 0 of the gradient workspace). beta_partials receives one value per
// CTA at [bh][tile].
template <int D, int P>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm)
    key_grad_kernel(const bf16* __restrict__ keys, const bf16* __restrict__ values,
                    const float* __restrict__ planes, const float* __restrict__ beta_ptr,
                    const float* __restrict__ grad_totals, bf16* __restrict__ grad_keys,
                    bf16* __restrict__ grad_values, float* __restrict__ beta_partials,
                    int seq_len, int num_tables) {
    using Layout = TotalsSmem<D, P>;
    using Scratch = TokenScratch<D, P, false>;
    constexpr int kCorners = Dims<D, P>::kCorners;
    constexpr int kVec = Dims<D, P>::kVec;
    constexpr int kBucketFloats = kCorners * D;
    constexpr int kHashFloats = 3 * P;

    extern __shared__ __align__(16) float smem[];
    __shared__ float warp_beta_s[kWarps];
    float* buckets_s = smem + Layout::buckets(num_tables);  // [L][R][D] dB
    float* mass_s = smem + Layout::mass(num_tables);        // [L][R] dA
    float* planes_s = smem + Layout::planes(num_tables);    // [L][P][D]

    const int bh = blockIdx.y;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int corner = lane & (kCorners - 1);
    float* scratch = smem + Layout::scratch(num_tables) + warp * Scratch::stride(num_tables);
    float* probs_s = scratch + Scratch::probs(num_tables);  // [L][R]
    float* hash_s = scratch + Scratch::hash(num_tables);    // [L][3][P]
    float* grad_z_s = scratch + Scratch::grad_z(num_tables);  // [L][P]

    load_totals<D, P>(
        grad_totals + static_cast<size_t>(bh) * num_tables * Dims<D, P>::kSliceFloats, planes,
        num_tables, buckets_s, mass_s, planes_s);
    const float beta = *beta_ptr;
    __syncthreads();

    const int tile_begin = blockIdx.x * kQueryTileTokens;
    const int tile_end = min(seq_len, tile_begin + kQueryTileTokens);
    float beta_grad = 0.0f;  // identical on all lanes of the warp

    // Warp-uniform loop: only __syncwarp inside, no block barriers.
    for (int token = tile_begin + warp; token < tile_end; token += kWarps) {
        const size_t row = token_row_offset(bh, token, seq_len, D) + lane * kVec;
        float value[kVec];
        load_bf16_vec<kVec>(values + row, value);
        {
            float key[kVec];
            load_bf16_vec<kVec>(keys + row, key);
            // Pass 1: hash every table; the key row is not needed afterwards.
#pragma unroll 1
            for (int table = 0; table < num_tables; ++table) {
                hash_to_scratch<D, P>(key, planes_s + table * P * D, beta, lane, corner,
                                      probs_s + table * kCorners, hash_s + table * kHashFloats);
            }
        }
        __syncwarp();  // every lane reads all R probabilities below

        // Pass 2, per table, one sweep over dB_l: dv += phi[r] dB_l[r] and the
        // partials of dB_l[r] . v, then dphi[r] = dB_l[r] . v + dA_l[r] and
        // the chain rule back to dL/dz. Lane owns columns
        // [lane * kVec, lane * kVec + kVec) of dk and dv.
        float grad_value[kVec];
#pragma unroll
        for (int j = 0; j < kVec; ++j) grad_value[j] = 0.0f;
#pragma unroll 1
        for (int table = 0; table < num_tables; ++table) {
            const float* table_probs = probs_s + table * kCorners;
            const float* table_buckets = buckets_s + table * kBucketFloats + lane * kVec;
            const float corner_grad = corner_reduce_scatter<P>(lane, [&](int r) {
                float bucket[kVec];
                load_f32_vec<kVec>(table_buckets + r * D, bucket);
                const float phi_r = table_probs[r];
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < kVec; ++j) {
                    grad_value[j] = fmaf(phi_r, bucket[j], grad_value[j]);
                    dot = fmaf(bucket[j], value[j], dot);
                }
                return dot;
            });
            const float grad_phi = corner_grad + mass_s[table * kCorners + corner];
            hash_grad_to_scratch<P>(table_probs[corner] * grad_phi, hash_s + table * kHashFloats,
                                    corner, beta, lane, grad_z_s + table * P, beta_grad);
        }
        __syncwarp();  // grad_z_s written by lanes 0..P-1 is read by all lanes

        float grad_key[kVec];
        projection_grads_to_row<D, P>(grad_z_s, planes_s, num_tables, lane, grad_key);
        store_bf16_vec<kVec>(grad_keys + row, grad_key);
        store_bf16_vec<kVec>(grad_values + row, grad_value);
        __syncwarp();  // scratch is rewritten by the next token
    }

    write_beta_partial(beta_grad, warp_beta_s,
                       beta_partials + static_cast<size_t>(bh) * gridDim.x + blockIdx.x);
}

// ---------------------------------------------------------------------------
// Kernel 5: beta gradient reduce
// ---------------------------------------------------------------------------

// One CTA. Thread t sums partials t, t + kThreads, ... serially, then a fixed
// pairwise tree over the threads; the order depends only on `count`.
__global__ void __launch_bounds__(kThreads)
    beta_grad_reduce_kernel(const float* __restrict__ partials, int64_t count,
                            float* __restrict__ grad_beta) {
    __shared__ float sums_s[kThreads];
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < count; i += kThreads) sum += partials[i];
    sums_s[threadIdx.x] = sum;
    __syncthreads();
#pragma unroll
    for (int width = kThreads / 2; width > 0; width >>= 1) {
        if (threadIdx.x < width) sums_s[threadIdx.x] += sums_s[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) *grad_beta = sums_s[0];
}

// Sets the dynamic shared memory limit when a launch needs more than the
// default. The 48 KB default bounds static plus dynamic shared memory, and
// both per-token kernels also hold warp_beta_s statically, so a dynamic size
// of exactly 48 KB (D = 64, P = 5, L = 4 in query_grad_kernel) already needs
// the opt-in. The attribute is per device, so it is set on every call rather
// than cached.
template <typename Kernel>
cudaError_t opt_in_dynamic_smem(Kernel* kernel, size_t smem_bytes) {
    constexpr size_t kStaticSmemBytes = kWarps * sizeof(float);  // warp_beta_s
    if (smem_bytes + kStaticSmemBytes <= kDefaultDynamicSmemBytes) return cudaSuccess;
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(smem_bytes));
}

size_t round_up4(size_t n) { return (n + 3) & ~static_cast<size_t>(3); }

}  // namespace

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

int num_query_tiles(const ForwardShape& shape) {
    return ceil_div(shape.seq_len, kQueryTileTokens);
}

size_t token_weight_floats(const ForwardShape& shape) {
    return 2 * static_cast<size_t>(shape.batch_heads) * shape.seq_len;
}

size_t beta_partial_floats(const ForwardShape& shape) {
    return 2 * static_cast<size_t>(shape.batch_heads) * num_query_tiles(shape);
}

size_t backward_workspace_floats(const ForwardShape& shape) {
    return round_up4(workspace_floats(shape)) + round_up4(token_weight_floats(shape)) +
           beta_partial_floats(shape);
}

cudaError_t race_query_grad(const bf16* q, const bf16* grad_out, const float* planes,
                            const float* beta, const float* bucket_totals, bf16* grad_q,
                            float2* token_weights, float* beta_partials,
                            const ForwardShape& shape, cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const dim3 grid(num_query_tiles(shape), shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        auto* kernel = query_grad_kernel<D, P>;
        const int scratch_stride = TokenScratch<D, P, true>::stride(shape.num_tables);
        const size_t smem_bytes =
            static_cast<size_t>(TotalsSmem<D, P>::total_floats(shape.num_tables, scratch_stride)) *
            sizeof(float);
        const cudaError_t err = opt_in_dynamic_smem(kernel, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel<<<grid, kThreads, smem_bytes, stream>>>(q, grad_out, planes, beta, bucket_totals,
                                                       grad_q, token_weights, beta_partials,
                                                       shape.seq_len, shape.num_tables);
        return cudaGetLastError();
    });
}

cudaError_t race_bucket_grad_build(const bf16* q, const bf16* grad_out,
                                   const float2* token_weights, const float* planes,
                                   const float* beta, float* grad_workspace,
                                   const ForwardShape& shape, cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const dim3 grid(num_build_tiles(shape) * shape.num_tables, shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        bucket_build_kernel<D, P, true><<<grid, kThreads, 0, stream>>>(
            q, grad_out, token_weights, planes, beta, grad_workspace, shape.seq_len,
            shape.batch_heads, shape.num_tables);
        return cudaGetLastError();
    });
}

cudaError_t race_key_grad(const bf16* k, const bf16* v, const float* planes, const float* beta,
                          const float* grad_totals, bf16* grad_k, bf16* grad_v,
                          float* beta_partials, const ForwardShape& shape, cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const dim3 grid(num_query_tiles(shape), shape.batch_heads);
    return dispatch_shape(shape, [&](auto dim, auto planes_count) {
        constexpr int D = decltype(dim)::value;
        constexpr int P = decltype(planes_count)::value;
        auto* kernel = key_grad_kernel<D, P>;
        const int scratch_stride = TokenScratch<D, P, false>::stride(shape.num_tables);
        const size_t smem_bytes =
            static_cast<size_t>(TotalsSmem<D, P>::total_floats(shape.num_tables, scratch_stride)) *
            sizeof(float);
        const cudaError_t err = opt_in_dynamic_smem(kernel, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel<<<grid, kThreads, smem_bytes, stream>>>(k, v, planes, beta, grad_totals, grad_k,
                                                       grad_v, beta_partials, shape.seq_len,
                                                       shape.num_tables);
        return cudaGetLastError();
    });
}

cudaError_t race_beta_grad_reduce(const float* beta_partials, float* grad_beta,
                                  const ForwardShape& shape, cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    const int64_t count = static_cast<int64_t>(beta_partial_floats(shape));
    beta_grad_reduce_kernel<<<1, kThreads, 0, stream>>>(beta_partials, count, grad_beta);
    return cudaGetLastError();
}

cudaError_t race_backward(const bf16* grad_out, const bf16* q, const bf16* k, const bf16* v,
                          const float* planes, const float* beta, const float* bucket_totals,
                          float* workspace, bf16* grad_q, bf16* grad_k, bf16* grad_v,
                          float* grad_beta, const ForwardShape& shape, cudaStream_t stream) {
    if (!is_supported(shape)) return cudaErrorInvalidValue;
    // Region offsets are multiples of 4 floats, so every region keeps the
    // buffer's 16-byte alignment (token_weights needs 8).
    float* grad_workspace = workspace;
    float2* token_weights =
        reinterpret_cast<float2*>(workspace + round_up4(workspace_floats(shape)));
    float* query_beta_partials =
        workspace + round_up4(workspace_floats(shape)) + round_up4(token_weight_floats(shape));
    float* key_beta_partials =
        query_beta_partials + static_cast<size_t>(shape.batch_heads) * num_query_tiles(shape);

    cudaError_t err = race_query_grad(q, grad_out, planes, beta, bucket_totals, grad_q,
                                      token_weights, query_beta_partials, shape, stream);
    if (err != cudaSuccess) return err;
    err = race_bucket_grad_build(q, grad_out, token_weights, planes, beta, grad_workspace, shape,
                                 stream);
    if (err != cudaSuccess) return err;
    err = race_bucket_reduce(grad_workspace, shape, stream);
    if (err != cudaSuccess) return err;
    err = race_key_grad(k, v, planes, beta, grad_workspace, grad_k, grad_v, key_beta_partials,
                        shape, stream);
    if (err != cudaSuccess) return err;
    return race_beta_grad_reduce(query_beta_partials, grad_beta, shape, stream);
}

}  // namespace race
