// Device helpers shared by the non-causal RACE forward and backward kernels.
//
// Conventions used throughout:
//   - One warp owns one token row of length D (64 or 128). Lane `lane` holds
//     the D / 32 consecutive elements starting at lane * (D / 32), so a warp
//     reads a row with one fully coalesced vector load per lane.
//   - Corner r of the hypercube {-1, +1}^P has sign +1 on plane t iff bit
//     (P - 1 - t) of r is set (plane 0 is the most significant bit), matching
//     itertools.product([-1, +1], repeat=P) in the reference.
//   - All arithmetic is fp32 with accurate tanhf / expf (no fast-math).
//   - The warp helpers use full-mask shuffles, so all 32 lanes call them.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace race {

constexpr int kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;

// Butterfly all-reduce. Every lane ends with a bitwise identical sum because
// at each level both partners add the same two operands (fp add commutes),
// which keeps per-lane downstream values (tanh, sigmoid) consistent.
__device__ __forceinline__ float warp_allreduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

// Loads VEC consecutive bf16 values (VEC = 2 or 4) as fp32. The address must
// be aligned to VEC * 2 bytes.
template <int VEC>
__device__ __forceinline__ void load_bf16_vec(const __nv_bfloat16* src, float (&dst)[VEC]) {
    static_assert(VEC == 2 || VEC == 4, "bf16 vector width must be 2 or 4");
    if constexpr (VEC == 2) {
        const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src));
        dst[0] = pair.x;
        dst[1] = pair.y;
    } else {
        const uint2 raw = *reinterpret_cast<const uint2*>(src);
        __nv_bfloat162 low, high;
        memcpy(&low, &raw.x, sizeof(low));
        memcpy(&high, &raw.y, sizeof(high));
        const float2 a = __bfloat1622float2(low);
        const float2 b = __bfloat1622float2(high);
        dst[0] = a.x;
        dst[1] = a.y;
        dst[2] = b.x;
        dst[3] = b.y;
    }
}

template <int VEC>
__device__ __forceinline__ void store_bf16_vec(__nv_bfloat16* dst, const float (&src)[VEC]) {
    static_assert(VEC == 2 || VEC == 4, "bf16 vector width must be 2 or 4");
    if constexpr (VEC == 2) {
        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(src[0], src[1]);
    } else {
        const __nv_bfloat162 low = __floats2bfloat162_rn(src[0], src[1]);
        const __nv_bfloat162 high = __floats2bfloat162_rn(src[2], src[3]);
        uint2 raw;
        memcpy(&raw.x, &low, sizeof(low));
        memcpy(&raw.y, &high, sizeof(high));
        *reinterpret_cast<uint2*>(dst) = raw;
    }
}

// Loads COUNT consecutive fp32 values from shared memory with the widest
// vector access the count allows. The address must be aligned to the access
// width (16 bytes for COUNT % 4 == 0, 8 bytes for COUNT == 2).
template <int COUNT>
__device__ __forceinline__ void load_f32_vec(const float* src, float (&dst)[COUNT]) {
    if constexpr (COUNT % 4 == 0) {
#pragma unroll
        for (int i = 0; i < COUNT; i += 4) {
            const float4 chunk = *reinterpret_cast<const float4*>(src + i);
            dst[i + 0] = chunk.x;
            dst[i + 1] = chunk.y;
            dst[i + 2] = chunk.z;
            dst[i + 3] = chunk.w;
        }
    } else if constexpr (COUNT == 2) {
        const float2 chunk = *reinterpret_cast<const float2*>(src);
        dst[0] = chunk.x;
        dst[1] = chunk.y;
    } else {
#pragma unroll
        for (int i = 0; i < COUNT; ++i) dst[i] = src[i];
    }
}

template <int COUNT>
__device__ __forceinline__ void store_f32_vec(float* dst, const float (&src)[COUNT]) {
    if constexpr (COUNT % 4 == 0) {
#pragma unroll
        for (int i = 0; i < COUNT; i += 4) {
            *reinterpret_cast<float4*>(dst + i) =
                make_float4(src[i + 0], src[i + 1], src[i + 2], src[i + 3]);
        }
    } else if constexpr (COUNT == 2) {
        *reinterpret_cast<float2*>(dst) = make_float2(src[0], src[1]);
    } else {
#pragma unroll
        for (int i = 0; i < COUNT; ++i) dst[i] = src[i];
    }
}

// sigmoid(z) and sigmoid(-z) from a single exp. Evaluating the smaller one as
// e / (1 + e) instead of 1 - sigmoid(z) keeps full relative precision in the
// tail, which matters because the corner products multiply up to 5 of them.
__device__ __forceinline__ void sigmoid_pair(float z, float& prob_plus, float& prob_minus) {
    const float e = expf(-fabsf(z));
    const float large = 1.0f / (1.0f + e);
    const float small = e * large;
    prob_plus = z >= 0.0f ? large : small;
    prob_minus = z >= 0.0f ? small : large;
}

// Probability of corner `corner` as a product of P independent Bernoullis.
template <int P>
__device__ __forceinline__ float bernoulli_product(const float (&prob_plus)[P],
                                                   const float (&prob_minus)[P], int corner) {
    float product = 1.0f;
#pragma unroll
    for (int t = 0; t < P; ++t) {
        const bool plus = (corner >> (P - 1 - t)) & 1;
        product *= plus ? prob_plus[t] : prob_minus[t];
    }
    return product;
}

// Corner probability phi(x)[lane] for one hash table, computed by a full warp.
//
// `row` is this lane's slice of the token (D / 32 values) and `planes` points
// at the table's P x D planes in shared memory. Each projection is a warp-wide
// dot product; after the all-reduce every lane holds the same u_t, and lane r
// evaluates corner r. Lanes r >= R return a value the caller must discard.
template <int D, int P>
__device__ __forceinline__ float corner_probability(const float (&row)[D / kWarpSize],
                                                    const float* planes, float beta, int lane) {
    constexpr int kVec = D / kWarpSize;
    float prob_plus[P];
    float prob_minus[P];
#pragma unroll
    for (int t = 0; t < P; ++t) {
        float plane_slice[kVec];
        load_f32_vec<kVec>(planes + t * D + lane * kVec, plane_slice);
        float partial = 0.0f;
#pragma unroll
        for (int j = 0; j < kVec; ++j) partial = fmaf(row[j], plane_slice[j], partial);
        const float projection = tanhf(warp_allreduce_sum(partial));
        sigmoid_pair(2.0f * beta * projection, prob_plus[t], prob_minus[t]);
    }
    return bernoulli_product<P>(prob_plus, prob_minus, lane & ((1 << P) - 1));
}

// c_{r,t} = s_{r,t} - (2 p_t - 1) with s_{r,t} = +-1 the sign of corner r on
// plane t, so that d phi[r] / d u_t = beta * phi[r] * c_{r,t}. Written as
// 2 sigmoid(-2 beta u_t) for s = +1 and -2 sigmoid(2 beta u_t) for s = -1,
// which is exact algebra (1 - (2p - 1) = 2 (1 - p)) and avoids the
// cancellation of 1 - (2p - 1) when p is close to 1.
template <int P>
__device__ __forceinline__ float corner_coefficient(float prob_plus, float prob_minus, int corner,
                                                    int plane) {
    const bool plus = (corner >> (P - 1 - plane)) & 1;
    return plus ? 2.0f * prob_minus : -2.0f * prob_plus;
}

// Sum over the R = 2^P corners of a per-corner value held in corner layout:
// lane r holds the value of corner r & (R - 1), so every group of R
// consecutive lanes holds one copy of all R values. A butterfly over the low
// P lane bits sums one copy; the groups hold identical copies and run
// identical operations, and both partners at each level add the same two
// operands, so every lane ends with the same bitwise result.
template <int P>
__device__ __forceinline__ float corner_allreduce_sum(float value) {
#pragma unroll
    for (int offset = (1 << P) / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

// Warp-wide sum of per-lane partials, one sum per corner, delivered in corner
// layout: returns on lane l the sum over all 32 lanes of partial(l & (R - 1)).
//
// This is a reduce-scatter. At the level for lane bit b, each lane pairs every
// live value whose corner has bit b clear with the one whose corner has it
// set, keeps the value whose bit b matches its own lane bit b, adds the
// partner's copy of it (one shuffle at offset b), and sends the other; the
// live count halves every level and the lane ends with its own corner. The
// remaining 32 / R lane groups then combine with a plain butterfly over the
// high lane bits. Cost: R - 1 + (5 - P) shuffles for R sums.
//
// Corners are consumed in groups of G = 2^floor(P / 2) consecutive corners:
// the G partials of a group are folded over the low log2(G) lane bits right
// away, leaving one value per group, and the R / G group values are then
// folded over the next P - log2(G) lane bits. At most G + R / G values are
// live (12 at P = 5 instead of R / 2 = 16 for a plain top-down order), which
// keeps the P = 5 callers within 64 registers.
//
// `partial(r)` must return this lane's partial for corner r; it is called
// exactly once per corner, in the order 0, 1, ..., R - 1. Every lane with the
// same l & (R - 1) ends with the same bitwise result.
template <int P, typename PartialFn>
__device__ __forceinline__ float corner_reduce_scatter(int lane, PartialFn&& partial) {
    constexpr int kCorners = 1 << P;
    constexpr int kGroup = 1 << (P / 2);
    constexpr int kGroups = kCorners / kGroup;

    // Folds values[2 i] and values[2 i + 1] over lane bit `lane_bit` for
    // i < count, writing the results to values[i]. values[2 i + x] belongs to
    // the corner whose bit for this level is x.
    const auto fold_pairs = [lane](float* values, int count, int lane_bit) {
        const bool upper = lane & lane_bit;
#pragma unroll
        for (int i = 0; i < count; ++i) {
            const float send = upper ? values[2 * i] : values[2 * i + 1];
            const float keep = upper ? values[2 * i + 1] : values[2 * i];
            values[i] = keep + __shfl_xor_sync(kFullWarpMask, send, lane_bit);
        }
    };

    // group_sums[g]: corner g * G + (lane & (G - 1)), summed over the lanes
    // that differ from this one in the low log2(G) bits.
    float group_sums[kGroups];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
        float values[kGroup];
#pragma unroll
        for (int i = 0; i < kGroup; ++i) values[i] = partial(g * kGroup + i);
#pragma unroll
        for (int width = 1; width < kGroup; width <<= 1) {
            fold_pairs(values, kGroup / (2 * width), width);
        }
        group_sums[g] = values[0];
    }
    // The group index holds corner bits log2(G) .. P - 1. Folding them lowest
    // first means the bit of each level is always the lowest bit of the live
    // index, so fold_pairs pairs neighbouring entries, as in the groups.
#pragma unroll
    for (int width = 1; width < kGroups; width <<= 1) {
        fold_pairs(group_sums, kGroups / (2 * width), width * kGroup);
    }
#pragma unroll
    for (int offset = kCorners; offset < kWarpSize; offset <<= 1) {
        group_sums[0] += __shfl_xor_sync(kFullWarpMask, group_sums[0], offset);
    }
    return group_sums[0];
}

}  // namespace race
