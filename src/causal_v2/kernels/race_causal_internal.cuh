// Declarations shared by race_causal_fwd.cu (v2a and the public launchers)
// and race_causal_fwd_tc.cu (v2b). Internal to the two .cu files.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

#include "race_causal_fwd.h"
#include "../../noncausal/kernels/race_internal.cuh"

// v2b carry precision (docs/causal_v2_design.md section 6.3). With 1 the
// carry Phi_Q B uses B = B_hi + B_lo (two bf16 matrices, two MMAs), which
// keeps Num and Den consistent to fp32 accuracy, so a constant V comes back
// exactly. With 0 it uses B_hi only: fewer tensor-core FLOPs and less shared
// memory, at up to ~1e-3 relative carry error (design section 7.4).
#ifndef RACE_CAUSAL_PRECISE_CARRY
#define RACE_CAUSAL_PRECISE_CARRY 1
#endif

namespace race {
namespace causal {

// The smallest per-CTA opt-in shared memory among sm_80 (163 KiB), sm_89
// (99 KiB) and sm_90 (227 KiB). Used only to choose C, not to launch, so a
// shape gets the same C (and the same summation order) on every target.
constexpr size_t kPortableSmemBytes = 99 * 1024;

// Calls fn(integral_constant<D>, integral_constant<P>) for the runtime shape.
template <typename Fn>
cudaError_t dispatch_shape(const CausalShape& shape, Fn&& fn) {
    switch (shape.head_dim) {
        case 64: return detail::dispatch_planes<64>(shape.num_planes, fn);
        case 128: return detail::dispatch_planes<128>(shape.num_planes, fn);
        default: return cudaErrorInvalidValue;
    }
}

// Opts `kernel` in to `smem_bytes` of dynamic shared memory on the current
// device (refusing sizes above the device limit) and asks for the largest
// shared-memory carveout. The attributes are per device, so callers set them
// on every launch rather than caching them.
template <typename Kernel>
cudaError_t opt_in_shared_memory(Kernel kernel, size_t smem_bytes) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    int limit = 0;
    err = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (err != cudaSuccess) return err;
    if (smem_bytes > static_cast<size_t>(limit)) return cudaErrorInvalidConfiguration;
    if (smem_bytes > detail::kDefaultDynamicSmemBytes) {
        err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(smem_bytes));
        if (err != cudaSuccess) return err;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                cudaSharedmemCarveoutMaxShared);
}

// v2b (race_causal_fwd_tc.cu). The public launchers in race_causal_fwd.cu
// validate shapes and tile lengths before calling these.
namespace tc {

// Sub-chunk length C of the v2b kernels for (D, P): 64 or 32.
int sub_chunk_tokens(int head_dim, int num_planes);

// Dynamic shared memory of the v2b output pass in bytes.
size_t output_pass_smem_bytes(const CausalShape& shape);

// Resident CTAs per SM of the v2b output pass (after the shared-memory opt-in).
cudaError_t output_pass_occupancy(const CausalShape& shape, int* ctas_per_sm);

cudaError_t tile_sums(const __nv_bfloat16* k, const __nv_bfloat16* v, const float* planes,
                      const float* beta, float* workspace, const CausalShape& shape,
                      int tile_tokens, cudaStream_t stream);

cudaError_t output(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                   const float* planes, const float* beta, const float* workspace,
                   __nv_bfloat16* out, const CausalShape& shape, int tile_tokens,
                   cudaStream_t stream);

}  // namespace tc
}  // namespace causal
}  // namespace race
