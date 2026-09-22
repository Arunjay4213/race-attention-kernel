// Fused causal RACE forward - Phase 3.
//
// One kernel does the whole forward pass the reference spreads over ~10
// PyTorch ops: hash projections, tanh, softmax over buckets, the causal
// running totals A/B, and the query-side blend. Nothing of size O(T * S * d)
// ever touches HBM; the only global traffic is reading Q/K/V once (bf16) and
// writing the output once (bf16).
//
// Contrast with the repo's shipped (dead) kernels/gpu/forward_kernel.cu:
//   - that kernel takes probsK/probsQ precomputed in fp32 ([N,T,S] in HBM),
//   - every timestep, all S bucket-threads atomicAdd into the same out[n,t,d]
//     element (S-way serialized contention),
//   - fp32 only.
// Here the bucket state lives in shared memory, the blend is a shared-memory
// reduction (no global atomics), and I/O is bf16 with fp32 accumulation.
//
// Layout / parallelization:
//   grid.x  = N streams (N = M*B*H). One block owns one stream's scan.
//   block   = 256 threads.
//   State in shared memory per block: A[L*R], Bst[L*R][DK] (padded +1 to
//   dodge bank conflicts), ~22 KB total at L=4.
//   The scan over T is sequential inside the block (that is what causal
//   running totals are); all per-token work is parallel across threads:
//     - Q/K/V token loads: coalesced bf16 reads by threads 0..191
//     - 32 hash projections (L*K for K and Q): one 8-thread subgroup each,
//       shuffle-reduced
//     - softmax over R per lane: 2*L threads
//     - A/B update: 256 threads cover L*R*DK accumulates
//     - output blend sum_s pQ[s]*B[s][d]/(A[s]+eps): thread-per-d serial over
//       S, then one coalesced bf16 store
//
// Fixed config (compile-time): DK=64, KBITS=4 (R=16), L<=4. That matches the
// benchmarked layer (d_k=64, K=4, L=4). Generalizing is a template exercise,
// not a design change.
//
// Forward only. Backward needs the same trick run right-to-left plus saved
// chunk checkpoints - future work.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#define DK 64
#define KBITS 4
#define RB 16          // R = 1 << KBITS
#define LMAX 4
#define THREADS 256
#define EPS 1e-6f

using bf16 = __nv_bfloat16;

__global__ void race_fused_fwd_kernel(
    const bf16 *__restrict__ Kin,   // [N, T, DK]
    const bf16 *__restrict__ Qin,   // [N, T, DK]
    const bf16 *__restrict__ Vin,   // [N, T, DK]
    const float *__restrict__ planes, // [L*KBITS, DK]  (row-major)
    bf16 *__restrict__ out,         // [N, T, DK]
    int T, int L)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int S = L * RB;
    const float scale = sqrtf((float)DK);

    __shared__ float sh_planes[LMAX * KBITS * DK];
    __shared__ float A[LMAX * RB];
    __shared__ float Bst[LMAX * RB][DK + 1];      // +1 pad: no bank conflicts
    __shared__ float ktok[DK], qtok[DK], vtok[DK];
    __shared__ float thK[LMAX * KBITS], thQ[LMAX * KBITS];
    __shared__ float pK[LMAX * RB], pQ[LMAX * RB];

    // load planes once; zero the running state
    for (int i = tid; i < L * KBITS * DK; i += THREADS) sh_planes[i] = planes[i];
    for (int i = tid; i < S; i += THREADS) A[i] = 0.0f;
    for (int i = tid; i < S * DK; i += THREADS) Bst[i / DK][i % DK] = 0.0f;
    __syncthreads();

    const long long base = (long long)n * T * DK;

    for (long long t = 0; t < T; ++t) {
        // ---- 1. load this token's q/k/v (coalesced bf16 -> fp32) ----
        const long long off = base + t * DK;
        if (tid < DK)                 ktok[tid]       = __bfloat162float(Kin[off + tid]);
        else if (tid < 2 * DK)        qtok[tid - DK]  = __bfloat162float(Qin[off + tid - DK]);
        else if (tid < 3 * DK)        vtok[tid - 128] = __bfloat162float(Vin[off + tid - 128]);
        __syncthreads();

        // ---- 2. hash projections: proj[l,k] = <tok, plane[l,k,:]> ----
        // 32 dot products (L*KBITS for K, same for Q), 8 threads each.
        {
            const int grp = tid >> 3;          // 0..31
            const int lane = tid & 7;
            const int lk = (grp < 16) ? grp : grp - 16;
            if (lk < L * KBITS) {
                const float *tok = (grp < 16) ? ktok : qtok;
                float part = 0.0f;
                for (int j = lane; j < DK; j += 8)
                    part += tok[j] * sh_planes[lk * DK + j];
                // mask = this 8-thread subgroup only (branch is uniform inside it)
                const unsigned sub = 0xFFu << (((tid & 31) >> 3) << 3);
                for (int o = 4; o > 0; o >>= 1)
                    part += __shfl_down_sync(sub, part, o, 8);
                if (lane == 0) {
                    float th = tanhf(part) / scale;
                    if (grp < 16) thK[lk] = th; else thQ[lk] = th;
                }
            }
        }
        __syncthreads();

        // ---- 3. bucket logits + softmax over R, per lane l ----
        // proto corner r: bit (KBITS-1-k) of r maps 0 -> -1, 1 -> +1.
        if (tid < 2 * L) {
            const int l = tid % L;
            const float *th = (tid < L) ? &thK[l * KBITS] : &thQ[l * KBITS];
            float *p = (tid < L) ? &pK[l * RB] : &pQ[l * RB];
            float logit[RB], mx = -1e30f;
            for (int r = 0; r < RB; ++r) {
                float z = 0.0f;
                for (int k = 0; k < KBITS; ++k)
                    z += (((r >> (KBITS - 1 - k)) & 1) ? th[k] : -th[k]);
                logit[r] = z;
                mx = fmaxf(mx, z);
            }
            float sum = 0.0f;
            for (int r = 0; r < RB; ++r) { logit[r] = expf(logit[r] - mx); sum += logit[r]; }
            for (int r = 0; r < RB; ++r) p[r] = logit[r] / sum;
        }
        __syncthreads();

        // ---- 4. update running totals A[s] += pK, B[s][d] += pK * v[d] ----
        for (int i = tid; i < S * DK; i += THREADS) {
            const int s = i / DK, d = i % DK;
            Bst[s][d] += pK[s] * vtok[d];
        }
        if (tid < S) A[tid] += pK[tid];
        __syncthreads();

        // ---- 5. blend: out[d] = sum_s pQ[s] * B[s][d] / (A[s]+eps) ----
        if (tid < DK) {
            float acc = 0.0f;
            for (int s = 0; s < S; ++s)
                acc += pQ[s] * Bst[s][tid] / (A[s] + EPS);
            out[off + tid] = __float2bfloat16(acc);
        }
        __syncthreads();
    }
}

torch::Tensor race_fused_fwd(torch::Tensor K, torch::Tensor Q, torch::Tensor V,
                             torch::Tensor planes, int64_t L)
{
    TORCH_CHECK(K.is_cuda() && Q.is_cuda() && V.is_cuda() && planes.is_cuda());
    TORCH_CHECK(K.scalar_type() == at::kBFloat16, "K/Q/V must be bf16");
    TORCH_CHECK(K.dim() == 3 && K.size(2) == DK, "K must be [N,T,64]");
    TORCH_CHECK(planes.scalar_type() == at::kFloat && planes.dim() == 2 &&
                planes.size(1) == DK && planes.size(0) == L * KBITS,
                "planes must be fp32 [L*4, 64]");
    TORCH_CHECK(L >= 1 && L <= LMAX, "L must be 1..4");
    TORCH_CHECK(K.is_contiguous() && Q.is_contiguous() && V.is_contiguous() &&
                planes.is_contiguous());

    const int N = K.size(0), T = K.size(1);
    auto out = torch::empty_like(K);
    race_fused_fwd_kernel<<<N, THREADS>>>(
        (const bf16 *)K.data_ptr(), (const bf16 *)Q.data_ptr(),
        (const bf16 *)V.data_ptr(), planes.data_ptr<float>(),
        (bf16 *)out.data_ptr(), T, (int)L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
