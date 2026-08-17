# Where RACE Attention actually spends its memory

This is a trace of the reference implementation, [sahiljoshi515/RACE_Attention](https://github.com/sahiljoshi515/RACE_Attention), pinned at commit `b24a3d0` (2026-02-27).

The question I set out to answer was simple.
The repository ships CUDA kernels under `kernels/gpu/`, and it advertises "scalable kernel implementations" in its README.
So before writing a fused kernel of my own, I wanted to know what the existing kernels do and how much of the runtime they already cover.

The short answer is that they cover none of it.
The `.cu` files in that repository are not built, not bound to Python, and not called by any code path.
Every model in the repository runs a pure PyTorch implementation, and in the causal (autoregressive) case that implementation materializes a five-dimensional intermediate tensor whose size grows linearly with sequence length.
That tensor is the memory wall.

Everything below is traceable to a file and line in the reference repository.

## 1. The CUDA kernels are never reached

There are two CUDA source files:

- `kernels/gpu/forward_kernel.cu` (97 lines)
- `kernels/gpu/backward_kernels.cu` (256 lines)

Three separate things have to be true for a `.cu` file to run from Python, and none of them are true here.

**There is no build step.**
The only build script in the entire repository is `kernels/cpu/setup.py`, and it compiles C++ for the CPU, not CUDA.
A search of the whole tree for `nvcc`, `CUDAExtension`, `cuda_sources`, or any reference to the string `kernels/gpu` returns nothing.
`requirements.txt` lists only `torch`, `torchvision`, `transformers`, `datasets`, `tiktoken`, `tqdm`, and `matplotlib`; there is no build-time dependency that would compile a CUDA extension.

**There are no Python bindings.**
Neither `.cu` file contains a `PYBIND11_MODULE` block or a `TORCH_LIBRARY` registration.
`forward_kernel.cu` ends at line 97 with the closing brace of the host wrapper `race_fused_fwd`, and nothing follows it.
So even if someone compiled these files by hand, there would be no symbol for Python to import.

**Nothing imports them.**
Searching the full repository, including all four notebooks and the README, for `forward_kernel`, `backward_kernel`, `race_fused_fwd`, `race_bwd_kq_noscan`, or `race_bwd_v_noscan` produces zero matches outside the `.cu` files that define them.

The kernels are well-formed CUDA.
`race_fused_fwd_cuda` at `kernels/gpu/forward_kernel.cu:19` implements exactly the fusion this project is after: one thread walks the time axis for a fixed `(n, s, d)` triple, carrying the running sums `A` and `B` in registers, and never writes a prefix table to memory.

```
// forward_kernel.cu:35-52
float A = 0.0f;
float B = 0.0f;

for (int t = 0; t < T; ++t)
{
    ...
    A += pk;
    B += pk * v;

    float e = B / (A + eps);
    atomicAdd(&out[idxV], pq * e);
}
```

That is the right idea, and it is the idea this project builds on.
It is simply not connected to anything.
It reads as work-in-progress that was committed alongside the paper artifact but never wired into the models.

## 2. The CPU extension is also dead in the default path

Worth noting, because it is easy to mistake for the live path.

`kernels/cpu/race_pref.cpp` implements a real prefix-mean forward and backward, exported at `kernels/cpu/race_pref.cpp:224-225`.
`misc/race.py:7` imports it at module scope:

```python
from race_ext import race_pref
```

But `kernels/cpu/race_ext.py:23` and `:31` both pass an empty source list to the compiler:

```python
race_pref = load(
    name="race_pref",
    sources=[""], # Path to the .cpp file
    ...
)
```

The path was stripped before publication and never filled back in.
The sibling `kernels/cpu/setup.py:7` still carries the author's original absolute path, `/Users/sahiljoshi/Documents/Research/race_pref.cpp`, which confirms this is a packaging gap rather than a design choice.

More to the point, even a working CPU extension would not change the GPU story: the call site that would use it, `misc/race.py:114-133`, is commented out in favor of the PyTorch path above it.
The comment at `misc/race.py:114` says so directly: "If a user wants to run this on CPU - they can uncomment the following and comment out the #2 on top."

## 3. What actually runs

The live causal path is `BatchedACE.forward` in `misc/race.py:63-142`.
This is the class the language-modeling stack uses: `misc/lm.py:16` imports `RACEBlock` from `race`, `misc/race.py:179` wires `RACEBlock` to `RACEAttention`, and `misc/race.py:156` wires `RACEAttention` to `BatchedACE`.

The first half of the forward is cheap and linear, as advertised.
Keys and queries are projected onto `L*K` random hyperplanes with a single GEMM (`misc/race.py:79-80`), passed through `tanh`, and softmaxed against the `R = 2^K` bucket prototypes to give soft hash assignments (`misc/race.py:102-105`):

```python
logitsK = (projK.tanh().div(scale) @ self.protos_T)  # [N, T, L, R]
probsK  = F.softmax(logitsK, dim=-1)
```

Then comes the causal prefix step, `misc/race.py:108-110`:

```python
A_pref = probsK.cumsum(dim=1)                                                 # [N, T, L, R]
B_pref = (probsK.unsqueeze(-1) * V2.unsqueeze(2).unsqueeze(3)).cumsum(dim=1)  # [N, T, L, R, d_k]
E_pref = B_pref.div(A_pref.unsqueeze(-1).add(1e-6))                           # [N, T, L, R, d_k]
```

Line 109 is the bottleneck.

`B_pref` is five-dimensional: `[N, T, L, R, d_k]`, where `N = M * B * H` is the number of independent streams and `S = L * R` is the number of hash buckets.
Its size is `N * T * S * d_k` elements.
Two things make this expensive.

First, the broadcast multiply `probsK.unsqueeze(-1) * V2.unsqueeze(2).unsqueeze(3)` materializes the full `[N, T, L, R, d_k]` product *before* `cumsum` ever sees it.
So the peak is at least two tensors of that shape live at once, and `E_pref` at line 110 makes a third.

Second, the cumulative sum is computed and stored for every timestep, even though the only consumer needs one slice at a time.
That consumer is the lookup at `misc/race.py:135-138`:

```python
out2 = torch.bmm(
    probsQ.view(N*T, 1, S),                      # [N*T, 1, S]
    E_pref.contiguous().view(N*T, S, dk)         # [N*T, S, d_k]
).view(N, T, dk)
```

For each token `t`, this contracts `probsQ[:, t]` against `E_pref[:, t]` and throws the rest away.
Nothing downstream ever needs two different timesteps of `E_pref` simultaneously.
The entire `[N, T, L, R, d_k]` table exists only to hand one `[N, S, d_k]` slice per step to a `bmm`, which is exactly the pattern a fused kernel removes: carry the running `A` and `B` in registers, and emit the output for step `t` before moving on.

The `.contiguous()` on line 137 is worth flagging separately.
`E_pref` is the result of a broadcast division and is not contiguous in the layout `bmm` wants, so this call allocates yet another full-size copy of the five-dimensional tensor at the moment of peak memory pressure.

The arithmetic matches what the benchmark observes.
At the configuration used in the Phase 1 sweep (`M=1, B=1, H=8, d_k=64, K=4, L=4`, giving `N=8` streams and `S=64` buckets), one timestep of `B_pref` costs `N * S * d_k * 4` bytes, which is `8 * 64 * 64 * 4 = 131072` bytes, or 128 KB per token.
Peak memory therefore grows by roughly a fixed amount per token with no dependence on anything else, which is the straight line in the Phase 1 memory plot and the reason a 40 GB A100 runs out at 131072 tokens.

Note that this is linear in `T`, not quadratic.
RACE's linear-time claim is not in question here.
The problem is the constant in front: a linear-memory attention that spends 128 KB per token still hits a wall on one GPU, and it hits it much earlier than it needs to, because the tensor it is spending that memory on is never read more than once.

## 4. A note on scope

The blowup is specific to the causal path.
The non-causal variants used for classification and vision do not build a prefix table at all.
`scaling/benchmark_time.py:155-162` reduces over the whole sequence in one step:

```python
b_sum = probsK_S.transpose(1, 2).bmm(V2)   # [N,S,dk]
A     = probsK_S.sum(dim=1)                # [N,S]
E     = b_sum / (A.unsqueeze(-1) + eps)    # [N,S,dk]
out2  = probsQ_S.bmm(E)                    # [N,T,dk]
```

That version is already memory-light, because `E` is `[N, S, d_k]` with no `T` axis.
A search for `cumsum` across the repository confirms the split: it appears only in `misc/race.py:108-109`, in the corresponding inline copy inside `notebooks/LanguageModelling.ipynb`, and in the CPU benchmark harness `kernels/cpu/setup.py:45-46`.

Every training script in `misc/` carries its own inlined copy of `BatchedACE` rather than importing a shared one (`misc/mlm.py:149`, `misc/vit.py:124`, `misc/classification.py:257`, `misc/food-101.py:224`, `misc/arxiv_64K.py:584`, `scaling/benchmark_time.py:77`).
So there is no single place to fix this, and the causal copy in `misc/race.py` is the one that matters for long-context language modeling.

## Summary

| Claim | Verdict |
|---|---|
| `kernels/gpu/*.cu` are unused at runtime | Confirmed. No build script, no Python bindings, no call sites. |
| The real path is PyTorch `cumsum` + `matmul` | Confirmed. `misc/race.py:108-109` and `misc/race.py:135-138`. |
| It materializes a large intermediate tensor | Confirmed. `B_pref` is 5-D, `[N, T, L, R, d_k]`, plus a same-shape broadcast product, `E_pref`, and a `.contiguous()` copy. |
| That tensor is what caps context length | Consistent with measurement. 128 KB/token at the benchmark config; the A100-40GB sweep OOMs at T=131072. |

The fused kernel this project is building targets `misc/race.py:108-138` as one operation.
The reference `race_fused_fwd_cuda` at `kernels/gpu/forward_kernel.cu:19` shows the shape of the answer and is the natural starting point.
