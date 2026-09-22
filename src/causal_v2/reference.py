"""Reference implementations of causal RACE Attention and a CPU model of the v2 kernels.

Ground truth (docs/causal_v2_design.md section 2.3): paper Algorithm 1 made causal,
with one global normalization per query:

    O_i = sum_{j<=i} k_ij v_j / sum_{j<=i} k_ij,   k_ij = <Phi(q_i), Phi(k_j)>

where Phi(x) in R^S (S = L * R) concatenates the L tables' corner
probabilities phi_l(x) = softmax_r(beta * <tanh(W_l x), c_r>), table-major.
The paper's 1/L factors multiply numerator and denominator and cancel.

Three functions compute causal outputs:

  - race_causal_reference: Algorithm 1 from its prefix-sum definition,
        B_i = sum_{j<=i} Phi_K,j (x) v_j,   A_i = sum_{j<=i} Phi_K,j,
        O_i = (Phi_Q,i . B_i) / (Phi_Q,i . A_i),
    evaluated with torch.cumsum in blocks of tokens so memory stays
    O(block * S * d) at any T. It uses the direct length-R softmax for phi, so
    a kernel test against it also checks the Bernoulli product factorization.
  - race_causal_dense_reference: the same estimator as a masked T x T matrix,
    for tiny T only. It exists to check race_causal_reference independently.
  - race_causal_perbucket_reference: the repo's per-bucket normalization,
        O_i = sum_s Phi_Q,i,s B_i,s / (A_i,s + eps)   (no 1/L),
    which is what src/race_baseline.py computes. It is the "bridge": matching
    race_baseline in fp64 proves this module uses the repo's planes, corner
    order and beta convention (beta = 1/scale, the repo freezes 1/sqrt(d)).

race_causal_chunked_emulation models the kernels' decomposition (plan section
3.3): K1 tile totals over tiles of `tile_tokens`, K2 exclusive scan over tiles
in order, K3 per tile walking sub-chunks of `chunk_tokens` with an exclusive
carry, a causal mask that includes the diagonal, and a state update after each
sub-chunk. It follows the kernels' summation structure (not their exact fp32
operation order) and returns the per-tile prefix states the debug binding
exposes, so the design can be tested on CPU and the GPU states compared.

Layouts: q, k, v are [B, H, T, d]; W is [L, P, d] (plane t of table l is
W[l, t]); beta is a 0-dim tensor. Corner r has +1 on plane t iff bit
(P - 1 - t) of r is set, the itertools.product([-1, +1]) order the repo uses.

A zero denominator (only possible after fp underflow at very large beta)
gives an output row of 0, the convention of the non-causal kernels.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import NamedTuple

import torch


def _load_noncausal_reference():
    # The non-causal reference defines the corner probabilities; load it by
    # path under a distinct module name so it never collides with this file,
    # which is also called reference.py.
    path = pathlib.Path(__file__).resolve().parents[1] / "noncausal" / "reference.py"
    spec = importlib.util.spec_from_file_location("race_noncausal_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_noncausal = _load_noncausal_reference()
corner_probs_softmax = _noncausal.corner_probs_softmax
corner_probs_bernoulli = _noncausal.corner_probs_bernoulli

DEFAULT_BLOCK_TOKENS = 256
PERBUCKET_EPS = 1e-6  # the eps of src/race_baseline.py


def features(x: torch.Tensor, W: torch.Tensor, beta: torch.Tensor, *, bernoulli: bool = False) -> torch.Tensor:
    """Phi(x): [..., d] -> [..., S], the L tables' corner probabilities, table-major."""
    probs = corner_probs_bernoulli(x, W, beta) if bernoulli else corner_probs_softmax(x, W, beta)
    return probs.flatten(-2)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """numerator / denominator row by row, 0 where the denominator is exactly 0."""
    empty = denominator == 0
    safe = torch.where(empty, torch.ones_like(denominator), denominator)
    return torch.where(empty, torch.zeros_like(numerator), numerator / safe)


def _prefix_states(phi_k: torch.Tensor, v: torch.Tensor, block_tokens: int):
    """Yields (begin, end, A_prefix [.., n, S], B_prefix [.., n, S, d]) block by block.

    A_prefix[..., t, :] and B_prefix[..., t, :, :] are the inclusive prefix
    states after token begin + t, built as a carried total plus a cumsum over
    the block, which is the definition of the prefix sum.
    """
    seq_len = phi_k.shape[-2]
    carry_mass = phi_k.new_zeros(*phi_k.shape[:-2], phi_k.shape[-1])
    carry_values = phi_k.new_zeros(*phi_k.shape[:-2], phi_k.shape[-1], v.shape[-1])
    for begin in range(0, seq_len, block_tokens):
        end = min(seq_len, begin + block_tokens)
        block_phi = phi_k[..., begin:end, :]
        block_values = v[..., begin:end, :]
        mass = carry_mass.unsqueeze(-2) + torch.cumsum(block_phi, dim=-2)
        weighted = block_phi.unsqueeze(-1) * block_values.unsqueeze(-2)
        values = carry_values.unsqueeze(-3) + torch.cumsum(weighted, dim=-3)
        yield begin, end, mass, values
        carry_mass = mass[..., -1, :]
        carry_values = values[..., -1, :, :]


def race_causal_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: torch.Tensor, beta: torch.Tensor,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
) -> torch.Tensor:
    """Causal Algorithm 1 (global normalization) from the prefix-sum definition.

    Args:
        q, k, v: [B, H, T, d]. W: [L, P, d], same dtype. beta: 0-dim tensor.
        block_tokens: tokens per cumsum block; memory is O(B*H*block*S*d).

    Returns:
        o: [B, H, T, d].
    """
    phi_q = features(q, W, beta)
    phi_k = features(k, W, beta)
    out = torch.empty_like(v)
    for begin, end, mass, values in _prefix_states(phi_k, v, block_tokens):
        query = phi_q[..., begin:end, :]
        numerator = torch.einsum("...ts,...tsd->...td", query, values)
        denominator = torch.einsum("...ts,...ts->...t", query, mass).unsqueeze(-1)
        out[..., begin:end, :] = _safe_divide(numerator, denominator)
    return out


def race_causal_dense_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Causal Algorithm 1 as masked attention with kernel k_ij = <Phi_Q,i, Phi_K,j>.

    O(T^2) memory; for tiny T only. The mask keeps j <= i, diagonal included.
    """
    scores = features(q, W, beta) @ features(k, W, beta).transpose(-1, -2)
    seq_len = q.shape[-2]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
    scores = torch.where(causal, scores, torch.zeros_like(scores))
    return _safe_divide(scores @ v, scores.sum(-1, keepdim=True))


def race_causal_perbucket_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: torch.Tensor, beta: torch.Tensor,
    eps: float = PERBUCKET_EPS, block_tokens: int = DEFAULT_BLOCK_TOKENS,
) -> torch.Tensor:
    """The repo's per-bucket causal normalization (src/race_baseline.py), the bridge.

    O_i = sum_l sum_r phi_l(q_i)[r] * B_i,l,r / (A_i,l,r + eps), with no 1/L.
    Equals BatchedACE.forward when beta = 1/sqrt(d) and W is its planes.
    """
    phi_q = features(q, W, beta)
    phi_k = features(k, W, beta)
    out = torch.empty_like(v)
    for begin, end, mass, values in _prefix_states(phi_k, v, block_tokens):
        normalized = values / (mass.unsqueeze(-1) + eps)
        out[..., begin:end, :] = torch.einsum("...ts,...tsd->...td", phi_q[..., begin:end, :], normalized)
    return out


class ChunkedEmulation(NamedTuple):
    """Outputs of race_causal_chunked_emulation, shaped like the debug binding's.

    out:           [B, H, T, d]
    prefix_mass:   [B, H, tiles, L, R]     exclusive prefix A at each tile start (after K2)
    prefix_values: [B, H, tiles, L, R, d]  exclusive prefix B at each tile start
    final_mass:    [B, H, L, R]            A over all T tokens
    final_values:  [B, H, L, R, d]         B over all T tokens
    """

    out: torch.Tensor
    prefix_mass: torch.Tensor
    prefix_values: torch.Tensor
    final_mass: torch.Tensor
    final_values: torch.Tensor


def race_causal_chunked_emulation(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: torch.Tensor, beta: torch.Tensor,
    tile_tokens: int, chunk_tokens: int,
) -> ChunkedEmulation:
    """The v2 kernels' decomposition in PyTorch, in the dtype of the inputs.

    K1: per tile, A_t = sum Phi_K and B_t = Phi_K^T V over the tile's tokens.
    K2: exclusive scan over tiles in index order: prefix_t = sum_{t' < t} (A, B)_t'.
    K3: every tile starts from its prefix and walks its sub-chunks in order.
        For a sub-chunk with rows Phi_Q, Phi_K, V (C rows each):
            G   = tril(Phi_Q Phi_K^T)               (diagonal included)
            Den = Phi_Q A + rowsum(G)
            Num = Phi_Q B + G V
            O   = Num * (1 / Den)
            A  += colsum(Phi_K),   B += Phi_K^T V   (sub-chunk partial, then added)
    Rows past T contribute Phi_K = 0 and v = 0, as the kernels arrange; zero
    keys are not enough because phi(0) = 1/R per corner, not 0.

    All tiles run their sub-chunk loops in lockstep here (vectorized over the
    tile axis), which is equivalent because tiles are independent once their
    prefixes are known.
    """
    if tile_tokens <= 0 or chunk_tokens <= 0 or tile_tokens % chunk_tokens != 0:
        raise ValueError("tile_tokens must be a positive multiple of chunk_tokens")
    batch, heads, seq_len, head_dim = q.shape
    num_tables, num_planes = W.shape[0], W.shape[1]
    corners = 1 << num_planes
    num_tiles = max(1, -(-seq_len // tile_tokens))
    padded = num_tiles * tile_tokens

    def pad_rows(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, (0, 0, 0, padded - seq_len))

    def by_tile(x: torch.Tensor) -> torch.Tensor:
        # [B, H, padded, n] -> [B, H, tiles, tile_tokens, n]
        return x.reshape(batch, heads, num_tiles, tile_tokens, x.shape[-1])

    phi_q = by_tile(pad_rows(features(q, W, beta, bernoulli=True)))
    phi_k = by_tile(pad_rows(features(k, W, beta, bernoulli=True)))  # zero past T
    values = by_tile(pad_rows(v))

    # K1: tile totals.
    tile_mass = phi_k.sum(dim=-2)                                   # [B, H, tiles, S]
    tile_values = phi_k.transpose(-1, -2) @ values                  # [B, H, tiles, S, d]

    # K2: exclusive scan over tiles, sequential in tile order.
    prefix_mass = torch.empty_like(tile_mass)
    prefix_values = torch.empty_like(tile_values)
    running_mass = torch.zeros_like(tile_mass[:, :, 0])
    running_values = torch.zeros_like(tile_values[:, :, 0])
    for tile in range(num_tiles):
        prefix_mass[:, :, tile] = running_mass
        prefix_values[:, :, tile] = running_values
        running_mass = running_mass + tile_mass[:, :, tile]
        running_values = running_values + tile_values[:, :, tile]

    # K3: sub-chunk loop per tile, all tiles at once.
    mass = prefix_mass.clone()
    weighted = prefix_values.clone()
    causal = torch.ones(chunk_tokens, chunk_tokens, dtype=torch.bool, device=q.device).tril()
    out = torch.empty(batch, heads, num_tiles, tile_tokens, head_dim, dtype=v.dtype, device=v.device)
    for begin in range(0, tile_tokens, chunk_tokens):
        rows = slice(begin, begin + chunk_tokens)
        chunk_q, chunk_k, chunk_v = phi_q[..., rows, :], phi_k[..., rows, :], values[..., rows, :]
        gram = chunk_q @ chunk_k.transpose(-1, -2)
        gram = torch.where(causal, gram, torch.zeros_like(gram))
        denominator = (chunk_q @ mass.unsqueeze(-1)).squeeze(-1) + gram.sum(-1)
        numerator = chunk_q @ weighted + gram @ chunk_v
        inverse = torch.where(denominator == 0, torch.zeros_like(denominator), 1.0 / denominator)
        out[..., rows, :] = numerator * inverse.unsqueeze(-1)
        mass = mass + chunk_k.sum(dim=-2)
        weighted = weighted + chunk_k.transpose(-1, -2) @ chunk_v

    out = out.reshape(batch, heads, padded, head_dim)[:, :, :seq_len]
    return ChunkedEmulation(
        out=out,
        prefix_mass=prefix_mass.reshape(batch, heads, num_tiles, num_tables, corners),
        prefix_values=prefix_values.reshape(batch, heads, num_tiles, num_tables, corners, head_dim),
        final_mass=running_mass.reshape(batch, heads, num_tables, corners),
        final_values=running_values.reshape(batch, heads, num_tables, corners, head_dim),
    )
