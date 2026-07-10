"""Synthetic long-range memory tasks -> (inputs, targets, loss mask).

Every generator returns a :class:`Batch` of freshly sampled data (synthetic, so
the model must learn the *algorithm*, not memorise a fixed set). Loss/accuracy are
scored only where ``mask`` is True -- the answer positions. Sequences are padded
at the front to a multiple of ``segment_size`` (the MAC block requires it), and
the padding is what pushes the queried item far from the answer so the long-term
memory -- not local attention -- has to carry it.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import torch

# distance: positions between the queried item and the answer (None if n/a).
# answer_classes: number of possible answer values -> chance accuracy = 1/answer_classes.
Batch = namedtuple("Batch", ["x", "y", "mask", "vocab_size", "distance", "answer_classes"])


def _gen(seed: Optional[int]) -> Optional[torch.Generator]:
    return torch.Generator().manual_seed(seed) if seed is not None else None


def _pad_len(core_len: int, segment_size: int) -> int:
    """Front-padding needed to make the total length a multiple of segment_size."""
    return (-core_len) % segment_size


def associative_recall(
    batch: int = 64,
    num_pairs: int = 8,
    segment_size: int = 8,
    key_vocab: int = 16,
    val_vocab: int = 16,
    seed: Optional[int] = None,
) -> Batch:
    """Show ``num_pairs`` (key, value) pairs, then query one key -> predict its value.

    Layout (before front-padding): ``k0 v0 k1 v1 ... QUERY kq`` and the answer
    ``vq`` is predicted at the final position. This is the most direct probe of an
    associative memory ``M(k) ~= v``. Keys are distinct within a sequence so the
    query is unambiguous; ``distance = 2*(num_pairs - j)`` grows for earlier pairs.
    """
    g = _gen(seed)
    B, n = batch, num_pairs
    if key_vocab < n:
        raise ValueError("key_vocab must be >= num_pairs (distinct keys per row)")
    QUERY = key_vocab + val_vocab
    PAD = QUERY + 1
    vocab = PAD + 1

    keys = torch.stack([torch.randperm(key_vocab, generator=g)[:n] for _ in range(B)])
    vals = torch.randint(0, val_vocab, (B, n), generator=g) + key_vocab
    seq = torch.empty(B, 2 * n, dtype=torch.long)
    seq[:, 0::2] = keys
    seq[:, 1::2] = vals

    j = torch.randint(0, n, (B,), generator=g)
    ar = torch.arange(B)
    qk, qv = keys[ar, j], vals[ar, j]
    core = torch.cat([seq, torch.full((B, 1), QUERY), qk[:, None]], dim=1)  # [B, 2n+2]

    pad = _pad_len(core.shape[1], segment_size)
    L = pad + core.shape[1]
    x = torch.full((B, L), PAD, dtype=torch.long)
    x[:, pad:] = core
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y[:, -1] = qv
    mask[:, -1] = True
    distance = 2 * (n - j)
    return Batch(x, y, mask, vocab, distance, val_vocab)


def copy_task(
    batch: int = 64,
    length: int = 16,
    segment_size: int = 8,
    content_vocab: int = 16,
    seed: Optional[int] = None,
) -> Batch:
    """Show ``length`` random tokens, a SEP, then reproduce them in order.

    Layout: ``c0..c_{L-1} SEP _ _ ... _`` and the target at each trailing slot is
    the corresponding ``c_i``. Tests ordered retention over a fixed lag (``length +
    1``). ``distance`` is None (constant lag per position).
    """
    g = _gen(seed)
    B, L = batch, length
    SEP = content_vocab
    PAD = content_vocab + 1
    vocab = content_vocab + 2

    content = torch.randint(0, content_vocab, (B, L), generator=g)
    core = torch.cat([content, torch.full((B, 1), SEP), torch.full((B, L), PAD)], dim=1)
    pad = _pad_len(core.shape[1], segment_size)
    x = torch.full((B, pad + core.shape[1]), PAD, dtype=torch.long)
    x[:, pad:] = core
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y[:, -L:] = content          # trailing L positions reproduce the content
    mask[:, -L:] = True
    return Batch(x, y, mask, vocab, None, content_vocab)


def selective_copy(
    batch: int = 64,
    num_tokens: int = 8,
    num_slots: int = 32,
    segment_size: int = 8,
    content_vocab: int = 16,
    seed: Optional[int] = None,
) -> Batch:
    """Copy only the *marked* content tokens, in order, ignoring BLANK filler.

    MAD "selective copying" (Poli et al., arXiv:2302.10866): ``num_tokens`` real
    content tokens are scattered at random positions among ``num_slots`` slots, the
    rest BLANK; after a SEP the model reproduces the content tokens *in order of
    appearance*. Because the content positions are random, positional copying is
    useless -- the model must content-address the non-blank slots -- which is what
    distinguishes real selection from a fixed positional shift (cf. plain
    ``copy_task``, where every position is content at a fixed lag). ``distance`` is
    None (multiple answer slots at mixed lags).
    """
    g = _gen(seed)
    B, k, S = batch, num_tokens, num_slots
    if k > S:
        raise ValueError("num_tokens must be <= num_slots")
    BLANK = content_vocab
    SEP = content_vocab + 1
    PAD = content_vocab + 2
    vocab = content_vocab + 3

    slots = torch.full((B, S), BLANK, dtype=torch.long)
    content = torch.randint(0, content_vocab, (B, k), generator=g)
    # choose k distinct slot positions per row, then sort so "order of appearance"
    # is well-defined and matches the left-to-right target order.
    pos = torch.stack([torch.randperm(S, generator=g)[:k] for _ in range(B)])
    pos, _ = pos.sort(dim=1)
    ar = torch.arange(B).unsqueeze(1)
    slots[ar, pos] = content

    core = torch.cat(
        [slots, torch.full((B, 1), SEP), torch.full((B, k), PAD)], dim=1
    )
    pad = _pad_len(core.shape[1], segment_size)
    x = torch.full((B, pad + core.shape[1]), PAD, dtype=torch.long)
    x[:, pad:] = core
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y[:, -k:] = content          # trailing k positions reproduce marked content
    mask[:, -k:] = True
    return Batch(x, y, mask, vocab, None, content_vocab)


def fuzzy_recall(
    batch: int = 64,
    num_pairs: int = 8,
    segment_size: int = 8,
    key_vocab: int = 32,
    val_vocab: int = 16,
    num_distractors: Optional[int] = None,
    seed: Optional[int] = None,
) -> Batch:
    """Associative recall with near-duplicate distractor keys (MAD fuzzy recall).

    Like :func:`associative_recall`, but every real key ``k`` is drawn from the
    *even* ids ``{0, 2, 4, ...}`` and shadowed by a near-duplicate lure key
    ``k + 1`` (its odd neighbour) that appears with a *different* value. The query
    is always a real (even) key, so the answer is unique; the odd near-duplicates
    are distractors whose ids sit right next to the real keys, so a memory that
    content-addresses coarsely (near-exact, not exact) is lured into the wrong
    value. This probes robustness of retrieval, not just exact-match lookup.

    Design note (documented per the project convention): token embeddings are
    *learned*, so "adjacent id" is not intrinsically "similar" -- the adjacency is
    a soft near-duplicate signal, and the dominant extra difficulty over plain
    ``associative_recall`` is the added same-neighbourhood interference. Uses the
    even/odd lattice split purely to guarantee real and lure keys never collide.
    ``num_distractors`` (default ``num_pairs``) sets how many real keys get a lure.
    ``distance = 2 * (num_pairs - j)`` as in ``associative_recall`` (over the real
    pairs only, before interleaving of lures).
    """
    g = _gen(seed)
    B, n = batch, num_pairs
    d = num_pairs if num_distractors is None else num_distractors
    if d > n:
        raise ValueError("num_distractors must be <= num_pairs")
    if key_vocab < 2 * n:
        raise ValueError("key_vocab must be >= 2*num_pairs (even/odd lattice split)")
    n_even = key_vocab // 2
    if n_even < n:
        raise ValueError("too few even keys for num_pairs")
    QUERY = key_vocab + val_vocab
    PAD = QUERY + 1
    vocab = PAD + 1

    # real keys: distinct even ids; values in the value band.
    even_ids = torch.arange(0, 2 * n_even, 2)
    keys = torch.stack(
        [even_ids[torch.randperm(n_even, generator=g)[:n]] for _ in range(B)]
    )
    vals = torch.randint(0, val_vocab, (B, n), generator=g) + key_vocab

    # near-duplicate lures: key+1 for the first ``d`` (post-shuffle) real keys,
    # each with a different value than its real neighbour.
    lure_sel = torch.stack(
        [torch.randperm(n, generator=g)[:d] for _ in range(B)]
    )
    ar1 = torch.arange(B).unsqueeze(1)
    lure_keys = keys[ar1, lure_sel] + 1
    lure_vals = (
        torch.randint(0, val_vocab, (B, d), generator=g) + key_vocab
    )

    # build the (real ++ lure) pair stream and shuffle pair order per row.
    all_keys = torch.cat([keys, lure_keys], dim=1)      # [B, n+d]
    all_vals = torch.cat([vals, lure_vals], dim=1)
    perm = torch.stack(
        [torch.randperm(n + d, generator=g) for _ in range(B)]
    )
    all_keys = all_keys[ar1, perm]
    all_vals = all_vals[ar1, perm]
    seq = torch.empty(B, 2 * (n + d), dtype=torch.long)
    seq[:, 0::2] = all_keys
    seq[:, 1::2] = all_vals

    # query a real key (index j into the real pairs, pre-shuffle).
    j = torch.randint(0, n, (B,), generator=g)
    ar = torch.arange(B)
    qk, qv = keys[ar, j], vals[ar, j]
    core = torch.cat([seq, torch.full((B, 1), QUERY), qk[:, None]], dim=1)

    pad = _pad_len(core.shape[1], segment_size)
    L = pad + core.shape[1]
    x = torch.full((B, L), PAD, dtype=torch.long)
    x[:, pad:] = core
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y[:, -1] = qv
    mask[:, -1] = True
    distance = 2 * (n - j)
    return Batch(x, y, mask, vocab, distance, val_vocab)


def needle(
    batch: int = 64,
    length: int = 64,
    segment_size: int = 8,
    filler_vocab: int = 16,
    payload_vocab: int = 16,
    seed: Optional[int] = None,
) -> Batch:
    """One (MARKER, payload) needle buried in filler; query at the end -> payload.

    A long filler haystack with a single marked payload inserted at a random depth,
    then a QUERY token. Retention of one item over a long, controllable distance --
    the classic needle-in-haystack probe. ``distance`` = positions from the payload
    to the answer.
    """
    g = _gen(seed)
    B, L = batch, length
    MARKER = filler_vocab
    QUERY = filler_vocab + 1
    PAD = filler_vocab + 2
    payload_base = filler_vocab + 3
    vocab = payload_base + payload_vocab

    filler = torch.randint(0, filler_vocab, (B, L), generator=g)
    payload = torch.randint(0, payload_vocab, (B,), generator=g) + payload_base
    # insert MARKER, payload at a random depth p in [0, L-2]
    p = torch.randint(0, L - 1, (B,), generator=g)
    ar = torch.arange(B)
    filler[ar, p] = MARKER
    filler[ar, p + 1] = payload
    core = torch.cat([filler, torch.full((B, 1), QUERY)], dim=1)  # [B, L+1]
    pad = _pad_len(core.shape[1], segment_size)
    Lp = pad + core.shape[1]
    x = torch.full((B, Lp), PAD, dtype=torch.long)
    x[:, pad:] = core
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y[:, -1] = payload
    mask[:, -1] = True
    distance = (L - 1 - (p + 1)) + 1  # payload position to the final QUERY answer
    return Batch(x, y, mask, vocab, distance, payload_vocab)
