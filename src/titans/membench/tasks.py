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
