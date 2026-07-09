"""Token model for the memory benchmark: embedding -> MAC blocks -> classifier.

A minimal sequence model over the ``TitansMACBlock`` we want to probe. With the
sequence much longer than ``segment_size``, causal attention only sees within a
segment, so solving a long-range recall task *forces* the long-term memory to
carry information across segments -- which is exactly what we want to measure.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mac import TitansMACBlock


class RecallModel(nn.Module):
    """Embed tokens, run stacked MAC blocks, predict a token per position.

    Parameters mirror :class:`~titans.mac.TitansMACBlock` (``segment_size``,
    ``num_heads``, ``memory_window_size`` for the ATLAS Omega rule, ...), forwarded
    via ``**block_kwargs`` so the benchmark can compare memory variants.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        depth: int = 2,
        segment_size: int = 8,
        num_heads: int = 1,
        max_len: int = 1024,
        **block_kwargs,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(max_len, dim) * 0.02)
        self.blocks = nn.ModuleList(
            TitansMACBlock(dim, segment_size=segment_size, num_heads=num_heads,
                           **block_kwargs)
            for _ in range(depth)
        )
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` tokens ``[B, L]`` -> logits ``[B, L, vocab_size]``."""
        B, L = x.shape
        if L > self.pos.shape[0]:
            raise ValueError(f"sequence length {L} exceeds max_len {self.pos.shape[0]}")
        h = self.embed(x) + self.pos[:L]
        for block in self.blocks:
            h, _, _ = block(h)
        return self.head(self.norm(h))
