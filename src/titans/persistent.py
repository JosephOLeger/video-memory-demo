"""Persistent memory tokens (Titans, arXiv:2501.00663 -- Section 4.3 / 4.1).

A small set of *input-independent*, learnable tokens prepended to every segment's
attention context. They act as task-level "memory" / attention sinks: the model
learns fixed knowledge that is always visible, independent of the current input.
Being ordinary ``nn.Parameter``s they are trained by the outer optimizer, unlike
the neural long-term memory whose weights are learned online at test time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    """``num_tokens`` learnable tokens of width ``dim``, shared across the batch."""

    def __init__(self, dim: int, num_tokens: int):
        super().__init__()
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        self.dim = dim
        self.num_tokens = num_tokens
        # Small init: these are prepended context, not meant to dominate at start.
        self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

    def forward(self, batch: int) -> torch.Tensor:
        """Return the persistent tokens expanded to ``[batch, num_tokens, dim]``."""
        return self.tokens.unsqueeze(0).expand(batch, self.num_tokens, self.dim)
