"""Causal self-attention with the Titans §4.4 architectural details.

The paper's "attention as short-term memory" branch (shared across MAC/MAG/MAL)
adds three details on top of ordinary causal multi-head attention:

* **SiLU** activation applied to the query / key / value projections;
* **L2 normalisation** of Q and K before the dot product (cosine-style attention,
  which the paper reports stabilises the memory-augmented model);
* a **depthwise-separable 1D causal convolution** after each of the Q/K/V
  projections (a short-range mixing that the paper adopts from recent linear-
  attention / SSM work).

All three are implemented causally: the convolution is left-padded so position
``t`` only ever sees positions ``<= t``, and attention uses a causal mask. This
is essential for MAC's causality guarantee.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableCausalConv1d(nn.Module):
    """Depthwise (per-channel) + pointwise (1x1) causal conv over the seq dim.

    Input / output are ``[B, N, dim]``. Left-padding by ``kernel_size - 1`` makes
    it strictly causal (no look-ahead).
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, dim, N]
        x = F.pad(x, (self.kernel_size - 1, 0))  # causal: pad on the left only
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x.transpose(1, 2)  # [B, N, dim]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with the Titans §4.4 details.

    Parameters
    ----------
    dim: model width.
    num_heads: number of attention heads (must divide ``dim``).
    conv_kernel_size: kernel of the depthwise-separable causal conv (default 4,
        a documented choice -- the paper only says "a small kernel").
    """

    def __init__(self, dim: int, num_heads: int = 1, conv_kernel_size: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_conv = DepthwiseSeparableCausalConv1d(dim, conv_kernel_size)
        self.k_conv = DepthwiseSeparableCausalConv1d(dim, conv_kernel_size)
        self.v_conv = DepthwiseSeparableCausalConv1d(dim, conv_kernel_size)

    def _prep(self, x: torch.Tensor, proj: nn.Linear,
              conv: DepthwiseSeparableCausalConv1d) -> torch.Tensor:
        # SiLU on the projection, then causal conv, then split into heads.
        h = conv(F.silu(proj(x)))  # [B, N, dim]
        B, N, _ = h.shape
        return h.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, N, dim]`` -> ``[B, N, dim]`` (causal)."""
        q = self._prep(x, self.q_proj, self.q_conv)  # [B, H, N, hd]
        k = self._prep(x, self.k_proj, self.k_conv)
        v = self._prep(x, self.v_proj, self.v_conv)

        # Cosine-style attention: L2-normalise Q and K before the dot product.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, _, N, _ = out.shape
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        return self.out_proj(out)
