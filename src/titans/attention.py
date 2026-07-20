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

import math

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
    learned_qk_scale: opt-in learned attention temperature (default ``False`` ->
        bit-identical to the previous behaviour). Because Q and K are L2-normalised
        above, ``q.k`` is a *cosine* in ``[-1, 1]``; ``F.scaled_dot_product_attention``
        then applies its default ``1/sqrt(head_dim)`` on top, shrinking the logits a
        second time. At ``head_dim=16`` that is ``1/4``, so logits live in
        ``[-0.25, 0.25]`` and the softmax is nearly uniform -- attention degenerates
        into an *average* over its context window. Measured on this repo: max
        attention weight 0.0556 against 0.0500 for a perfectly uniform distribution,
        which drives the mean pairwise cosine of the attention output to ~0.90 even
        for i.i.d. input, and hence the memory's per-token gradients (outer products
        ``e_t (x) h0_t``) to be near-collinear -- a rank-1 momentum sum from the very
        first segment. The QK-normalisation literature (Henry et al. 2020; ViT-22B;
        Swin-V2) pairs the normalisation with a *learned* temperature for exactly
        this reason: normalised logits need scaling **up**, not down by
        ``1/sqrt(d)``. One scale per head, stored in log space and clamped at
        ``qk_scale_max`` so it cannot run away into a hard argmax.
    qk_scale_init, qk_scale_max: initial and maximum value of that temperature
        (defaults 10.0 / 100.0; Swin-V2's clamp is equivalent to 100).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        conv_kernel_size: int = 4,
        learned_qk_scale: bool = False,
        qk_scale_init: float = 10.0,
        qk_scale_max: float = 100.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if qk_scale_init <= 0 or qk_scale_max <= 0:
            raise ValueError("qk_scale_init and qk_scale_max must be > 0")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.learned_qk_scale = learned_qk_scale
        if learned_qk_scale:
            # Log-space so the scale stays positive under unconstrained gradient
            # descent; one entry per head so heads can specialise sharp vs broad.
            self.log_qk_scale = nn.Parameter(
                torch.full((num_heads,), math.log(qk_scale_init))
            )
            self.log_qk_scale_max = math.log(qk_scale_max)

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

        if self.learned_qk_scale:
            # Fold the per-head temperature into Q (SDPA's own ``scale`` is a
            # single scalar and cannot vary by head), then neutralise SDPA's
            # default 1/sqrt(head_dim) with scale=1.0 so the learned value is the
            # *only* temperature applied.
            scale = self.log_qk_scale.clamp(max=self.log_qk_scale_max).exp()
            q = q * scale.view(1, -1, 1, 1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, _, N, _ = out.shape
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        return self.out_proj(out)
