"""MAC -- Memory As Context (Titans, arXiv:2501.00663 -- Section 4.1, Eqs 19-25).

A ``TitansMACBlock`` processes a sequence in fixed-size segments. For each
segment it (1) retrieves *historical* context from the neural long-term memory
using the memory carried from earlier segments, (2) builds a local attention
context ``[persistent tokens; retrieved history; segment tokens]``, (3) runs
causal attention over it, (4) updates the long-term memory using the attention
output (so the memory learns "what attention decided was worth attending to"),
and (5) produces the segment output by gating the attention output with a fresh
retrieval from the just-updated memory.

The memory carry ``(state, S)`` is threaded across segments *sequentially*; the
parallelism lives inside each segment (attention + the memory's chunked update),
not across segments. See the module-level algorithm notes in the task/README.

Design choices where the paper is under-specified (documented per the README):

* **Update / output gate** (§4.4 "normalization and gating with a linear layer"):
  we RMSNorm each branch, then combine with a learnable per-channel sigmoid gate
  ``o = norm(y) * sigmoid(W_r norm(r)) + norm(r) * sigmoid(W_y norm(y))``.
* **Residual**: a residual connection wraps the whole block, ``x + out_proj(o)``,
  where ``out_proj`` is the single "final output projection" (§4.4).
* ``N % segment_size == 0`` is required (no padding logic this phase).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .memory import NeuralMemory, Params
from .persistent import PersistentMemory


class TitansMACBlock(nn.Module):
    """A single Memory-As-Context block.

    Parameters
    ----------
    dim: model width.
    num_heads: attention heads.
    segment_size: ``C``; the full sequence length must be a multiple of this.
    num_persistent_tokens: ``N_p`` learnable persistent-memory tokens.
    memory_hidden_dim, memory_num_layers: geometry of the memory MLP ``M``.
    memory_chunk_size: chunk size for the memory's online update within a segment
        (defaults to ``segment_size`` -- the whole segment as one parallel chunk).
    conv_kernel_size: kernel of the attention's causal conv (default 4).
    theta_scale: passed to the memory (scales the inner learning rate head).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        segment_size: int = 16,
        num_persistent_tokens: int = 4,
        memory_hidden_dim: Optional[int] = None,
        memory_num_layers: int = 2,
        memory_chunk_size: Optional[int] = None,
        conv_kernel_size: int = 4,
        theta_scale: float = 1.0,
        memory_window_size: int = 1,
        use_memory: bool = True,
        reset_memory_each_segment: bool = False,
        memory_use_cascade: bool = False,
        cascade_reservoirs: int = 4,
        cascade_beta1: float = 0.5,
        cascade_g: float = 2.0,
        cascade_selective: bool = False,
        cascade_topology: str = "chain",
        cascade_readout: Optional[str] = None,
        cascade_leak: str = "full",
        cascade_slow_decay: float = 0.99,
    ):
        super().__init__()
        self.dim = dim
        # Ablation switch: when False the long-term neural memory branch is
        # disabled -- no historical retrieval (Eq 21), no memory update / output
        # retrieval (Eq 24-25). Only core causal attention over
        # [persistent ; segment] (Eq 22-23) remains.
        self.use_memory = use_memory
        # Parameter-matched control: keep the *entire* memory branch (identical
        # params), but re-initialise its state at every segment so it carries no
        # information across segments. Since attention is confined to one segment
        # window, the threaded memory is the block's only long-range pathway --
        # so "threaded" vs "reset-each-segment" isolates the long-term memory's
        # contribution with zero parameter difference (unlike ``use_memory``).
        self.reset_memory_each_segment = reset_memory_each_segment
        self.segment_size = segment_size
        self.memory_chunk_size = memory_chunk_size or segment_size

        self.memory = NeuralMemory(
            dim,
            hidden_dim=memory_hidden_dim,
            num_layers=memory_num_layers,
            theta_scale=theta_scale,
            window_size=memory_window_size,
            use_cascade_momentum=memory_use_cascade,
            cascade_reservoirs=cascade_reservoirs,
            cascade_beta1=cascade_beta1,
            cascade_g=cascade_g,
            cascade_selective=cascade_selective,
            cascade_topology=cascade_topology,
            cascade_readout=cascade_readout,
            cascade_leak=cascade_leak,
            cascade_slow_decay=cascade_slow_decay,
        )
        self.persistent = PersistentMemory(dim, num_persistent_tokens)
        self.attn = CausalSelfAttention(dim, num_heads, conv_kernel_size)

        # Normalisation + gating before the final output projection (§4.4).
        self.norm_y = nn.RMSNorm(dim)
        self.norm_r = nn.RMSNorm(dim)
        self.gate_y = nn.Linear(dim, dim)
        self.gate_r = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _gate(self, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Combine attention output ``y`` and memory retrieval ``r`` per §4.4."""
        y_n = self.norm_y(y)
        r_n = self.norm_r(r)
        return y_n * torch.sigmoid(self.gate_r(r_n)) + r_n * torch.sigmoid(
            self.gate_y(y_n)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Params] = None,
        S: Optional[Params] = None,
    ) -> Tuple[torch.Tensor, Params, Params]:
        """Run the block over ``x`` of shape ``[B, N, d]``.

        ``state`` / ``S`` are the incoming memory carry; if ``None`` a fresh carry
        is created (``M_0``, zero momentum). Returns ``(output, final_state,
        final_S)`` so blocks can be stacked or state carried across batches of a
        longer stream in a future phase.
        """
        B, N, d = x.shape
        C = self.segment_size
        if d != self.dim:
            raise ValueError(f"input dim {d} != block dim {self.dim}")
        if N % C != 0:
            raise ValueError(
                f"sequence length N={N} must be a multiple of segment_size C={C} "
                "(padding is intentionally not handled this phase)"
            )
        if state is None or S is None:
            state, S = self.memory.init_memory(B)

        outs = []
        for i in range(N // C):
            seg = x[:, i * C : (i + 1) * C, :]  # [B, C, d]
            p = self.persistent(B)  # [B, N_p, d]

            if not self.use_memory:
                # Ablated: core attention over [persistent; segment] only,
                # output = the segment's attention (no memory branch).
                y_full = self.attn(torch.cat([p, seg], dim=1))
                outs.append(y_full[:, -C:, :])
                continue

            if self.reset_memory_each_segment:
                # Parameter-matched control: wipe the carry so nothing crosses
                # segment boundaries (memory becomes purely within-segment).
                state, S = self.memory.init_memory(B)

            # 1. Historical retrieval from the memory carried from earlier segs.
            q_hist = self.memory.q_proj(seg)  # [B, C, d]
            h_i = self.memory.retrieve(state, q_hist)  # [B, C, d]

            # 2. Local context: [persistent; retrieved history; segment].
            tilde = torch.cat([p, h_i, seg], dim=1)  # [B, N_p + C + C, d]

            # 3. Causal attention over the context window.
            y_full = self.attn(tilde)

            # 4. The segment's attention output = the last C positions.
            y_i = y_full[:, -C:, :]  # [B, C, d]

            # 5. Update long-term memory using the attention output y_i.
            state, S = self.memory.update(y_i, state, S, self.memory_chunk_size)

            # 6. Output retrieval from the just-updated memory, then gate.
            q_out = self.memory.q_proj(y_i)
            r_i = self.memory.retrieve(state, q_out)  # [B, C, d]
            outs.append(self._gate(y_i, r_i))

        block_out = torch.cat(outs, dim=1)  # [B, N, d]
        out = x + self.out_proj(block_out)  # residual around the whole block
        return out, state, S
