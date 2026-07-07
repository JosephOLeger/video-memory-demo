"""World model built on Titans MAC blocks (Phase 3, arXiv:2501.00663 application).

The idea is deliberately small and self-contained: given a sequence of latent
vectors ``Z_1, ..., Z_T`` (one per time step, e.g. produced by a frozen video
encoder), predict the *next* latent ``Z_{t+1}`` from everything seen up to ``t``.
The long-term memory of the Titans blocks is what lets the prediction depend on
far-past context rather than only the last few steps.

Shapes throughout: ``z`` is ``[B, T, latent_dim]``; internally the model works in
its own width ``dim`` and projects back to ``latent_dim`` at the end.

    prediction[:, t]  is the model's guess for  z[:, t + 1]

so a next-step loss compares ``model(z)[:, :-1]`` with ``z[:, 1:]`` (see
:func:`titans.worldmodel.train.next_step_loss`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mac import TitansMACBlock


class WorldModel(nn.Module):
    """Predict the next latent from past latents, with Titans long-term memory.

    Parameters
    ----------
    latent_dim: width of the input/target latent vectors (e.g. the encoder's
        feature dimension).
    dim: internal model width the MAC blocks operate in.
    depth: number of stacked :class:`~titans.mac.TitansMACBlock` layers.
    segment_size: ``C`` for every block; the sequence length ``T`` must be a
        multiple of this (the blocks raise a clear error otherwise).
    **block_kwargs: forwarded to each ``TitansMACBlock`` (e.g. ``num_heads``,
        ``num_persistent_tokens``, ``memory_num_layers``).
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int = 128,
        depth: int = 2,
        segment_size: int = 8,
        **block_kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim

        self.in_proj = nn.Linear(latent_dim, dim)
        self.blocks = nn.ModuleList(
            TitansMACBlock(dim, segment_size=segment_size, **block_kwargs)
            for _ in range(depth)
        )
        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latents ``z`` ``[B, T, latent_dim]`` to per-step next-latent guesses.

        Returns a tensor of the same shape where position ``t`` is the predicted
        ``z[:, t + 1]``. Each block manages its own memory carry internally (fresh
        per call); state is not threaded across calls in this phase.
        """
        h = self.in_proj(z)
        for block in self.blocks:
            h, _, _ = block(h)  # each block returns (out, state, S); carry unused
        return self.out_proj(self.out_norm(h))
