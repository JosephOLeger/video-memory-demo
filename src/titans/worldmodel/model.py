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

    def forward(self, z: torch.Tensor, carry=None, return_carry: bool = False,
                return_features: bool = False):
        """Map latents ``z`` ``[B, T, latent_dim]`` to per-step next-latent guesses.

        Position ``t`` of the output predicts ``z[:, t + 1]``. Within a call the
        memory recurrence ``M_0 -> M_1 -> ...`` already threads across segments
        (paper Eq 21-25); ``carry`` lets that same continuous recurrence continue
        *across* calls, so a sequence too long for one forward can be processed in
        consecutive chunks (truncated BPTT) without resetting the memory.

        ``carry`` is a per-block list of ``(state, S)`` from a previous call (or
        ``None`` to start fresh). With ``return_carry=True`` the updated per-block
        carry is returned alongside the prediction as ``(pred, carry)``.

        With ``return_features=True`` the pre-output-projection features
        ``[B, T, dim]`` are returned instead of the latent prediction, so an
        alternative output head (e.g. the GMM/MDN head in :mod:`titans.worldmodel.gmm`)
        can consume them in place of the linear ``out_proj``. The default path is
        unchanged.
        """
        if carry is None:
            carry = [(None, None)] * len(self.blocks)

        h = self.in_proj(z)
        new_carry = []
        for block, (state, S) in zip(self.blocks, carry):
            h, state, S = block(h, state, S)
            new_carry.append((state, S))

        feat = self.out_norm(h)
        out = feat if return_features else self.out_proj(feat)
        return (out, new_carry) if return_carry else out
