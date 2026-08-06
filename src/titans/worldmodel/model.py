"""World model built on Titans MAC blocks (Phase 3, arXiv:2501.00663 application).

The idea is deliberately small and self-contained: given a sequence of latent
vectors ``Z_1, ..., Z_T`` (one per time step, e.g. produced by a frozen video
encoder), predict the *next* latent ``Z_{t+1}`` from everything seen up to ``t``.
The long-term memory of the Titans blocks is what lets the prediction depend on
far-past context rather than only the last few steps.

Shapes throughout: ``z`` is ``[B, T, dim]`` and the model operates **directly** in
that latent space -- there is no learned input or output projection. The
dimensionality reduction is a property of the *dataset* (a fixed PCA basis, see
:mod:`titans.worldmodel.bottleneck`), so no learned linear map sits between the
data and the memory. The block's own ``out_proj`` residual (§4.4) still supplies a
learnable output map inside each MAC block; what is gone is the model-level I/O.

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
    latent_dim: width of the input/target latent vectors. Must equal ``dim``: the
        model has no I/O projection, so the data has to arrive already reduced.
        Kept as an explicit argument (rather than inferred) so a mismatched cache
        fails loudly at construction instead of silently later.
    dim: model width, which is also the latent width.
    depth: number of stacked :class:`~titans.mac.TitansMACBlock` layers.
    segment_size: ``C`` for every block; the sequence length ``T`` must be a
        multiple of this (the blocks raise a clear error otherwise).
    **block_kwargs: forwarded to each ``TitansMACBlock`` (e.g. ``num_heads``,
        ``num_persistent_tokens``, ``memory_num_layers``).

    The learned ``in_proj``/``out_norm``/``out_proj`` were removed on 2026-08-06.
    At ``dim=64`` over 1024-D latents they held 51% of the model's parameters, which
    made every memory comparison a comparison of I/O capacity instead; and the RMSNorm
    pinned every prediction to unit RMS while the targets kept their own scale. Runs
    in that space are retired -- results quoted against it are not comparable to the
    PCA space, and the corresponding caches are not usable here.
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
        if latent_dim != dim:
            raise ValueError(
                f"latent_dim must equal dim, got {latent_dim} != {dim}. The model runs "
                "directly in the latent space; reduce the latents first "
                "(e.g. bottleneck.pca_reduce_splits(k=dim))."
            )
        self.latent_dim = latent_dim
        self.dim = dim
        self.blocks = nn.ModuleList(
            TitansMACBlock(dim, segment_size=segment_size, **block_kwargs)
            for _ in range(depth)
        )

    def forward(self, z: torch.Tensor, carry=None, return_carry: bool = False,
                return_features: bool = False):
        """Map latents ``z`` ``[B, T, dim]`` to per-step next-latent guesses.

        Position ``t`` of the output predicts ``z[:, t + 1]``. Within a call the
        memory recurrence ``M_0 -> M_1 -> ...`` already threads across segments
        (paper Eq 21-25); ``carry`` lets that same continuous recurrence continue
        *across* calls, so a sequence too long for one forward can be processed in
        consecutive chunks (truncated BPTT) without resetting the memory.

        ``carry`` is a per-block list of ``(state, S)`` from a previous call (or
        ``None`` to start fresh). With ``return_carry=True`` the updated per-block
        carry is returned alongside the prediction as ``(pred, carry)``.

        ``return_features`` is now a no-op kept for call compatibility: with the I/O
        projection gone the features *are* the prediction, so both branches return the
        same tensor. It remains the natural seam for any future non-L1 output head.
        """
        if carry is None:
            carry = [(None, None)] * len(self.blocks)

        h = z
        new_carry = []
        for block, (state, S) in zip(self.blocks, carry):
            h, state, S = block(h, state, S)
            new_carry.append((state, S))

        return (h, new_carry) if return_carry else h
