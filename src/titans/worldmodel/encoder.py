"""Optional frozen V-JEPA-2 encoder: video clips -> latent sequences (Phase 3).

This is the bridge from real video to the ``[B, T, latent_dim]`` latents that
:class:`~titans.worldmodel.model.WorldModel` consumes. It is kept deliberately
apart from the core: ``transformers`` is imported lazily, so the world model and
its demo run with zero extra dependencies -- you only need this file (and a big
download) when you actually want to encode video.

Recipe (per the Phase-3 plan): run the frozen ``facebook/vjepa2-vitl-fpc16-256``
encoder on 16-frame clips, then collapse each temporal group of patches to a
single vector by spatial mean-pooling, giving ``T`` latents per clip.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc16-256"


class VJEPA2Encoder(nn.Module):
    """Frozen V-JEPA-2 wrapper producing one latent per temporal group.

    Parameters
    ----------
    model_name: HF id of the V-JEPA-2 checkpoint.
    time_steps: ``T`` -- number of temporal groups to pool the patch grid into.

    The heavy model is loaded lazily on first use so that merely importing this
    module (or constructing the object) stays cheap and offline-friendly.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, time_steps: int = 8):
        super().__init__()
        self.model_name = model_name
        self.time_steps = time_steps
        self._backbone: nn.Module | None = None

    def _lazy_load(self) -> nn.Module:
        if self._backbone is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "VJEPA2Encoder needs `transformers`; install it (and torch "
                    "vision deps) or use synthetic latents instead."
                ) from exc
            backbone = AutoModel.from_pretrained(self.model_name)
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad_(False)
            self._backbone = backbone
        return self._backbone

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode video ``clips`` ``[B, frames, C, H, W]`` to ``[B, T, latent_dim]``.

        The encoder returns a flat sequence of patch tokens; we reshape it into
        ``time_steps`` temporal groups and spatially mean-pool each group.
        """
        backbone = self._lazy_load()
        out = backbone(clips)
        tokens = out.last_hidden_state  # [B, num_patches, latent_dim]

        B, num_patches, d = tokens.shape
        if num_patches % self.time_steps != 0:
            raise ValueError(
                f"patch count {num_patches} not divisible by time_steps="
                f"{self.time_steps}; adjust time_steps to the encoder's grid"
            )
        grouped = tokens.view(B, self.time_steps, -1, d)  # [B, T, patches/T, d]
        return grouped.mean(dim=2)  # spatial mean per temporal group -> [B, T, d]
