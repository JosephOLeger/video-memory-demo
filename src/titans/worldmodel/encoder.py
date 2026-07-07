"""Optional frozen V-JEPA-2 encoder: video clips -> latent sequences (Phase 3).

This is the bridge from real video to the ``[B, T, latent_dim]`` latents that
:class:`~titans.worldmodel.model.WorldModel` consumes. It is kept deliberately
apart from the core: ``transformers`` is imported lazily, so the world model and
its demo run with zero extra dependencies -- you only need this file (and a ~1 GB
download) when you actually want to encode video.

Recipe (verified against ``transformers`` 5.13): the frozen
``facebook/vjepa2-vitl-fpc64-256`` encoder (the base self-supervised ViT-L; the
``fpc16`` names are all SSv2-finetuned classifiers, not what we want) maps a clip
``[B, num_frames, 3, 256, 256]`` to a flat token sequence
``last_hidden_state`` ``[B, N, 1024]``. The Conv3d patch-embed flattens in
**time-major** order (``t', h', w'``), so the ``N`` tokens split cleanly into
``T' = N / spatial_patches`` contiguous temporal groups; we spatially mean-pool
each group to one latent, giving ``T'`` latents per clip.

``T'`` is *derived* from the clip's actual token count and the encoder's spatial
grid (``(crop_size / patch_size) ** 2``), so a clip of any (tubelet-divisible)
length works -- nothing about the temporal length is hard-coded.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2Encoder(nn.Module):
    """Frozen V-JEPA-2 wrapper producing one latent per temporal group.

    The heavy model is loaded lazily on first use so that merely importing this
    module (or constructing the object) stays cheap and offline-friendly. After
    loading, :attr:`latent_dim` and :attr:`spatial_patches` describe the encoder.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        super().__init__()
        self.model_name = model_name
        # Default to CUDA when available -- the frozen ViT-L is the heavy step and
        # runs ~10-20x faster on GPU than CPU.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._backbone: nn.Module | None = None
        self.spatial_patches: int | None = None
        self.latent_dim: int | None = None

    def _lazy_load(self) -> nn.Module:
        if self._backbone is None:
            try:
                from transformers import VJEPA2Model
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "VJEPA2Encoder needs `transformers`; install it (and torch "
                    "vision deps) or use synthetic latents instead."
                ) from exc
            backbone = VJEPA2Model.from_pretrained(self.model_name)
            backbone.eval().to(self.device)
            for p in backbone.parameters():
                p.requires_grad_(False)
            cfg = backbone.config
            self.spatial_patches = (cfg.crop_size // cfg.patch_size) ** 2
            self.latent_dim = cfg.hidden_size
            self._backbone = backbone
        return self._backbone

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode ``clips`` ``[B, num_frames, 3, H, W]`` to ``[B, T', latent_dim]``.

        Frames are moved to the backbone's device; the returned latents live on
        the same device. ``num_frames`` may be any multiple of the tubelet size.
        """
        backbone = self._lazy_load()
        device = next(backbone.parameters()).device
        out = backbone(pixel_values_videos=clips.to(device))
        tokens = out.last_hidden_state  # [B, N, latent_dim]

        B, N, d = tokens.shape
        spatial = self.spatial_patches
        if N % spatial != 0:
            raise ValueError(
                f"token count {N} not divisible by spatial_patches={spatial}; "
                "the clip length may be incompatible with the encoder grid"
            )
        time_steps = N // spatial
        grouped = tokens.view(B, time_steps, spatial, d)  # [B, T', spatial, d]
        return grouped.mean(dim=2)  # spatial mean per temporal group -> [B, T', d]
