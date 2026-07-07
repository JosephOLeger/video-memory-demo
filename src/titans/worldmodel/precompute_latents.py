"""Precompute frozen V-JEPA-2 latents for raw videos and cache them to disk.

This is the end-to-end glue between real video and the world model:

    raw frames [F, H, W, C]
      -> aggregate_frames(window, stride)     temporal average-pool (longer span)
      -> split_into_clips(clip_len=64)         fixed-size clips for the encoder
      -> VJEPA2VideoProcessor                  resize 256 + normalise -> pixel_values
      -> VJEPA2Encoder (frozen)                -> T' latents per clip
      -> concatenate clips along time          -> [K * T', latent_dim] per video

Because the encoder is frozen and heavy, we run it *once* and cache the latents;
the world model then trains fast on the cached tensors (see
:func:`titans.worldmodel.train.train_demo`, which loads a cache when given one).

Runnable smoke test (uses the real encoder but synthetic videos, so no dataset
download -- only the ~1.2 GB encoder is fetched, then cached)::

    python -m titans.worldmodel.precompute_latents
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .encoder import VJEPA2Encoder
from .frames import aggregate_frames, split_into_clips

CLIP_LEN = 64  # V-JEPA-2 fpc64 native clip length (frames_per_clip)


def load_processor(model_name: str):
    """Load the matching V-JEPA-2 video processor (resize + normalise)."""
    from transformers import AutoVideoProcessor

    return AutoVideoProcessor.from_pretrained(model_name)


def video_to_latents(
    frames: torch.Tensor,
    encoder: VJEPA2Encoder,
    processor,
    *,
    agg_window: int,
    agg_stride: int,
    clip_len: int = CLIP_LEN,
) -> torch.Tensor:
    """Encode one raw video ``[F, H, W, C]`` to a latent sequence ``[T_total, d]``.

    Consecutive frames are averaged over a sliding ``agg_window`` moved by
    ``agg_stride`` (longer effective span), the result is chunked into
    ``clip_len``-frame clips, each clip is normalised by ``processor`` and encoded,
    and the clips' latents are concatenated along time: ``T_total = K * T'`` where
    ``K`` is the number of clips and ``T'`` the encoder's temporal groups per clip.
    """
    supers = aggregate_frames(frames, agg_window, agg_stride)  # [S, H, W, C]
    clips = split_into_clips(supers, clip_len)                 # [K, clip_len, H, W, C]

    per_clip = []
    for clip in clips:  # [clip_len, H, W, C]
        arr = clip.round().clamp(0, 255).to(torch.uint8).numpy()
        pv = processor(arr, return_tensors="pt")["pixel_values_videos"]
        z = encoder(pv)  # [1, T', d]
        per_clip.append(z[0])  # [T', d]
    return torch.cat(per_clip, dim=0)  # [K * T', d]


def precompute(
    videos: Iterable[torch.Tensor],
    out_path: str | Path,
    *,
    model_name: str | None = None,
    agg_window: int = 2,
    agg_stride: int = 2,
    clip_len: int = CLIP_LEN,
) -> torch.Tensor:
    """Encode a set of raw videos and save a stacked latent tensor to ``out_path``.

    All videos must yield the same number of latents (same frame count / agg
    params) so they stack into ``[num_videos, T_total, latent_dim]``. The saved
    file is a dict with the latents plus the settings used to produce them.
    """
    encoder = VJEPA2Encoder(model_name) if model_name else VJEPA2Encoder()
    processor = load_processor(encoder.model_name)

    seqs = [
        video_to_latents(
            v, encoder, processor,
            agg_window=agg_window, agg_stride=agg_stride, clip_len=clip_len,
        )
        for v in videos
    ]
    latents = torch.stack(seqs, dim=0)  # [num_videos, T_total, d]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": latents,
            "model_name": encoder.model_name,
            "agg_window": agg_window,
            "agg_stride": agg_stride,
            "clip_len": clip_len,
        },
        out_path,
    )
    return latents


def _synthetic_videos(n: int = 2, frames: int = 128, size: int = 64) -> list:
    """A few fake raw videos ``[F, H, W, C]`` in [0,255], for the smoke test."""
    g = torch.Generator().manual_seed(0)
    return [
        (torch.rand(frames, size, size, 3, generator=g) * 255) for _ in range(n)
    ]


if __name__ == "__main__":
    out = Path("cache/latents_demo.pt")
    latents = precompute(_synthetic_videos(), out, agg_window=2, agg_stride=2)
    print(f"cached latents {tuple(latents.shape)} -> {out}")
