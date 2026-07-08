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
    per_clip = [_encode_clip(clip, encoder, processor) for clip in clips]
    return torch.cat(per_clip, dim=0)  # [K * T', d]


def _encode_clip(clip: torch.Tensor, encoder: VJEPA2Encoder, processor) -> torch.Tensor:
    """Normalise one clip ``[clip_len, H, W, C]`` and encode it to ``[T', d]`` (CPU)."""
    arr = clip.round().clamp(0, 255).to(torch.uint8).numpy()
    pv = processor(arr, return_tensors="pt")["pixel_values_videos"]
    return encoder(pv)[0].cpu()  # [T', d] -- CPU-portable cache


def video_to_latents_sliding(
    frames: torch.Tensor,
    encoder: VJEPA2Encoder,
    processor,
    *,
    clip_len: int = CLIP_LEN,
    stride_frames: int = 32,
) -> torch.Tensor:
    """Encode one video as a *continuous* latent stream via overlapping windows.

    V-JEPA encodes each clip independently, so concatenating adjacent clips
    (:func:`video_to_latents`) hard-cuts the latent sequence every clip -- a
    long-memory model gets no continuity to exploit across those seams. Instead we
    slide a ``clip_len``-frame window by ``stride_frames`` (with overlap) and keep
    only the *central* band of each window's latents: the positions that had full
    temporal context on both sides. Adjacent kept bands tile the video
    contiguously (``keep = stride_frames / tubelet`` latents per step), so the
    result is a smooth, continuous sequence -- each latent well-contextualised,
    seams softened by overlap.

    Smaller ``stride_frames`` = more overlap = more context per latent = smoother,
    at more encoder passes. ``stride_frames == clip_len`` recovers the
    discontinuous non-overlapping behaviour.
    """
    F = frames.shape[0]
    starts = list(range(0, max(F - clip_len, 0) + 1, stride_frames)) or [0]

    bands = []
    for wi, s in enumerate(starts):
        clip = frames[s : s + clip_len]
        if clip.shape[0] < clip_len:  # pad the final short window by repeating
            pad = clip_len - clip.shape[0]
            clip = torch.cat([clip, clip[-1:].repeat(pad, 1, 1, 1)], dim=0)
        z = _encode_clip(clip, encoder, processor)  # [T', d]
        Tp = z.shape[0]
        tubelet = clip_len // Tp
        if stride_frames % tubelet != 0:
            raise ValueError(
                f"stride_frames={stride_frames} must be a multiple of tubelet={tubelet}"
            )
        keep = stride_frames // tubelet
        if keep > Tp:
            raise ValueError(
                f"stride_frames={stride_frames} too large: keep={keep} > "
                f"latents/clip={Tp}; use a smaller stride so windows overlap"
            )
        margin = (Tp - keep) // 2
        # Keep the central band; extend to the very edges on the first/last window
        # so the whole video is covered.
        lo = 0 if wi == 0 else margin
        hi = Tp if wi == len(starts) - 1 else margin + keep
        bands.append(z[lo:hi])
    return torch.cat(bands, dim=0)  # [T_total, d]


def precompute(
    videos: Iterable[torch.Tensor],
    out_path: str | Path,
    *,
    model_name: str | None = None,
    agg_window: int = 2,
    agg_stride: int = 2,
    clip_len: int = CLIP_LEN,
    sliding_stride: int | None = None,
) -> torch.Tensor:
    """Encode a set of raw videos and save a stacked latent tensor to ``out_path``.

    With ``sliding_stride`` set, videos are encoded as *continuous* latent streams
    via overlapping windows (:func:`video_to_latents_sliding`); otherwise the
    non-overlapping aggregate-and-chunk path (:func:`video_to_latents`) is used.
    All videos must yield the same number of latents so they stack into
    ``[num_videos, T_total, latent_dim]``. The saved file is a dict with the
    latents plus the settings used to produce them.
    """
    encoder = VJEPA2Encoder(model_name) if model_name else VJEPA2Encoder()
    processor = load_processor(encoder.model_name)

    if sliding_stride is not None:
        seqs = [
            video_to_latents_sliding(
                v, encoder, processor, clip_len=clip_len, stride_frames=sliding_stride,
            )
            for v in videos
        ]
    else:
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
            "sliding_stride": sliding_stride,
        },
        out_path,
    )
    return latents


def precompute_kinetics(
    out_path: str | Path,
    *,
    split: str = "train",
    classes: list[str] | None = None,
    per_class: int = 2,
    num_frames: int = 256,
    model_name: str | None = None,
    agg_window: int = 2,
    agg_stride: int = 2,
    clip_len: int = CLIP_LEN,
    sliding_stride: int | None = None,
) -> torch.Tensor:
    """Fetch Kinetics-mini clips, encode them, and cache the stacked latents.

    Thin wrapper tying :func:`titans.worldmodel.data.kinetics_frames` (download +
    decode) to :func:`precompute`. Every clip is decoded to the same
    ``num_frames``, so the latents stack into ``[num_videos, T_total, latent_dim]``.
    Pass ``sliding_stride`` to encode continuous latent streams (overlapping
    windows) instead of the discontinuous aggregate-and-chunk default.
    """
    from .data import kinetics_frames

    videos = kinetics_frames(
        split=split, classes=classes, per_class=per_class, num_frames=num_frames
    )
    return precompute(
        videos, out_path,
        model_name=model_name,
        agg_window=agg_window, agg_stride=agg_stride, clip_len=clip_len,
        sliding_stride=sliding_stride,
    )


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
