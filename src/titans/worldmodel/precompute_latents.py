"""Precompute frozen V-JEPA-2 latents for raw videos and cache them to disk.

The end-to-end glue between real video and the world model, as a *continuous*
latent stream:

    raw frames [F, H, W, C]
      -> video_to_latents_sliding(clip_len, stride_frames)   overlapping windows
      -> VJEPA2VideoProcessor                                 resize 256 + normalise
      -> VJEPA2Encoder (frozen)                               -> T' latents / window
      -> keep each window's central band, tiled contiguously  -> [T_total, latent_dim]

Because the encoder is frozen and heavy, we run it *once* and cache the latents; the
world model then trains fast on the cached tensors.

Note on why we slide rather than temporally average: V-JEPA-2 is a motion-aware video
encoder, so averaging frames *before* the encoder (an earlier approach) destroys the
very temporal signal it reads. Instead we feed real frames and get continuity by
overlapping the encoder windows and keeping their well-contextualised central bands.

Runnable smoke test (real encoder, synthetic videos, no dataset download -- only the
~1.2 GB encoder is fetched, then cached)::

    python -m titans.worldmodel.precompute_latents
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .encoder import VJEPA2Encoder

CLIP_LEN = 64  # V-JEPA-2 fpc64 native clip length (frames_per_clip)


def load_processor(model_name: str):
    """Load the matching V-JEPA-2 video processor (resize + normalise)."""
    from transformers import AutoVideoProcessor

    return AutoVideoProcessor.from_pretrained(model_name)


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

    V-JEPA encodes each clip independently, so a long-memory model gets no continuity
    to exploit across clip seams. Instead we slide a ``clip_len``-frame window by
    ``stride_frames`` (with overlap) and keep only the *central* band of each window's
    latents: the positions that had full temporal context on both sides. Adjacent kept
    bands tile the video contiguously (``keep = stride_frames / tubelet`` latents per
    step), so the result is a smooth, continuous sequence -- each latent
    well-contextualised, seams softened by overlap.

    Smaller ``stride_frames`` = more overlap = more context per latent = smoother, at
    more encoder passes. ``stride_frames == clip_len`` recovers the discontinuous
    non-overlapping behaviour.
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
        # Keep the central band; extend to the very edges on the first/last window so
        # the whole video is covered.
        lo = 0 if wi == 0 else margin
        hi = Tp if wi == len(starts) - 1 else margin + keep
        bands.append(z[lo:hi])
    return torch.cat(bands, dim=0)  # [T_total, d]


def precompute(
    videos: Iterable[torch.Tensor],
    out_path: str | Path,
    *,
    model_name: str | None = None,
    clip_len: int = CLIP_LEN,
    sliding_stride: int = 32,
) -> torch.Tensor:
    """Encode a set of raw videos as continuous latent streams and save them.

    Every video is encoded via :func:`video_to_latents_sliding`; all must yield the
    same number of latents so they stack into ``[num_videos, T_total, latent_dim]``.
    The saved file is a dict with the latents plus the settings used to produce them.
    """
    encoder = VJEPA2Encoder(model_name) if model_name else VJEPA2Encoder()
    processor = load_processor(encoder.model_name)

    seqs = [
        video_to_latents_sliding(
            v, encoder, processor, clip_len=clip_len, stride_frames=sliding_stride
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
            "clip_len": clip_len,
            "sliding_stride": sliding_stride,
        },
        out_path,
    )
    return latents


def precompute_egolife(
    out_path: str | Path,
    *,
    persons: list[str],
    per_person: int,
    num_frames: int = 128,
    model_name: str | None = None,
    clip_len: int = CLIP_LEN,
    sliding_stride: int = 32,
) -> torch.Tensor:
    """Fetch EgoLife segments for the given persons, encode, and cache latents.

    Person-disjoint train/val split lives in the caller's choice of ``persons`` (see
    :data:`titans.worldmodel.egolife.TRAIN_PERSONS` / ``VAL_PERSONS``).
    """
    from .egolife import egolife_frames

    videos = egolife_frames(persons=persons, per_person=per_person, num_frames=num_frames)
    return precompute(
        videos, out_path, model_name=model_name,
        clip_len=clip_len, sliding_stride=sliding_stride,
    )


def _synthetic_videos(n: int = 2, frames: int = 128, size: int = 64) -> list:
    """A few fake raw videos ``[F, H, W, C]`` in [0,255], for the smoke test."""
    g = torch.Generator().manual_seed(0)
    return [
        (torch.rand(frames, size, size, 3, generator=g) * 255) for _ in range(n)
    ]


if __name__ == "__main__":
    out = Path("cache/latents_demo.pt")
    latents = precompute(_synthetic_videos(), out, sliding_stride=32)
    print(f"cached latents {tuple(latents.shape)} -> {out}")
