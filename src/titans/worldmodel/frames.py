"""Temporal frame handling to build longer-span latent sequences (Phase 4 prep).

Raw videos are high frame-rate and consecutive frames are highly redundant, so
feeding every frame to the encoder wastes compute and gives short latent
sequences. Instead we *aggregate consecutive frames by averaging over a sliding
window with a stride*: this smooths away the redundancy and, by choosing a stride
> 1, lets a fixed number of "super-frames" span a much longer real-time window --
i.e. effectively longer videos, which is exactly what a long-memory model like
Titans needs to shine.

The two steps here are pure tensor ops (no model, no I/O), so they are cheap to
test in isolation:

    raw frames  [F, *spatial]
        |  aggregate_frames(window, stride)      (temporal average pool)
        v
    super-frames  [S, *spatial]
        |  split_into_clips(clip_len)            (chunk for the fixed-size encoder)
        v
    clips  [K, clip_len, *spatial]              (each clip -> encoder -> latents)

Both functions only touch the leading time axis and pass the trailing spatial
dims through unchanged, so they work for either layout -- ``[F, C, H, W]``
(channels-first, torch) or ``[F, H, W, C]`` (what decord decodes and what the
V-JEPA-2 video processor expects).
"""

from __future__ import annotations

import torch


def aggregate_frames(
    frames: torch.Tensor, window: int, stride: int
) -> torch.Tensor:
    """Average consecutive frames over a sliding ``window`` moved by ``stride``.

    ``frames`` is ``[F, C, H, W]``. Super-frame ``j`` is the mean of
    ``frames[j*stride : j*stride + window]``, giving
    ``S = 1 + (F - window) // stride`` output frames of shape ``[S, C, H, W]``.

    * ``window == stride`` -> non-overlapping average pool (classic temporal
      downsample);
    * ``stride > window`` -> gaps between windows (drops frames, longest span);
    * ``stride < window`` -> overlapping windows (smoother, shorter span).

    Integer inputs are averaged in floating point to avoid overflow/truncation.
    """
    if frames.dim() != 4:
        raise ValueError("frames must have shape [F, C, H, W]")
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    F = frames.shape[0]
    if F < window:
        raise ValueError(f"need at least window={window} frames, got F={F}")

    x = frames.float()
    # unfold along time -> [S, C, H, W, window], then mean over the window.
    windows = x.unfold(dimension=0, size=window, step=stride)
    return windows.mean(dim=-1)


def split_into_clips(
    frames: torch.Tensor, clip_len: int, drop_last: bool = True
) -> torch.Tensor:
    """Chunk a frame sequence ``[S, C, H, W]`` into clips ``[K, clip_len, C, H, W]``.

    The encoder consumes fixed-length clips, so a long (super-)frame sequence is
    split into ``K = S // clip_len`` consecutive, non-overlapping clips. The
    trailing remainder is dropped by default (``drop_last``); set it ``False`` to
    raise if the sequence is not an exact multiple of ``clip_len``.
    """
    if frames.dim() != 4:
        raise ValueError("frames must have shape [S, C, H, W]")
    S = frames.shape[0]
    K = S // clip_len
    if K == 0:
        raise ValueError(f"only {S} frames, fewer than one clip of {clip_len}")
    if not drop_last and S % clip_len != 0:
        raise ValueError(
            f"{S} frames is not a multiple of clip_len={clip_len} "
            "(pass drop_last=True to trim the remainder)"
        )
    frames = frames[: K * clip_len]
    C, H, W = frames.shape[1:]
    return frames.view(K, clip_len, C, H, W)
