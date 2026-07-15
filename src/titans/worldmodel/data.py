"""Video frame decoding: ``.mp4`` -> frame tensor ``[F, H, W, C]`` (uint8).

Turns a video file into a raw frame tensor via ``decord``. Dataset-specific listing
and download live with each source (see :mod:`titans.worldmodel.egolife`); this
module only decodes, so it stays source-agnostic and import-light (``decord`` is
imported lazily).
"""

from __future__ import annotations

from typing import Optional

import torch


def load_video_frames(
    local_path: str, num_frames: Optional[int] = None
) -> torch.Tensor:
    """Decode an ``.mp4`` to a frame tensor ``[F, H, W, C]`` (uint8).

    If ``num_frames`` is given, exactly that many frames are returned, chosen by
    **uniform temporal sampling** across the whole clip. This is robust to variable
    clip lengths: longer clips are subsampled, and clips shorter than ``num_frames``
    simply repeat a few frames rather than raising. Every video therefore yields the
    same length, so the resulting latents stack cleanly. ``H``/``W`` may differ
    between videos -- the V-JEPA-2 processor resizes to a fixed crop downstream.
    """
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(local_path)
    total = len(vr)
    if num_frames is None:
        idx = list(range(total))
    else:
        # Uniform sample across [0, total-1]; handles total <, ==, or > num_frames.
        idx = torch.linspace(0, total - 1, num_frames).round().long().tolist()
    return vr.get_batch(idx)  # [num_frames, H, W, C] uint8
