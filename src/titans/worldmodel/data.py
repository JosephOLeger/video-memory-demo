"""Kinetics-mini loader: fetch .mp4 clips and decode them to frame tensors.

The dataset ``nateraw/kinetics-mini`` is just a folder of raw ``.mp4`` files laid
out as ``<split>/<class>/<id>.mp4`` (train/val, 5 classes x 10 clips each, ~10 s
at 30 fps). It is *not* a parquet ``datasets`` dataset, so we list files via the
Hub API, download the ones we want, and decode frames with ``decord``.

This module only turns videos into raw frame tensors ``[F, H, W, C]`` (uint8) --
the temporal aggregation, clipping, and encoding live in :mod:`frames` /
:mod:`precompute_latents`, so decoding stays cleanly separated from encoding.

Both ``huggingface_hub`` and ``decord`` are imported lazily, keeping the rest of
the package import-light and offline-friendly.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch

DATASET_ID = "nateraw/kinetics-mini"


def list_clips(
    split: str = "train",
    classes: Optional[list[str]] = None,
    per_class: Optional[int] = None,
) -> list[str]:
    """List clip paths ``<split>/<class>/<id>.mp4`` in the dataset repo.

    ``classes`` restricts to those label folders (default: all); ``per_class``
    caps how many clips per class are returned (default: all). Paths come back
    sorted and grouped by class so ``per_class`` is a deterministic prefix.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
    clips = sorted(f for f in files if f.startswith(f"{split}/") and f.endswith(".mp4"))

    kept: list[str] = []
    seen: dict[str, int] = {}
    for f in clips:
        cls = f.split("/")[1]
        if classes is not None and cls not in classes:
            continue
        if per_class is not None and seen.get(cls, 0) >= per_class:
            continue
        seen[cls] = seen.get(cls, 0) + 1
        kept.append(f)
    return kept


def download_clip(path: str) -> str:
    """Download one clip from the dataset repo, returning its local file path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET_ID, path, repo_type="dataset")


def load_video_frames(
    local_path: str, num_frames: Optional[int] = None
) -> torch.Tensor:
    """Decode an ``.mp4`` to a frame tensor ``[F, H, W, C]`` (uint8).

    If ``num_frames`` is given, the first ``num_frames`` frames are kept (a clear
    error if the video is shorter), so every video yields the same length and the
    resulting latents stack cleanly. ``H``/``W`` may differ between videos -- the
    V-JEPA-2 processor resizes to a fixed crop downstream.
    """
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(local_path)
    total = len(vr)
    if num_frames is not None:
        if total < num_frames:
            raise ValueError(
                f"{local_path} has {total} frames, fewer than num_frames={num_frames}"
            )
        total = num_frames
    return vr.get_batch(range(total))  # [F, H, W, C] uint8


def kinetics_frames(
    split: str = "train",
    classes: Optional[list[str]] = None,
    per_class: Optional[int] = 2,
    num_frames: int = 256,
) -> Iterator[torch.Tensor]:
    """Yield decoded frame tensors ``[num_frames, H, W, C]`` for selected clips.

    Downloads each clip on demand (cached by the Hub) and decodes a fixed
    ``num_frames`` per video so the downstream latent sequences all share one
    length ``T``. Ready to hand straight to
    :func:`titans.worldmodel.precompute_latents.precompute`.
    """
    for path in list_clips(split, classes, per_class):
        yield load_video_frames(download_clip(path), num_frames=num_frames)
