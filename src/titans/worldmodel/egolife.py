"""EgoLife loader: long egocentric videos (lmms-lab/EgoLife).

EgoLife is continuous first-person life-logging: 6 people x 7 days, cut into
~30s / 600-frame / 20fps segments (``<person>/<day>/<...>.mp4``). That is 3x
longer and far more temporally continuous than Kinetics-mini clips -- better for a
world model, and a bigger, more diverse pool against overfitting.

Split is by *person* (train vs val people are disjoint), so there is no identity
leakage. Within a person, segments are sampled evenly across the whole week so the
training set spans diverse scenes rather than near-duplicate consecutive segments.

Reuses :func:`titans.worldmodel.data.load_video_frames` for decoding.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch

from .data import load_video_frames

DATASET_ID = "lmms-lab/EgoLife"
TRAIN_PERSONS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA"]
VAL_PERSONS = ["A5_KATRINA", "A6_SHURE"]


def _all_mp4() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".mp4"))


def list_egolife_clips(persons: list[str], per_person: int) -> list[str]:
    """Evenly sample ``per_person`` segments from each person's whole week.

    Even spacing over the chronologically-sorted segments spreads the sample
    across days/times, maximising scene diversity (vs consecutive near-duplicates).
    """
    all_mp4 = _all_mp4()
    kept: list[str] = []
    for person in persons:
        segs = [f for f in all_mp4 if f.startswith(f"{person}/")]
        if not segs:
            continue
        if per_person >= len(segs):
            kept.extend(segs)
        else:
            step = len(segs) / per_person
            kept.extend(segs[int(i * step)] for i in range(per_person))
    return kept


def download_egolife_clip(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET_ID, path, repo_type="dataset")


def egolife_frames(
    persons: list[str],
    per_person: int,
    num_frames: int = 128,
) -> Iterator[torch.Tensor]:
    """Yield decoded frame tensors ``[num_frames, H, W, C]`` for sampled segments.

    ``num_frames`` uniformly subsamples each ~600-frame/30s segment; a smaller
    value = larger temporal stride = more diverse, less redundant latents.
    """
    for path in list_egolife_clips(persons, per_person):
        yield load_video_frames(download_egolife_clip(path), num_frames=num_frames)
