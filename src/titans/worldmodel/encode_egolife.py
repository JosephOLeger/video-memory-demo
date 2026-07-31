"""Encode the *whole* of EgoLife at native frame rate, sharded and resumable.

Why this exists rather than :func:`precompute_egolife`. That function samples a
subset of segments, subsamples each to 128 frames, holds every latent in RAM and
writes once at the end. For the full corpus that is three separate failures:

* **We were using 4% of the data.** EgoLife has 32 001 segments (267 h); the sampler
  took 5 999 (19 %), and within each one ``num_frames=128`` kept 128 of 600 frames
  (21 %). Combined: 0.38 M of a possible 9.2 M latents.
* **RAM.** 9.2 M latents at 1024-d float32 is 38 GB, and ``torch.stack`` briefly
  doubles that.
* **No resumability.** A single write at the end means a 58 h job that dies at hour
  50 loses everything -- and on a shared box it *will* be interrupted.

This module streams instead: for each shard of segments it downloads, encodes,
writes the shard, and deletes the video. Shards already on disk are skipped, so
re-running the same command resumes. Videos go to a scratch dir rather than the
shared HF cache, which would otherwise accumulate ~0.5 TB we do not need.

Native frame rate matters for quality, not just quantity: subsampling 600 frames to
128 spaces them 4.7x further apart than the encoder was trained on, and V-JEPA-2 is
motion-aware -- the same objection this repo already documents against pre-encoder
frame averaging. At 600 frames a 30 s segment yields 288 latents (104 ms each)
instead of 64 (469 ms), which truncated BPTT is what makes trainable.

    python -m titans.worldmodel.encode_egolife --split train --shard-size 250 \
        --out-dir cache/egolife_full --gpu-slice 0/4
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from .egolife import DATASET_ID, TRAIN_PERSONS, VAL_PERSONS, _all_mp4
from .data import load_video_frames
from .precompute_latents import CLIP_LEN, load_processor, video_to_latents_sliding
from .encoder import VJEPA2Encoder

NATIVE_FRAMES = 600          # ~30 s at 20 fps -- EgoLife's own segment length


def segments_for(split: str) -> list[str]:
    """All .mp4 paths for a split, person-disjoint (no identity leakage)."""
    persons = TRAIN_PERSONS if split == "train" else VAL_PERSONS
    return [f for f in _all_mp4() if f.split("/")[0] in persons]


def encode_shard(paths, encoder, processor, scratch: Path, num_frames: int,
                 clip_len: int, stride: int):
    """Download, encode and immediately discard each video in the shard."""
    out, kept = [], []
    for p in paths:
        from huggingface_hub import hf_hub_download
        local = None
        try:
            local = hf_hub_download(DATASET_ID, p, repo_type="dataset",
                                    local_dir=str(scratch))
            frames = load_video_frames(local, num_frames=num_frames)
            out.append(video_to_latents_sliding(frames, encoder, processor,
                                                clip_len=clip_len,
                                                stride_frames=stride).cpu())
            kept.append(p)
        except Exception as exc:                       # a corrupt segment must not
            print(f"    skip {p}: {type(exc).__name__} {exc}", flush=True)
        finally:                                       # kill a 58 h run
            if local and os.path.exists(local):
                os.remove(local)
    return (torch.stack(out) if out else None), kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--out-dir", default="cache/egolife_full")
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--num-frames", type=int, default=NATIVE_FRAMES)
    ap.add_argument("--clip-len", type=int, default=CLIP_LEN)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--gpu-slice", default="0/1",
                    help="'i/n': this process takes every n-th shard starting at i, "
                         "so n processes on n GPUs partition the work with no lock")
    ap.add_argument("--limit-shards", type=int, default=0)
    a = ap.parse_args()

    i, n = (int(x) for x in a.gpu_slice.split("/"))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / f".scratch_{a.split}_{i}"
    scratch.mkdir(parents=True, exist_ok=True)

    segs = segments_for(a.split)
    shards = [segs[s:s + a.shard_size] for s in range(0, len(segs), a.shard_size)]
    mine = [(k, sh) for k, sh in enumerate(shards) if k % n == i]
    if a.limit_shards:
        mine = mine[: a.limit_shards]
    todo = [(k, sh) for k, sh in mine
            if not (out_dir / f"{a.split}_{k:04d}.pt").exists()]
    print(f"[{a.split} {i}/{n}] {len(segs)} segments, {len(shards)} shards, "
          f"{len(mine)} for me, {len(todo)} still to do", flush=True)
    if not todo:
        print("nothing to do", flush=True); return

    encoder = VJEPA2Encoder()
    processor = load_processor(encoder.model_name)
    t0 = time.time()
    for done, (k, sh) in enumerate(todo, 1):
        ts = time.time()
        lat, kept = encode_shard(sh, encoder, processor, scratch,
                                 a.num_frames, a.clip_len, a.stride)
        if lat is None:
            print(f"  shard {k:04d}: all segments failed, skipped", flush=True)
            continue
        tmp = out_dir / f"{a.split}_{k:04d}.pt.tmp"
        torch.save({"latents": lat, "segments": kept, "model_name": encoder.model_name,
                    "clip_len": a.clip_len, "sliding_stride": a.stride,
                    "num_frames": a.num_frames}, tmp)
        os.replace(tmp, out_dir / f"{a.split}_{k:04d}.pt")   # atomic: a kill mid-write
        el = time.time() - ts                                # cannot leave a bad shard
        eta = (time.time() - t0) / done * (len(todo) - done) / 3600
        print(f"  shard {k:04d}: {tuple(lat.shape)} in {el/60:.1f} min "
              f"({done}/{len(todo)}, ETA {eta:.1f} h)", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
