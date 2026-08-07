"""Build genuinely long contiguous streams by re-joining EgoLife's own segments.

Why this exists. The motivating claim is that keeping long-term memory in weights is
cheaper than keeping it in context. That is only *demonstrable* past the length where
the memory's constant state costs less than the KV cache it replaces --
``state_bytes / (2 * dim * 4 * depth)``, which at the reference config is **258
latents**. Every sequence in the project is 64 or 144, so the claim has never been
tested in the regime where it bites.

The corpus already contains what is needed. EgoLife's ``.mp4`` files are consecutive
90-120 s chunks of one continuous recording: within a person-day the gap between
consecutive start times sits tightly at the person's median, apart from a handful of
multi-hour breaks (night, charging). Re-joining them in time order gives contiguous
streams of 12 000-19 000 latents per person-day -- 50x past the crossover, at no
encoding cost.

What this does **not** fix: each ``.mp4`` is uniformly resampled to a fixed latent
count regardless of its duration (``load_video_frames``), so a 90 s and a 120 s
segment both yield ``T`` latents and the seconds-per-latent varies a little along the
stream. That is a property the existing corpus already has; joining segments does not
add it.

Splitting stays **person-disjoint** because runs never cross a person, so the existing
train/val split is preserved exactly.

    python -m titans.worldmodel.concat_stream --raw cache/egolife_train_8000.pt \\
        --pca cache/egolife_pca128_train.pt --length 512 --out cache/egolife_long512_train.pt
"""

from __future__ import annotations

import argparse
import os
import re
import statistics as st
from collections import defaultdict

import torch

from .train import load_latent_cache

NAME = re.compile(r"^([^/]+)/([^/]+)/\2_\1_(\d{8})\.mp4$")


def parse_segment(name: str):
    """``(person, day, start_seconds)`` from an EgoLife path, or ``None``.

    The stem is ``HHMMSSFF`` -- hours, minutes, seconds, then a frame count, which is
    why the last two digits must not be read as hundredths.
    """
    m = NAME.match(name)
    if not m:
        return None
    person, day, ts = m.groups()
    h, mn, s, fr = int(ts[:2]), int(ts[2:4]), int(ts[4:6]), int(ts[6:])
    return person, day, h * 3600 + mn * 60 + s + fr / 30.0


def contiguous_runs(segments, tol: float = 1.5):
    """Index runs whose consecutive start times stay within ``tol`` x the median gap.

    The threshold is taken **per person**: the recorders differ (90 s for one, 120 s
    for the others), so a single global threshold would either split a whole person
    into singletons or swallow their night breaks.
    """
    by_group = defaultdict(list)
    for i, name in enumerate(segments):
        p = parse_segment(name)
        if p is not None:
            by_group[(p[0], p[1])].append((p[2], i))

    med_by_person = {}
    for (person, _), v in by_group.items():
        v.sort()
        med_by_person.setdefault(person, []).extend(b[0] - a[0] for a, b in zip(v, v[1:]))
    med_by_person = {k: (st.median(g) if g else float("inf"))
                     for k, g in med_by_person.items()}

    runs = []
    for (person, day), v in sorted(by_group.items()):
        v.sort()
        limit = tol * med_by_person[person]
        cur = [v[0][1]]
        for (t0, _), (t1, i1) in zip(v, v[1:]):
            if t1 - t0 <= limit:
                cur.append(i1)
            else:
                runs.append(((person, day), cur))
                cur = [i1]
        runs.append(((person, day), cur))
    return runs


def windows_from_runs(runs, latents: torch.Tensor, length: int):
    """Non-overlapping ``length``-latent windows cut out of each contiguous run.

    A run of ``k`` segments is ``k * T`` latents laid end to end; the tail that does
    not fill a whole window is dropped rather than padded, so every row of the cache
    is a genuine uninterrupted stretch.
    """
    T = latents.shape[1]
    out, prov = [], []
    for key, idx in runs:
        if len(idx) * T < length:
            continue
        stream = latents[idx].reshape(-1, latents.shape[-1])       # [k*T, D]
        n = stream.shape[0] // length
        for j in range(n):
            out.append(stream[j * length:(j + 1) * length])
            prov.append({"person": key[0], "day": key[1], "window": j,
                         "run_segments": len(idx)})
    if not out:
        return torch.empty(0, length, latents.shape[-1]), []
    return torch.stack(out), prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="cache carrying the `segments` list")
    ap.add_argument("--pca", required=True, help="PCA cache, same row order as --raw")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = torch.load(a.raw, map_location="cpu", weights_only=False)
    segments = raw["segments"]
    z, meta = load_latent_cache(a.pca)
    if "pca" not in meta:
        raise SystemExit(f"{a.pca} is not a PCA cache.")
    if z.shape[0] != len(segments):
        raise SystemExit(
            f"row mismatch: {a.pca} has {z.shape[0]} rows, {a.raw} lists "
            f"{len(segments)} segments. They must be index-aligned -- the PCA cache is "
            "built by applying the basis row-by-row to the raw one."
        )
    del raw

    runs = contiguous_runs(segments, tol=a.tol)
    lens = sorted((len(idx) * z.shape[1] for _, idx in runs), reverse=True)
    print(f"{len(segments)} segments -> {len(runs)} passages contigus")
    print(f"longueur des passages (latents): max {lens[0]}  med {lens[len(lens)//2]}  "
          f"min {lens[-1]}")
    print(f"passages >= {a.length} latents : {sum(1 for L in lens if L >= a.length)}")

    win, prov = windows_from_runs(runs, z, a.length)
    if win.shape[0] == 0:
        raise SystemExit(f"no run reaches {a.length} latents; lower --length")
    kept = win.shape[0] * a.length
    print(f"-> {win.shape[0]} fenetres de {a.length} latents "
          f"({kept} latents gardes sur {z.numel() // z.shape[-1]}, "
          f"{100 * kept / (z.numel() // z.shape[-1]):.0f}%)")
    print(f"   personnes: {sorted({p['person'] for p in prov})}")

    blob = {"latents": win, "pca": meta["pca"], "k": meta["k"],
            "whiten": meta["whiten"], "explained": meta["explained"],
            "latent_dim_original": meta.get("latent_dim_original", 0),
            "provenance": prov, "source_pca": a.pca, "source_raw": a.raw,
            "window_length": a.length, "gap_tol": a.tol}
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, a.out)
    print("ecrit:", a.out, tuple(win.shape))


if __name__ == "__main__":
    main()
