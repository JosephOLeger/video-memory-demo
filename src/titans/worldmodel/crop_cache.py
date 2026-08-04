"""Cut a latent cache into shorter sequences, for the echo benchmark's VRAM budget.

The echo benchmark trains on ``[V;V]`` (length ``2T``) and needs **full BPTT** over
it: a read at position ``T+j`` in copy 2 and the write at position ``j`` that it must
credit are exactly ``T`` apart, and :func:`~titans.worldmodel.training.tbptt_step`
detaches the carry at every window boundary. Any window ``<= T`` therefore severs
*every* read/write pair, and the benchmark measures nothing -- see README §6.1.

So the only lever against VRAM is the sequence length itself, not the window.
Cropping ``T=144`` to ``T=64`` takes ``cas_tag`` from 20.98 GiB (which OOMs a 22.18
GiB card) to 9.34 GiB, and it keeps the corpus size: two crops per sequence use 128
of the 144 available latents.

    python -m titans.worldmodel.crop_cache \
        --src cache/egolife_full_pca128_train.pt \
        --dst cache/egolife_full_pca128_T64_train.pt --T 64

Cropping also makes a corpus comparable to one recorded at a different sequence
length, which is what the metric needs: ``memory_gain`` is measured in prediction
*steps*, so ``T`` and ``K`` must match across corpora for two numbers to mean the
same thing.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def crop_latents(z: torch.Tensor, T: int) -> torch.Tensor:
    """``[B, T0, d] -> [B * (T0 // T), T, d]``, dropping the ragged tail.

    Consecutive crops stay adjacent in the batch dimension, which keeps the
    provenance expansion in :func:`crop_cache_file` a simple repeat.
    """
    B, T0, d = z.shape
    n = T0 // T
    if n < 1:
        raise ValueError(f"T={T} exceeds the cache's sequence length {T0}")
    return z[:, : n * T].reshape(B * n, T, d).contiguous()


def crop_cache_file(src: str | Path, dst: str | Path, T: int) -> dict:
    """Crop a cache file, preserving every metadata key the harness reads.

    The ``pca`` key in particular has to survive: it is what makes the harness
    treat the cache as pre-reduced and skip standardisation.
    """
    obj = torch.load(src, map_location="cpu", weights_only=False)
    z = obj["latents"]
    B, T0, _ = z.shape
    obj["latents"] = crop_latents(z, T)
    n = T0 // T
    obj["crop"] = {"from_T": T0, "to_T": T, "crops_per_seq": n, "source": str(src)}
    if "segments" in obj:                    # keep provenance aligned with rows
        obj["segments"] = [s for s in obj["segments"] for _ in range(n)]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, dst)                     # atomic: a kill mid-write leaves no
    return {"path": str(dst),                # half-written cache behind
            "shape": tuple(obj["latents"].shape),
            "from_shape": (B, T0, z.shape[-1]),
            "crops_per_seq": n,
            "gb": obj["latents"].numel() * 4 / 1e9}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--T", type=int, required=True)
    a = ap.parse_args()

    r = crop_cache_file(a.src, a.dst, a.T)
    print(f"{r['path']}: {r['from_shape']} -> {r['shape']}  "
          f"({r['gb']:.2f} GB, {r['crops_per_seq']} crops/seq)")


if __name__ == "__main__":
    main()
