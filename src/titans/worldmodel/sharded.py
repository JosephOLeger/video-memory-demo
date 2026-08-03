"""Read a shard directory, and fit/apply a PCA over it without materialising it.

:mod:`titans.worldmodel.encode_egolife` writes the corpus as ~320 shards rather
than one file, because a single write cannot be resumed and 4.6 M latents at 1024-d
float32 is 18.9 GB. That size is also why the PCA cannot simply call
:func:`~titans.worldmodel.bottleneck.fit_pca`: it standardises the whole matrix and
takes a full SVD, so the train split alone would need ~52 GB of float64 working set
(``Xs`` plus ``U``).

The fix is to never form the data matrix. PCA only needs second-order statistics:

    pass 1   accumulate per-channel sum and sum-of-squares  ->  mean, std
    pass 2   accumulate the Gram matrix  G = sum_shards Xs^T Xs   (1024 x 1024)
             eigh(G) = V S^2 V^T, so V's top-k columns *are* the components and
             the eigenvalues are the squared singular values

Both passes are O(d^2) in memory regardless of how many latents exist, and the
result is **exact**, not an approximation -- :func:`fit_pca_sharded` is tested
against :func:`fit_pca` on data small enough for both.

Eigenvectors are defined up to sign, so components may differ from ``fit_pca`` by a
per-component sign flip. That is irrelevant to PCA (the reduced coordinates flip
with them) and the tests compare on sign-invariant quantities.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterator

import torch


def shard_paths(shard_dir: str | Path, split: str) -> list[str]:
    """Sorted shard files for a split; sorting keeps sequence order reproducible."""
    return sorted(glob.glob(str(Path(shard_dir) / f"{split}_*.pt")))


def iter_shards(shard_dir: str | Path, split: str) -> Iterator[tuple[torch.Tensor, dict]]:
    """Yield ``(latents, metadata)`` per shard, one shard resident at a time."""
    for p in shard_paths(shard_dir, split):
        obj = torch.load(p, map_location="cpu", weights_only=False)
        yield obj["latents"], {k: v for k, v in obj.items() if k != "latents"}


def shard_summary(shard_dir: str | Path, split: str) -> dict:
    """Sequence count, shape and provenance, without holding the data."""
    n_seq = 0; T = d = None; persons = set(); shards = 0
    for z, meta in iter_shards(shard_dir, split):
        n_seq += z.shape[0]; T, d = z.shape[1], z.shape[2]; shards += 1
        for s in meta.get("segments", []):
            persons.add(s.split("/")[0])
    return {"shards": shards, "sequences": n_seq, "T": T, "latent_dim": d,
            "latents": n_seq * (T or 0), "persons": sorted(persons)}


def fit_pca_sharded(shard_dir: str | Path, split: str, k: int, *,
                    whiten: bool = True, eps: float = 1e-6) -> dict:
    """Exact PCA over a shard directory, in O(d^2) memory.

    Same output contract as :func:`~titans.worldmodel.bottleneck.fit_pca` -- a dict
    of ``mean``/``std``/``components``/``sv``/``whiten``/``k``/``explained`` -- so
    ``apply_pca`` and ``inverse_pca`` consume it unchanged.
    """
    # pass 1: per-channel statistics
    n = 0; s1 = None; s2 = None
    for z, _ in iter_shards(shard_dir, split):
        X = z.reshape(-1, z.shape[-1]).double()
        s1 = X.sum(0) if s1 is None else s1 + X.sum(0)
        s2 = (X * X).sum(0) if s2 is None else s2 + (X * X).sum(0)
        n += X.shape[0]
    if n == 0:
        raise ValueError(f"no shards found for split {split!r} in {shard_dir}")
    mean = s1 / n
    # Sample variance (Bessel), matching torch.Tensor.std's default and therefore
    # fit_pca. Using the population form instead leaves a sqrt(n/(n-1)) offset --
    # invisible at 4.6 M rows, but it would silently make the two paths disagree.
    var = ((s2 - n * mean ** 2) / max(n - 1, 1)).clamp_min(0)
    std = var.sqrt() + eps

    # pass 2: Gram matrix of the standardised data
    D = mean.numel()
    if not 1 <= k <= D:
        raise ValueError(f"k must be in [1, D={D}]")
    G = torch.zeros(D, D, dtype=torch.float64)
    for z, _ in iter_shards(shard_dir, split):
        Xs = (z.reshape(-1, D).double() - mean) / std
        G += Xs.T @ Xs

    evals, evecs = torch.linalg.eigh(G)          # ascending
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    comps = evecs[:, order[:k]].T                # [k, D], rows are components
    sv = (evals[:k] / (n - 1)).sqrt() + eps
    explained = float(evals[:k].sum() / evals.sum())
    return {"mean": mean.float(), "std": std.float(), "components": comps.float(),
            "sv": sv.float(), "whiten": whiten, "k": k, "explained": explained,
            "n_rows": n}


def write_reduced_cache(shard_dir: str | Path, split: str, pca: dict,
                        out_path: str | Path) -> dict:
    """Project every shard through ``pca`` and write one reduced cache file.

    The reduced corpus is small enough to live as a single tensor (4.6 M latents at
    k=128 is 2.4 GB, against 18.9 GB at 1024-d), which is what lets it sit on a GPU
    for training. Metadata carries the fitted basis, so predictions stay invertible
    to the encoder space, and the ``pca`` key is what makes the harness auto-detect
    a pre-reduced cache.
    """
    from .bottleneck import apply_pca

    chunks, segments = [], []
    for z, meta in iter_shards(shard_dir, split):
        chunks.append(apply_pca(z, pca))
        segments.extend(meta.get("segments", []))
    red = torch.cat(chunks, dim=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save({"latents": red, "pca": pca, "k": pca["k"], "whiten": pca["whiten"],
                "explained": pca["explained"], "segments": segments,
                "latent_dim_original": int(pca["mean"].numel()),
                "source_shards": str(shard_dir), "split": split}, tmp)
    tmp.replace(out_path)
    return {"path": str(out_path), "shape": tuple(red.shape),
            "gb": red.numel() * 4 / 1e9, "explained": pca["explained"]}


def main() -> None:
    import argparse, time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", default="cache/egolife_full")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--out-prefix", default="cache/egolife_full_pca")
    ap.add_argument("--no-whiten", action="store_true")
    ap.add_argument("--fit-split", default="train",
                    help="the basis is fitted on this split only -- no val leakage")
    a = ap.parse_args()

    for split in ("train", "val"):
        print(f"  {split}: {shard_summary(a.shard_dir, split)}", flush=True)

    t0 = time.time()
    pca = fit_pca_sharded(a.shard_dir, a.fit_split, a.k, whiten=not a.no_whiten)
    print(f"\nfit PCA k={a.k} on {a.fit_split} ({pca['n_rows']} rows): "
          f"{time.time()-t0:.0f}s, explains {100*pca['explained']:.2f}%", flush=True)

    for split in ("train", "val"):
        t = time.time()
        r = write_reduced_cache(a.shard_dir, split, pca,
                                f"{a.out_prefix}{a.k}_{split}.pt")
        print(f"  wrote {r['path']} {r['shape']} = {r['gb']:.2f} GB "
              f"({time.time()-t:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
