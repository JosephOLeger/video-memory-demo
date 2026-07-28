"""Write PCA-reduced latent caches, so the reduction is part of the *dataset*.

Motivation. With a learned ``in_proj``/``out_proj`` (1024 -> dim -> 1024) most of a
small model's parameters sit in mapping in and out of the frozen encoder's space
rather than in the long-term memory we actually study, and the projections are free
to co-adapt with the memory -- which muddies "what did the memory learn?". Fixing a
PCA basis instead makes the reduction a property of the data: the model then runs
directly on ``k``-dim latents with no I/O projection at all
(``WorldModel(io_projection=False)``).

The basis is fitted on **train only** and applied to both splits, matching the
standardisation convention, and it is stored alongside the reduced latents so any
prediction can be mapped back to the 1024-D encoder space with ``inverse_pca``.

    python -m titans.worldmodel.make_pca_cache \
        --train cache/egolife_train_8000.pt --val cache/egolife_val_8000.pt \
        --k 256 --out-prefix cache/egolife_pca256

writes ``<prefix>_train.pt`` / ``<prefix>_val.pt``, each a dict with ``latents``
(the reduced tensor) plus the fitted ``pca`` basis and provenance.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .bottleneck import apply_pca, fit_pca
from .train import load_latents


def build(train_path, val_path, k: int, out_prefix, *, whiten: bool = True) -> dict:
    t0 = time.time()
    tr = load_latents(train_path)
    va = load_latents(val_path)
    print(f"loaded train {tuple(tr.shape)}  val {tuple(va.shape)}  "
          f"({time.time() - t0:.1f}s)", flush=True)

    t1 = time.time()
    pca = fit_pca(tr, k, whiten=whiten)
    t_fit = time.time() - t1
    print(f"fit PCA k={k} whiten={whiten}: {t_fit:.1f}s  "
          f"explained={100 * pca['explained']:.2f}%", flush=True)

    t2 = time.time()
    tr_r, va_r = apply_pca(tr, pca), apply_pca(va, pca)
    t_apply = time.time() - t2
    print(f"applied: train {tuple(tr_r.shape)}  val {tuple(va_r.shape)}  "
          f"({t_apply:.1f}s)", flush=True)

    meta = dict(pca=pca, source_train=str(train_path), source_val=str(val_path),
                k=k, whiten=whiten, explained=pca["explained"],
                latent_dim_original=int(tr.shape[-1]))
    out_prefix = Path(out_prefix)
    tp = out_prefix.with_name(out_prefix.name + "_train.pt")
    vp = out_prefix.with_name(out_prefix.name + "_val.pt")
    torch.save({"latents": tr_r, **meta}, tp)
    torch.save({"latents": va_r, **meta}, vp)
    total = time.time() - t0
    print(f"wrote {tp} ({tp.stat().st_size / 1e6:.0f} MB) and "
          f"{vp} ({vp.stat().st_size / 1e6:.0f} MB)", flush=True)
    print(f"TOTAL {total:.1f}s  (fit {t_fit:.1f}s, apply {t_apply:.1f}s)", flush=True)
    return {"k": k, "explained": pca["explained"], "fit_s": t_fit,
            "apply_s": t_apply, "total_s": total, "train": str(tp), "val": str(vp)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--k", type=int, action="append", required=True,
                    help="kept components; repeat the flag for several k")
    ap.add_argument("--out-prefix", required=True,
                    help="'{prefix}' -> '{prefix}{k}_train.pt' / '_val.pt'")
    ap.add_argument("--no-whiten", action="store_true")
    a = ap.parse_args()
    for k in a.k:
        print(f"=== k={k} ===", flush=True)
        build(a.train, a.val, k, f"{a.out_prefix}{k}", whiten=not a.no_whiten)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
