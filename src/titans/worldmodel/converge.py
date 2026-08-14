"""Has anything actually converged at 2500 steps?

``echo.py`` has defaulted to ``--steps 2500`` since the commit that introduced the
benchmark, and the number was never justified: on the reference cache that is 5999
sequences at batch 8, i.e. **3.3 epochs**, and nothing logs the loss during training.

Why it matters more than a usual convergence question. The stability *margin* is
``theta_t * lambda_hidden``, and ``theta_t`` is a **learned head** -- it was seen
drifting by 16% over 3000 outer steps on the synthetic benchmark. If it is still
moving at 2500 here, every margin in the cascade-vs-scalar comparison is a snapshot
of a trajectory rather than an equilibrium -- and it is the *ordering* of those
margins that carries the conclusion.

So this scores both axes along one training trajectory: ``memory_gain`` (the [V;V]
echo) and the margin, at every snapshot written by ``echo.py --frozen-every``. One
run per arm, evaluated offline; re-training once per step count would cost the sum of
the points instead of the largest, and would answer a different question (a fresh
trajectory per point rather than one trajectory observed).

    python -m titans.worldmodel.echo ... --steps 10000 --frozen-every 1250 \\
        --frozen-out frozen_cv/nocas.pt
    python -m titans.worldmodel.converge --frozen 'frozen_cv/*_at*.pt' \\
        --val cache/egolife_pca128_val.pt
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re

import torch

from .echo import evaluate
from .frozen import load_frozen
from .margin import measure
from .train import load_latent_cache


def step_of(path: str) -> int:
    """The training step a snapshot was written at, from its ``_at<N>.pt`` suffix."""
    m = re.search(r"_at(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def arm_of(path: str) -> str:
    return re.sub(r"_at\d+\.pt$", "", os.path.basename(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", required=True, help="glob over numbered snapshots")
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", default=None,
                    help="the training cache; scores the same echo metrics on it so "
                         "the generalisation gap is measured along the trajectory. "
                         "Without it a rising val score cannot be told apart from "
                         "memorisation that happens not to have hurt val yet.")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=96, dest="n_eval")
    ap.add_argument("--nseq", type=int, default=8, help="sequences for the margin probe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    va, meta = load_latent_cache(a.val)
    if "pca" not in meta:
        raise SystemExit(f"{a.val} is not a PCA cache.")
    va = va.to(a.device)
    T = va.shape[1]
    # The overfitting question is not academic here: 2500 steps at batch 8 on the
    # 5999-sequence cache is 3.3 epochs, but 40000 steps is 53, and the model holds
    # 0.77 latents per parameter -- 50-100x outside the band ATLAS trains in. The
    # ratio has already mis-predicted once on this project (README 9), so the gap is
    # measured rather than argued.
    tr = None
    if a.train:
        tr, tmeta = load_latent_cache(a.train)
        if "pca" not in tmeta:
            raise SystemExit(f"{a.train} is not a PCA cache.")
        if tr.shape[1] != T:
            raise SystemExit(
                f"train T={tr.shape[1]} != val T={T}; the echo score is measured in "
                "prediction steps, so the two splits must share T to be subtracted.")
        tr = tr.to(a.device)

    paths = sorted(globmod.glob(a.frozen), key=lambda p: (arm_of(p), step_of(p)))
    rows = []
    for p in paths:
        try:
            model, info = load_frozen(p, a.device)
            rec = {**evaluate(model, va, T, a.horizon, a.n_eval),
                   **measure(model, va[:a.nseq], info)}
            if tr is not None:
                t = evaluate(model, tr, T, a.horizon, a.n_eval)
                rec.update(train_second=t["second"], train_mem_effect=t["mem_effect"],
                           train_memory_gain=t["memory_gain"],
                           # >0 means val is worse than train, the usual sign of
                           # overfitting; on mem_effect it means the *mechanism*
                           # transfers less well, which is the one that matters.
                           gap_l1=rec["second"] - t["second"],
                           gap_mem_effect=t["mem_effect"] - rec["mem_effect"])
        except Exception as e:
            rec = {"error": repr(e)[:200]}
        rec.update(arm=arm_of(p), step=step_of(p), file=os.path.basename(p))
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in
                          ("arm", "step", "memory_gain", "mem_effect", "margin",
                           "theta_t_max", "lambda_hidden_max", "pr", "cos_ret",
                           "second", "gap_l1", "gap_mem_effect", "error")}),
              flush=True)

    print()
    print("%-16s%8s%12s%9s%8s%9s%9s%11s" %
          ("bras", "pas", "mem_effect", "marge", "pr", "l1 (2e)", "ecart L1",
           "ecart m_eff"))
    print("-" * 92)
    last_arm = None
    for r in rows:
        if "error" in r:
            print("%-16s%8s  ERREUR %s" % (r["arm"], r["step"], r["error"][:50]))
            continue
        if last_arm is not None and r["arm"] != last_arm:
            print()
        last_arm = r["arm"]
        g1 = r.get("gap_l1")
        g2 = r.get("gap_mem_effect")
        print("%-16s%8d%12.4f%9.2f%8.1f%9.4f%9s%11s" %
              (r["arm"], r["step"], r["mem_effect"], r["margin"], r["pr"], r["second"],
               "-" if g1 is None else "%+.4f" % g1,
               "-" if g2 is None else "%+.4f" % g2))
    print("\nDeux criteres d'arret, et une colonne qui n'en est pas un :")
    print("  1. mem_effect (val) plafonne ou repart a la baisse -> arret classique")
    print("  2. l'ecart train/val s'ouvre -> surapprentissage")
    print("  la marge, elle, tombe de facon monotone avec les pas SANS que la survie")
    print("  en flux long ne bouge (test en aveugle 2026-08-12 : 7 predictions sur 9")
    print("  fausses, dont deux bras a marge < 8 qui ont tenu 64 000 tokens). C'est un")
    print("  diagnostic intra-famille, pas un critere d'arret. Pour la stabilite, lire")
    print("  d log||S||/dt sur une sonde long_run : ~0 chez les survivants, +0.013 a")
    print("  +0.065 chez les divergents, deux ordres de grandeur et aucun recouvrement.")

    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
