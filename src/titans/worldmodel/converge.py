"""Has anything actually converged at 2500 steps?

``echo.py`` has defaulted to ``--steps 2500`` since the commit that introduced the
benchmark, and the number was never justified: on the reference cache that is 5999
sequences at batch 8, i.e. **3.3 epochs**, and nothing logs the loss during training.

Why it matters more than a usual convergence question. The stability *margin* is
``theta_t * lambda_hidden``, and ``theta_t`` is a **learned head**. On MQAR it was
seen drifting from 0.68 to 0.57 over 3000 outer steps. If it is still moving at 2500
here, every margin in the cascade-vs-scalar comparison is a snapshot of a trajectory
rather than an equilibrium -- and it is the *ordering* of those margins that carries
the conclusion.

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

    paths = sorted(globmod.glob(a.frozen), key=lambda p: (arm_of(p), step_of(p)))
    rows = []
    for p in paths:
        try:
            model, info = load_frozen(p, a.device)
            rec = {**evaluate(model, va, T, a.horizon, a.n_eval),
                   **measure(model, va[:a.nseq], info)}
        except Exception as e:
            rec = {"error": repr(e)[:200]}
        rec.update(arm=arm_of(p), step=step_of(p), file=os.path.basename(p))
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in
                          ("arm", "step", "memory_gain", "margin", "theta_t_max",
                           "lambda_hidden_max", "pr", "cos_ret", "second", "error")}),
              flush=True)

    print()
    print("%-16s%8s%13s%9s%13s%13s%8s%9s" %
          ("bras", "pas", "memory_gain", "marge", "theta_t max", "lambda_h max",
           "pr", "l1 (2e)"))
    print("-" * 90)
    last_arm = None
    for r in rows:
        if "error" in r:
            print("%-16s%8s  ERREUR %s" % (r["arm"], r["step"], r["error"][:50]))
            continue
        if last_arm is not None and r["arm"] != last_arm:
            print()
        last_arm = r["arm"]
        print("%-16s%8d%13.4f%9.2f%13.5f%13.1f%8.1f%9.4f" %
              (r["arm"], r["step"], r["memory_gain"], r["margin"], r["theta_t_max"],
               r["lambda_hidden_max"], r["pr"], r["second"]))
    print("\nSi memory_gain et marge sont encore en mouvement au dernier pas,")
    print("2500 n'est pas un equilibre et l'ordre des marges n'est pas etabli.")

    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
