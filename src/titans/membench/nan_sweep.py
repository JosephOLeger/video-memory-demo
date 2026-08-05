"""One MQAR cell, run to measure whether the inner loop diverges.

The video echo benchmark cannot answer this. In PCA space the divergence has not
been observed at all -- 0 NaN in 160+ runs -- so a fix that prevents it has nothing
to prevent there. MQAR at ``dim=32`` is where it was characterised (README §8.5):
abrupt, inside a single forward, on a minority of seeds, and **deterministic per
seed**, so a failed cell must be replaced with a different seed and never retried.

The setup is the one that produced those numbers: ``segment_size=8``, ``dim=32``,
memory hidden 32, batch 32, 3000 steps, ``num_pairs=8``. ``--seed`` varies the model
initialisation, which is what the seed-dependence is about; the task stream itself is
seeded by step index and is identical across arms.

Outcome recorded per cell: ``diverged`` and, when it did, ``diverged_at`` -- the step
the loss first went non-finite. A NaN rate is only meaningful against a control run
at the same seeds, so arms share the seed grid.

    python -m titans.membench.nan_sweep --config sane_ns --seed 0 --out cell.json
    python -m titans.membench.nan_sweep --config sane_ns --seed 0 --normalize-hidden 1.0
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from ..worldmodel.atlas import CONFIGS
from .run import train_recall
from .tasks import associative_recall

SEG, DIM, MEM_HIDDEN = 8, 32, 32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--num-pairs", type=int, default=8, dest="num_pairs")
    ap.add_argument("--normalize-hidden", type=float, default=None, dest="nh",
                    help="target L2 norm for h_0; unset = off")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    kv = max(16, a.num_pairs)

    def make_task(seed):
        return associative_recall(batch=32, num_pairs=a.num_pairs,
                                  segment_size=SEG, key_vocab=kv, val_vocab=16,
                                  seed=seed)

    cfg = dict(CONFIGS[a.config])
    if a.nh is not None:
        cfg["normalize_hidden"] = a.nh

    torch.manual_seed(a.seed)
    t0 = time.time()
    res = train_recall(make_task, steps=a.steps, lr=1e-3, dim=DIM, depth=2,
                       segment_size=SEG, num_heads=1, memory_hidden_dim=MEM_HIDDEN,
                       eval_seeds=range(10_000_000, 10_000_020),
                       stop_on_nan=True, **cfg)

    loss, acc = float(res["train_loss"]), float(res["acc_mean"])
    out = {
        "label": a.label or a.config + ("" if a.nh is None else f"_nh{a.nh:g}"),
        "config": a.config, "seed": a.seed, "steps": a.steps,
        "num_pairs": a.num_pairs, "normalize_hidden": a.nh,
        "diverged": res["diverged_at"] is not None,
        "diverged_at": res["diverged_at"],
        "acc_mean": acc, "acc_std": float(res["acc_std"]),
        "chance": float(res["chance"]), "train_loss": loss,
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps(out), flush=True)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        import os
        os.replace(tmp, a.out)          # atomic: a kill mid-write leaves no half cell


if __name__ == "__main__":
    main()
