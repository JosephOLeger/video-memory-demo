"""One MQAR cell, run to measure whether the inner loop diverges.

The video echo benchmark cannot answer this. In PCA space the divergence has not
been observed at all -- 0 NaN in 160+ runs -- so a fix that prevents it has nothing
to prevent there. MQAR at ``dim=32`` is where it was characterised (README §8.5):
abrupt, inside a single forward, on a minority of seeds, and **deterministic per
seed**, so a failed cell must be replaced with a different seed and never retried.

The setup is the one that produced those numbers, taken from the ``nan_diag`` probe
that localised the failure to step 1639: ``segment_size=8``, ``dim=32``, batch 32,
3000 steps, ``num_pairs=8``, and **``key_vocab = val_vocab = 64``**. That last one is
not cosmetic -- a first attempt at ``vocab=16`` (chance 1/16 instead of 1/64) is an
easier task with a different curvature, and produced 0 divergences in 25 cells. The
memory's hidden width is left at its default rather than pinned, again to match.

``--seed`` varies the model initialisation, which is what the seed-dependence is
about; the task stream itself is seeded by step index and is identical across arms.

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

SEG, DIM = 8, 32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--num-pairs", type=int, default=8, dest="num_pairs")
    ap.add_argument("--vocab", type=int, default=64,
                    help="key AND value vocabulary; 64 is what diverged")
    ap.add_argument("--normalize-hidden", type=float, default=None, dest="nh",
                    help="target L2 norm for h_0; unset = off")
    # Levers for the predicted threshold theta* ~ (1 - eta) / m. Chunk length m is
    # varied through memory_chunk_size with segment_size held fixed, so attention
    # sees the same window at every m and only the inner loop's staleness changes.
    ap.add_argument("--theta-scale", type=float, default=None, dest="theta_scale")
    ap.add_argument("--eta-init", type=float, default=None, dest="eta_init")
    ap.add_argument("--chunk", type=int, default=None,
                    help="memory_chunk_size = m; defaults to the segment size")
    ap.add_argument("--seg", type=int, default=SEG, dest="seg")
    # Cascade flags, mirroring echo.py so an arm reads the same in both sweeps.
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    ap.add_argument("--leak", default="none")
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--gate", default="surprise")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.chunk is not None and a.chunk > a.seg:
        raise SystemExit(f"chunk {a.chunk} cannot exceed segment size {a.seg}")

    def make_task(seed):
        return associative_recall(batch=32, num_pairs=a.num_pairs,
                                  segment_size=a.seg, key_vocab=a.vocab,
                                  val_vocab=a.vocab, seed=seed)

    cfg = dict(CONFIGS[a.config])
    if a.nh is not None:
        cfg["normalize_hidden"] = a.nh
    if a.theta_scale is not None:
        cfg["theta_scale"] = a.theta_scale
    if a.eta_init is not None:
        cfg["eta_init"] = a.eta_init
    if a.chunk is not None:
        cfg["memory_chunk_size"] = a.chunk
    if a.cascade:
        cfg.update(memory_use_cascade=True, cascade_reservoirs=a.reservoirs,
                   cascade_leak=a.leak, cascade_gate_signal=a.gate)
        if a.readout != "none":
            cfg["cascade_readout"] = a.readout
        if a.selective:
            cfg["cascade_selective"] = True

    torch.manual_seed(a.seed)
    t0 = time.time()
    res = train_recall(make_task, steps=a.steps, lr=1e-3, dim=DIM, depth=2,
                       segment_size=a.seg, num_heads=1, memory_num_layers=2,
                       eval_seeds=range(10_000_000, 10_000_020),
                       stop_on_nan=True, **cfg)

    loss, acc = float(res["train_loss"]), float(res["acc_mean"])
    out = {
        "label": a.label or a.config + ("" if a.nh is None else f"_nh{a.nh:g}"),
        "config": a.config, "seed": a.seed, "steps": a.steps,
        "num_pairs": a.num_pairs, "vocab": a.vocab, "normalize_hidden": a.nh,
        "theta_scale": cfg.get("theta_scale"), "eta_init": cfg.get("eta_init"),
        "seg": a.seg, "chunk": a.chunk if a.chunk is not None else a.seg,
        "cascade": a.cascade, "leak": a.leak if a.cascade else None,
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
