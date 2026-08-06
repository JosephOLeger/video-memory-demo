"""Measure each arm's peak VRAM at the exact shape a sweep will run.

Sizing a pool's memory guard by extrapolating from the lightest arm has cost this
project three sweeps (74 jobs on 2026-07-29, 22 on 07-31, 2 on 08-03). The arms are
not close: at ``T=64`` / doubled 128 / ``K=24`` / batch 8 / full BPTT,

    nocas        6.36 GiB
    nocas_eta10  6.36 GiB
    cas_tag      9.34 GiB      <- +47%, from the gated readout and selective heads

so a guard set from ``nocas`` puts two ``cas_tag`` jobs on one card and OOMs.
``--cascade`` costs almost nothing in the *reservoirs* themselves (~1.01x whatever
``K`` is); the cost is ``--readout gated`` and ``--selective``, about +24% each.

Run it once per (cache, horizon, batch, window) before writing the queue:

    python -m titans.worldmodel.measure_vram \
        --train cache/egolife_full_pca128_T64_train.pt \
        --val cache/egolife_full_pca128_T64_val.pt --horizon 24 --batch 8

Then set the guard above the heaviest number *plus* the allocator's reserve (which
runs ~15-25% over ``max_memory_allocated``), and see ``scripts/pool4.sh`` for why
the guard also has to be read under a per-card lock.
"""

from __future__ import annotations

import shlex

import torch

from .echo import build_cfg, build_parser, double, prepare_data
from .model import WorldModel
from .training import tbptt_step

#: The reference arms, as their flags appear in a jobs file.
ARMS = {
    "nocas": "",
    "nocas_eta10": "--eta-init 0.9",
    "cas_tag": "--cascade --reservoirs 6 --leak none --readout gated "
               "--selective --gate surprise",
}


def peak_gib(train: str, val: str, *, horizon: int, batch: int, seg: int = 8,
             window: int | None = None, arms: dict[str, str] | None = None,
             steps: int = 3, device: str = "cuda") -> dict[str, float]:
    """Peak allocated GiB per arm, over a few real optimisation steps.

    Three steps is enough: the allocator has reached steady state once a backward
    and an ``Adam.step`` have both run once.
    """
    arms = ARMS if arms is None else arms
    tr, _, space = prepare_data(train, val)
    tr = tr[: max(batch, 1) * 4].to(device)
    out = {}
    for name, extra in arms.items():
        a = build_parser().parse_args(shlex.split(
            f"--label {name} --train {train} --val {val} --horizon {horizon} "
            f"--seg {seg} --batch {batch} " + extra))
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = WorldModel(latent_dim=tr.shape[-1], dim=space["k"], depth=2,
                           segment_size=seg,
                           **build_cfg(a)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            tbptt_step(model, double(tr[:batch]), horizon=horizon, window=window)
            opt.step()
        out[name] = torch.cuda.max_memory_allocated() / 2 ** 30
        del model, opt
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--bptt-window", type=int, default=None, dest="window",
                    help="leave unset for full BPTT, which is what echo.py needs")
    a = ap.parse_args()

    peaks = peak_gib(a.train, a.val, horizon=a.horizon, batch=a.batch,
                     seg=a.seg, window=a.window)
    for name, gib in peaks.items():
        print(f"{name:14s} peak {gib:6.2f} GiB")
    heaviest = max(peaks.values())
    print(f"\nheaviest {heaviest:.2f} GiB -> suggested pool guard "
          f"{int(heaviest * 1024 * 1.35 // 500 + 1) * 500} MiB")


if __name__ == "__main__":
    main()
