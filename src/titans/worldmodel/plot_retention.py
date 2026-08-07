"""Draw the retention curve: how far apart can the echo's two halves be?

Reads the JSON written by :mod:`titans.worldmodel.retention` and produces one figure
plus a CSV, so the result is readable without re-running anything.

The x axis is the number of latents inserted between the two presentations of ``V``
-- the distance the memory has to carry the content across. The y axis is
``retention_gain = l1(cross) - l1(repeat)``: positive means ``V`` is still held.
The memory-off control is drawn as the floor; where the curve meets it, the memory
has stopped carrying anything.

Seeds are pooled per arm and shown as mean with a standard-error band, because the
per-seed spread on this benchmark (~0.03 on a gain of ~0.25) is large enough that a
single seed would suggest orderings that do not survive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
from collections import defaultdict


def _agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return st.mean(vals), sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0, len(vals)


def collect(rows):
    """``{arm: {gap: (mean, sem, n)}}`` for the gain, plus the memory-off floor."""
    by = defaultdict(lambda: defaultdict(list))
    off = defaultdict(list)
    meta = {}
    for r in rows:
        if "curve" not in r:
            continue
        meta.setdefault(r["arm"], r)
        for c in r["curve"]:
            by[r["arm"]][c["gap_latents"]].append(c["retention_gain"])
            off[c["gap_latents"]].append(c["gain_memoff"])
    curves = {a: {g: _agg(v) for g, v in gs.items()} for a, gs in by.items()}
    floor = {g: _agg(v) for g, v in off.items()}
    return curves, floor, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="Rétention épisodique : [V ; distracteurs ; V]")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(a.json))
    curves, floor, meta = collect(rows)
    os.makedirs(a.out_dir, exist_ok=True)

    gaps = sorted({g for c in curves.values() for g in c})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # cascade warm, scalar cool, so the two families read as families
    cas = sorted([a_ for a_ in curves if meta[a_].get("cascade")])
    sca = sorted([a_ for a_ in curves if not meta[a_].get("cascade")])
    cmaps = {**{a_: plt.cm.autumn(i / max(len(cas) - 1, 1) * 0.8) for i, a_ in enumerate(cas)},
             **{a_: plt.cm.winter(i / max(len(sca) - 1, 1) * 0.8) for i, a_ in enumerate(sca)}}

    for arm in cas + sca:
        gs = sorted(curves[arm])
        m = [curves[arm][g][0] for g in gs]
        e = [curves[arm][g][1] for g in gs]
        ax.plot(gs, m, marker="o", ms=4, lw=1.8, color=cmaps[arm], label=arm)
        ax.fill_between(gs, [x - y for x, y in zip(m, e)],
                        [x + y for x, y in zip(m, e)], color=cmaps[arm], alpha=0.15,
                        linewidth=0)

    fl = [floor[g][0] for g in gaps if g in floor]
    ax.plot([g for g in gaps if g in floor], fl, "k--", lw=1.2,
            label="mémoire coupée (plancher)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("latents intercalés entre les deux présentations de V")
    ax.set_ylabel("retention_gain  =  L1(cross) − L1(repeat)")
    ax.set_title(a.title)
    ax.set_xscale("symlog", linthresh=64)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    png = os.path.join(a.out_dir, "retention_curve.png")
    fig.savefig(png, dpi=160)
    print("figure:", png)

    csv_path = os.path.join(a.out_dir, "retention_curve.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bras", "cascade", "leak", "leak_rate", "eta_init",
                    "latents_intercales", "retention_gain", "sem", "n_graines",
                    "plancher_memoire_coupee"])
        for arm in cas + sca:
            mt = meta[arm]
            for g in sorted(curves[arm]):
                m, e, n = curves[arm][g]
                w.writerow([arm, mt.get("cascade"), mt.get("leak"), mt.get("leak_rate"),
                            mt.get("eta_init"), g, round(m, 5), round(e, 5), n,
                            round(floor.get(g, (float('nan'),))[0], 5)])
    print("csv:", csv_path)

    print()
    print("%-18s" % "bras" + "".join("%10s" % f"+{g}" for g in gaps))
    print("-" * (18 + 10 * len(gaps)))
    for arm in cas + sca:
        line = "%-18s" % arm
        for g in gaps:
            line += "%10s" % (f"{curves[arm][g][0]:.4f}" if g in curves[arm] else "-")
        print(line)
    print("%-18s" % "(memoire coupee)" + "".join(
        "%10.4f" % floor[g][0] for g in gaps if g in floor))


if __name__ == "__main__":
    main()
