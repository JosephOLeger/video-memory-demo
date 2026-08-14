"""Build the trajectory queue for the eta values that bracket tau*.

Everything we know about the threshold is a snapshot at 2500 steps, and the 40 000-step
run showed 2500 is only ~30% of the asymptotic mem_effect. So "tau* = X" is really
"tau*(2500 steps) = X", and we cannot say whether the threshold itself moves as the
outer loop trains -- which is the question underneath "a memory trained on seconds of
video stops learning on an hour of it".

This picks three eta values around the measured transition and queues them at 10 000
steps with a snapshot every 1250, matching frozen_cv/ and frozen_nh/ exactly so the new
arms drop straight into the existing trajectory comparisons.

It refuses to guess: if the probes do not actually bracket the transition, it writes
nothing and says which side everything fell on. A queue built on an unbracketed sweep
would burn 50 GPU-hours confirming a threshold nobody located.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

SEEDS = (0, 1, 2)
STEPS = 10000
EVERY = 1250


def _seed_of(path, rec):
    m = re.search(r"_s(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return (rec.get("meta") or {}).get("seed")


def read_probes(root):
    """(eta, seed) -> diverged?, preferring the per-seed probes over the old n=1 runs."""
    seen = {}
    for pat in (os.path.join(root, "vvlong_nocas_e*.json"),      # older, one per arm
                os.path.join(root, "taucrit", "*.json")):        # newer, one per seed
        for p in sorted(glob.glob(pat)):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if d.get("cascade"):
                continue
            eta, sm = d.get("eta_init"), (d.get("summary") or {})
            if eta is None or "loop" not in sm:
                continue
            seen[(eta, _seed_of(p, d))] = (
                not sm["loop"]["all_finite"],
                not (sm.get("cycle") or {}).get("all_finite", True),
            )
    per = defaultdict(list)
    for (eta, _), v in seen.items():
        per[eta].append(v)
    return per


def _straddle(rows, key):
    """The two eta straddling a divergence fraction of 0.5, on one stream."""
    lo = hi = None
    for r in rows:                      # rows are sorted by eta, hence by tau
        if r[key] < 0.5:
            lo = r
        elif hi is None:
            hi = r
    return lo, hi


def bracket(per):
    rows = []
    for eta in sorted(per):
        v = per[eta]
        rows.append({"eta": eta, "tau": 1.0 / (1.0 - eta), "n": len(v),
                     "frac_loop": sum(a for a, _ in v) / len(v),
                     "frac_cycle": sum(b for _, b in v) / len(v)})
    lo, hi = _straddle(rows, "frac_loop")
    return rows, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="jobs_traj.txt")
    ap.add_argument("--train", default="cache/egolife_pca128_train.pt")
    ap.add_argument("--val", default="cache/egolife_pca128_val.pt")
    ap.add_argument("--python", default="$HOME/venvs/titans/bin/python")
    a = ap.parse_args()

    rows, lo, hi = bracket(read_probes(a.root))
    if not rows:
        print("aucune sonde de survie : rien a faire")
        return
    print("%-8s %8s %6s %10s %10s" % ("eta", "tau_acc", "n", "div loop", "div cycle"))
    for r in rows:
        print("%-8.3f %8.0f %6d %10.2f %10.2f"
              % (r["eta"], r["tau"], r["n"], r["frac_loop"], r["frac_cycle"]))
    # tau* is not one number: cycle (a rotating set of clips) is the harsher stream and
    # puts the threshold lower than loop (one clip repeated). Reported so the trajectory
    # arms are read against the stream they were chosen on, not against "the" threshold.
    clo, chi = _straddle(rows, "frac_cycle")
    if clo and chi:
        print("\nsur le flux cycle, plus dur : tau* entre %.0f et %.0f tokens"
              % (clo["tau"], chi["tau"]))

    if lo is None or hi is None:
        side = "tous survivent" if hi is None else "tous divergent"
        print("\npas d'encadrement sur loop (%s) : aucune file ecrite." % side)
        return

    tau_mid = math.sqrt(lo["tau"] * hi["tau"])          # geometric: tau spans decades
    etas = sorted({lo["eta"], round(1.0 - 1.0 / tau_mid, 4), hi["eta"]})
    print("\nencadrement : tau* entre %.0f et %.0f tokens" % (lo["tau"], hi["tau"]))
    print("trajectoires demandees pour eta = %s"
          % ", ".join("%.4g" % e for e in etas))

    lines = []
    for eta in etas:
        tag = ("%.4g" % eta).replace("0.", "0.")
        for s in SEEDS:
            base = "nocas_e%s_s%d" % (tag, s)
            lines.append(
                "%s -m titans.worldmodel.echo --train %s --val %s "
                "--horizon 24 --seg 8 --steps %d --seed %d --eta-init %s "
                "--label T_e%s --out T_e%s_s%d.json "
                "--frozen-out frozen_traj/%s.pt --frozen-every %d"
                % (a.python, a.train, a.val, STEPS, s, tag, tag, tag, s, base, EVERY))
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("%s : %d cellules, ~%.0f h GPU (5.4 h/cellule mesuree a 10 000 pas)"
          % (a.out, len(lines), 5.43 * len(lines)))


if __name__ == "__main__":
    main()
