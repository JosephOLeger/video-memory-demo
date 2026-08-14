"""Cascade against eta->1: does it add, is it weaker, or is it the same bound moved?

Both arms are one-parameter families indexed by the *same* thing -- how little the
momentum forgets. The scalar's knob is ``eta_init``; the cascade's is
``cascade_leak_rate``, and ``leak="none"`` is its ``eta = 1`` (measured 2026-08-06:
the two ladders line up, and ``r = 0.01`` is indistinguishable from no leak at all).

So the comparison is not two points. It is two curves in the (memory, stability)
plane, and the question is whether they are the same curve:

* **the same curve**  -> the cascade is a reparametrisation. It buys nothing; it
  indexes the identical trade-off by a different variable.
* **cascade above**   -> it adds: at equal stability it remembers more, or at equal
  memory it is safer.
* **cascade below**   -> it is strictly worse and the extra 47% VRAM is wasted.

Memory axis: ``memory_gain`` from the [V;V] echo, pooled over seeds.
Stability axis: the margin ``2.37 / max_chunk(theta_t * lambda_hidden)`` from
:mod:`titans.worldmodel.margin`, and where available the length probe's drift.

**Health is printed next to both.** A memory that has quietly stopped carrying
anything has an excellent margin, and would otherwise read as the most stable arm on
the board -- a mistake this project has actually made.

    python -m titans.worldmodel.vv_compare --glob 'vv_*.json' --margins vv_margins.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import statistics as st
from collections import defaultdict


def load(pattern):
    """Every dict-shaped JSON matching ``pattern``.

    Non-dicts are skipped rather than crashing: ``vv_margins.json`` is a *list* and
    it sits right inside the natural ``vv_*.json`` glob, so a strict reader breaks
    the moment phase 2 of the chain writes it.
    """
    out = []
    for p in globmod.glob(pattern):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def spc(r):
    s = r.get("space")
    return s.get("kind", str(s)) if isinstance(s, dict) else str(s)


def arm_of(r):
    """``(famille, bouton)`` -- the position of a run in its one-parameter family."""
    if r.get("cascade"):
        lr = r.get("leak_rate")
        knob = 0.0 if r.get("leak") in (None, "none") else float(lr if lr else 0.0)
        return ("cascade", knob)
    e = r.get("eta_init")
    return ("scalaire", 0.53 if e is None else float(e))


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None, None, 0
    sd = st.stdev(vals) if len(vals) > 1 else float("nan")
    return st.mean(vals), (sd / math.sqrt(len(vals)) if len(vals) > 1 else float("nan")), len(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="vv_*.json", help="new sweep cells")
    ap.add_argument("--also", default="echo_*.json",
                    help="earlier reference-config cells to pool in")
    ap.add_argument("--margins", default="vv_margins.json")
    ap.add_argument("--long", default="vvlong_*.json")
    a = ap.parse_args()

    rows = [r for r in load(a.glob) + load(a.also)
            if spc(r) == "pca" and r.get("T") == 64 and r.get("seg") == 8
            and r.get("dim") == 128 and r.get("horizon") == 24
            and r.get("theta_scale") in (None, 0.03)
            and r.get("normalize_hidden") is None
            and isinstance(r.get("memory_gain"), (int, float))
            and math.isfinite(r["memory_gain"])
            # only the reference cascade shape, so reservoirs/readout are not a
            # hidden second variable inside the "cascade" family
            and (not r.get("cascade")
                 or (r.get("reservoirs") == 6 and r.get("readout") == "gated"
                     and r.get("topology") in (None, "chain")))]

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[arm_of(r)].append(r)

    marg = defaultdict(list)
    try:
        for m in json.load(open(a.margins)):
            if "error" in m or not m.get("all_finite"):
                continue
            marg[arm_of(m)].append(m)
    except Exception:
        pass

    lng = {}
    for r in load(a.long):
        s = r.get("summary", {})
        lng[r.get("label")] = s

    print(f"{'famille':<10}{'oubli':>7}{'n':>4}{'memory_gain':>14}{'SEM':>8}"
          f"{'n_m':>5}{'marge':>8}{'pr':>7}{'cos_ret':>9}")
    print("-" * 72)
    for fam in ("scalaire", "cascade"):
        keys = sorted([k for k in by_arm if k[0] == fam], key=lambda k: k[1])
        for k in keys:
            v = by_arm[k]
            mg, sem, n = agg([x["memory_gain"] for x in v])
            mm, _, nm = agg([x["margin"] for x in marg.get(k, [])])
            pr, _, _ = agg([x["pr"] for x in marg.get(k, [])])
            cr, _, _ = agg([x["cos_ret"] for x in marg.get(k, [])])
            knob = "none" if (fam == "cascade" and k[1] == 0.0) else f"{k[1]:g}"
            f = lambda x, d=2: (f"{x:.{d}f}" if x is not None else "-")
            print(f"{fam:<10}{knob:>7}{n:>4}{mg:>14.4f}{sem:>8.4f}"
                  f"{nm:>5}{f(mm,1):>8}{f(pr,1):>7}{f(cr,3):>9}")
        print()

    if lng:
        print("sonde de longueur (1000 repetitions, poids hors memoire geles)")
        print(f"{'bras':<16}{'flux':<10}{'l1 fin':>9}{'gain fin':>10}{'pr fin':>8}"
              f"{'cos fin':>9}{'S fin':>10}{'fini':>6}")
        for lab in sorted(lng):
            for mode in ("loop", "cycle", "shuffle"):
                s = lng[lab].get(mode)
                if not s:
                    continue
                f = lambda k, d=3: (f"{s[k]:.{d}f}" if isinstance(s.get(k), (int, float)) else "-")
                print(f"{lab:<16}{mode:<10}{f('l1_tail'):>9}{f('memory_gain_tail',4):>10}"
                      f"{f('pr_last',1):>8}{f('cos_ret_last'):>9}{f('S_norm_last',4):>10}"
                      f"{str(s.get('all_finite')):>6}")

    print("\nLecture : si les deux familles tracent la meme courbe (memory_gain, marge),")
    print("la cascade est un reparametrage. Si sa courbe est au-dessus, elle apporte.")


if __name__ == "__main__":
    main()
