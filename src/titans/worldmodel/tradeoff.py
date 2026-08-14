"""The reactivity/stability trade-off, in characteristic times.

The project's central claim is now a trade-off claim: more accumulation buys
short-term recall and costs stability. Stated that way it is unfalsifiable, so this
module turns it into numbers on one shared axis.

**The shared axis is an accumulation time in tokens.** The scalar momentum
``S_t = eta S_{t-1} - theta g`` has horizon ``tau_acc = 1/(1-eta)``. The cascade's
surprise-driven leak retains ``exp(-r * theta_t)`` per token, so its horizon is
``1/(r * theta_bar)`` -- the same quantity, measured in the same unit. Putting both
families on that axis is what makes "the cascade relocates the bound rather than
extending it" checkable: at *matched* ``tau_acc`` the two are directly comparable.

**Two failure modes, two signatures, measured on different observables.**

``collapse``   ``w1`` loses rank while its norm stays bounded (it usually *shrinks*).
               The state is finite forever; the memory is simply no longer an
               associative memory. Characterised by ``tau_collapse`` (the e-folding
               time of ``pr``) and by the floor ``pr_inf`` it settles at.

``divergence`` ``||S||`` and ``||w1||`` grow geometrically until they overflow.
               Characterised by the growth rate of ``log||S||`` per token, whose
               reciprocal is an e-folding time, and by the token at which the state
               stops being finite.

Reading only ``finite`` conflates the two: a collapsed run is finite and useless.
Reading only ``pr`` conflates them too, because a diverging run's rank also falls on
the way out. The pair does separate them, which is why both are reported.

**Resolution matters and cost nothing.** With ``long_run --diag-every 10`` the first
diagnostic lands at 640 tokens, by which point every arm has already collapsed --
so the earlier sweeps could only report "tau < 640". ``--diag-every 1`` puts a point
every 64 tokens for the same forward pass. Run the probe at that resolution or this
module reports a bound rather than a time.

    python -m titans.worldmodel.long_run --frozen frozen_vv/nocas_e0.95_s0.pt \\
        --val cache/egolife_pca128_val.pt --reps 30 --diag-every 1 --out tau/x.json
    python -m titans.worldmodel.tradeoff --probes 'tau/*.json' --margins vv_margins.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
from collections import defaultdict

TOKENS_PER_REP = 64


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def tau_acc(rec: dict, theta_bar: float | None = None) -> float | None:
    """Accumulation horizon in tokens, for either family.

    Scalar: ``1/(1-eta)``. Cascade with the surprise leak: ``1/(r*theta_bar)``, and
    ``r=0`` is a pure accumulator, reported as ``inf``. Returns ``None`` when the
    arm's leak mode is not the surprise one, because then the retained fraction is
    not a function of ``theta`` and the two families do not share an axis.
    """
    if rec.get("cascade"):
        if rec.get("leak") != "surprise":
            return math.inf if rec.get("leak") == "none" else None
        r = rec.get("leak_rate")
        if r is None or theta_bar is None:
            return None
        return math.inf if r == 0 else 1.0 / (r * theta_bar)
    eta = rec.get("eta_init")
    if eta is None:
        eta = 0.5                      # sigmoid(0), the un-initialised resting value
    return 1.0 / (1.0 - eta)


def collapse_time(points: list[dict], key: str = "pr") -> dict:
    """e-folding time of ``pr``'s decay toward its floor, in tokens.

    Non-parametric on purpose: the decay is not a clean exponential (it is fast then
    plateaus), so a fitted rate would be dominated by whichever regime has more
    points. The e-folding crossing is well defined for any monotone-ish decay and is
    read off directly.
    """
    vals = [(p["rep"] * TOKENS_PER_REP, p["b0"][key]) for p in points
            if "b0" in p and _finite(p["b0"].get(key))]
    if len(vals) < 3:
        return {"tau_collapse": None, "pr0": None, "pr_inf": None, "n_points": len(vals)}
    pr0 = vals[0][1]
    tail = [v for _, v in vals[max(1, int(0.7 * len(vals))):]]
    pr_inf = sorted(tail)[len(tail) // 2]
    drop = pr0 - pr_inf
    if drop <= 0.1 * pr0:              # nothing collapsed; a time would be noise
        return {"tau_collapse": None, "pr0": pr0, "pr_inf": pr_inf,
                "n_points": len(vals), "collapsed": False}
    thresh = pr_inf + drop / math.e
    tau = next((t for t, v in vals if v <= thresh), None)
    return {"tau_collapse": tau, "pr0": pr0, "pr_inf": pr_inf,
            "n_points": len(vals), "collapsed": True}


def divergence_rate(points: list[dict], key: str = "S_norm") -> dict:
    """Growth rate of ``log ||S||`` per token, and the e-folding time ``1/rate``.

    Fitted by least squares over the finite points only, which is the whole trace up
    to the blow-up. A collapsing arm has a rate near zero (its norm is bounded), so
    the rate is what separates the two failure modes rather than merely ranking them.
    """
    vals = [(p["rep"] * TOKENS_PER_REP, p["b0"][key]) for p in points
            if "b0" in p and _finite(p["b0"].get(key)) and p["b0"].get(key, 0) > 0]
    if len(vals) < 3:
        return {"rate_per_token": None, "tau_diverge": None, "norm_first": None,
                "norm_last": None}
    xs = [t for t, _ in vals]
    ys = [math.log(v) for _, v in vals]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    rate = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return {"rate_per_token": rate,
            "tau_diverge": (1.0 / rate) if rate > 1e-9 else None,
            "norm_first": vals[0][1], "norm_last": vals[-1][1]}


def classify(surv: dict, div: dict, col: dict, reps_total: int) -> str:
    """``diverge`` / ``collapse`` / ``sain``, from the pair of signatures.

    Divergence wins when the state stopped being finite: a diverging trace also
    loses rank on its way out, so rank alone would mislabel it.
    """
    if surv["reps_finite"] < reps_total:
        return "diverge"
    if col.get("collapsed"):
        return "collapse"
    return "sain"


def rows_from(probe_glob: str, mode: str = "loop") -> list[dict]:
    out = []
    for p in sorted(globmod.glob(probe_glob)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        tr = (d.get("streams") or {}).get(mode)
        if not tr:
            continue
        pts = [x for x in tr if "b0" in x]
        summ = (d.get("summary") or {}).get(mode, {})
        reps_finite = summ.get("reps_finite", len([x for x in tr if x.get("finite")]))
        col = collapse_time(pts)
        div = divergence_rate(pts)
        out.append({
            "arm": d.get("label") or os.path.basename(p)[:-5],
            "mode": mode,
            "cascade": bool(d.get("cascade")),
            "leak": d.get("leak"), "leak_rate": d.get("leak_rate"),
            "eta_init": d.get("eta_init"),
            "tau_acc": tau_acc(d, theta_bar=d.get("theta_bar")),
            "reps_finite": reps_finite, "reps_total": d.get("reps"),
            "tokens_held": reps_finite * TOKENS_PER_REP,
            "l1_tail": summ.get("l1_tail"),
            **col, **div,
            "issue": classify({"reps_finite": reps_finite}, div, col, d.get("reps", 0)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", required=True, help="glob over long_run outputs")
    ap.add_argument("--mode", default="loop", help="loop | cycle | shuffle")
    ap.add_argument("--theta-bar", type=float, default=0.05, dest="theta_bar",
                    help="mean realised theta_t, for the cascade's tau_acc; read it "
                         "off vv_margins.json rather than guessing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = rows_from(a.probes, a.mode)
    for r in rows:                    # cascade tau needs theta, which is per-arm
        if r["cascade"] and r["leak"] == "surprise" and r["leak_rate"]:
            r["tau_acc"] = 1.0 / (r["leak_rate"] * a.theta_bar)

    rows.sort(key=lambda r: (r["cascade"], (r["tau_acc"] or 0)))
    print("mode = %s   (tau en tokens ; 64 latents par repetition)" % a.mode)
    print("%-14s%9s%9s%10s%9s%9s%11s%9s" %
          ("bras", "tau_acc", "issue", "tenus", "tau_col", "PR_inf", "taux |S|", "l1_fin"))
    print("-" * 82)
    for r in rows:
        ta = r["tau_acc"]
        print("%-14s%9s%9s%10d%9s%9s%11s%9s" % (
            r["arm"][:14],
            "inf" if ta == math.inf else ("-" if ta is None else "%.0f" % ta),
            r["issue"], r["tokens_held"],
            "-" if r["tau_collapse"] is None else "%d" % r["tau_collapse"],
            "-" if r["pr_inf"] is None else "%.1f" % r["pr_inf"],
            "-" if r["rate_per_token"] is None else "%+.4f" % r["rate_per_token"],
            "-" if r["l1_tail"] is None else "%.3f" % r["l1_tail"]))
    print()
    print("collapse : le rang tombe, la norme reste bornee -> fini pour toujours, inutile")
    print("divergence : log||S|| croit lineairement -> NaN ; taux > 0 est la signature")

    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
