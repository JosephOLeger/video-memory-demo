"""How far back can the memory reach, as a function of how big it is?

This is the sizing question, and it is the one a GPU request has to answer. The
retention benchmark already asks "is a trace of ``V`` still there after ``n``
intervening latents?"; sweeping the memory's width turns that into a *curve*:

    reach(N)  =  the distance at which retention_gain falls to a fraction of its
                 value at distance 0, while the memory state is still healthy

and the curve says what a bigger card buys. Read in the only unit that matters
for EgoLifeQA -- **minutes of video** -- because the target is fixed and external:
the median question there spans 5.9 h, and no amount of tuning moves that.

Units
-----
One latent is 0.469 s of video. That is measured, not assumed: the val cache holds
2000 sequences of 64 latents built from 2000 distinct 30 s EgoLife segments, so a
64-latent block *is* 30 s. Pass ``--seconds-per-latent`` if a cache is rebuilt at
another rate.

Censoring is reported, not hidden
---------------------------------
A memory whose gain never falls below the threshold inside the measured grid has a
reach *greater than* the largest gap tested -- a right-censored observation, not a
measurement of "exactly 32 minutes". Treating those as if they were observed would
bend the fitted exponent downwards precisely where the argument needs it to be
trustworthy, so they are excluded from the fit and counted in the output.

    python -m titans.worldmodel.capacity_reach --json retention_cap.json \\
        --train-json 'cap_h*_s*.json' --out-dir results/capacity
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

#: 30 s of EgoLife video per 64-latent sequence (measured on the val cache).
SECONDS_PER_LATENT = 30.0 / 64.0

#: EgoLifeQA's median question span. The external target the sizing is judged on.
EGOLIFEQA_MEDIAN_S = 5.9 * 3600.0


def hidden_from_arm(arm: str):
    """``h128`` -> 128. The cap sweep's naming; ``None`` when it does not apply."""
    m = re.search(r"h(\d+)", arm)
    return int(m.group(1)) if m else None


def sizes_from_training(pattern: str) -> dict:
    """``{(arm, seed): {"hidden": h, "params": p}}`` from the sweep's own results.

    Preferred over parsing names: it reports what the run actually built. ``label``
    carries a ``cap_`` prefix the artifacts do not, so it is stripped.
    """
    out = {}
    for p in sorted(globmod.glob(pattern)):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        arm = re.sub(r"^cap_", "", str(d.get("label", "")))
        if not arm or "seed" not in d:
            continue
        out[(arm, int(d["seed"]))] = {"hidden": d.get("memory_hidden_dim"),
                                      "params": d.get("params"),
                                      "dim": d.get("dim")}
    return out


def memory_params(dim, hidden, num_layers: int = 2) -> float:
    """Weights in the memory MLP -- the ``N`` the reach law is written in.

    ``dim -> hidden -> dim`` for the 2-layer default. Biases are excluded: they are
    ``O(hidden)`` against ``O(dim*hidden)`` and, under ``update_bias=False``, are
    not part of the online-updated state at all.
    """
    if dim is None or hidden is None:
        return float("nan")
    return float(dim * hidden * 2 if num_layers == 2 else dim * hidden * num_layers)


def seed_of(fname: str):
    m = re.search(r"_s(\d+)\.pt$", fname)
    return int(m.group(1)) if m else None


def reach_of(curve, frac: float = 0.1):
    """Distance in latents at which the gain drops to ``frac`` of its value at 0.

    Returns ``(reach_latents, censored, reason)``. The walk stops at the first gap
    the classifier does not call ``sain``: past a collapsed or diverged state the
    gain is not measuring retention any more, whatever its sign
    (a near-rank-1 state stays perfectly finite and scores whatever it likes).
    """
    pts = [c for c in sorted(curve, key=lambda c: c["gap_latents"])]
    if not pts or pts[0]["gap_latents"] != 0:
        return float("nan"), False, "no gap-0 reference"
    g0 = pts[0]["retention_gain"]
    if not (isinstance(g0, (int, float)) and math.isfinite(g0)) or g0 <= 0:
        return 0.0, False, "no trace even at distance 0"
    tau = frac * g0

    healthy = []
    for c in pts:
        if c["status"] != "sain":
            break
        healthy.append(c)
    if len(healthy) < 2:
        return 0.0, False, f"state stops being healthy at {pts[len(healthy)]['gap_latents']}"

    for prev, cur in zip(healthy, healthy[1:]):
        if cur["retention_gain"] < tau:
            x0, x1 = prev["gap_latents"], cur["gap_latents"]
            y0, y1 = prev["retention_gain"], cur["retention_gain"]
            # ``y0 > y1`` holds at the *crossing* interval whatever the curve does
            # elsewhere: this is the first interval to trigger, so ``y0 >= tau``
            # (or ``y0 = g0 > tau`` at the start) while ``y1 < tau``. A curve that
            # rises again later never reaches this line -- see the walk above.
            f = (y0 - tau) / (y0 - y1)
            # Interpolate in log-distance where both ends are positive: the grid is
            # geometric, so a linear reading between 512 and 1024 would sit high.
            if x0 > 0:
                return float(math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))), \
                    False, "crossing"
            return float(f * x1), False, "crossing (from 0)"

    last = healthy[-1]
    reason = ("still above threshold at the largest gap tested"
              if len(healthy) == len(pts) else
              f"healthy grid ends at {last['gap_latents']} "
              f"({pts[len(healthy)]['status']} beyond)")
    return float(last["gap_latents"]), True, reason


def loglog_fit(xs, ys):
    """Least-squares slope/intercept of ``log y`` on ``log x``, plus its stderr."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
           if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    n = len(pts)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan"), n
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    resid = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    se = math.sqrt(resid / (n - 2) / sxx) if n > 2 else float("nan")
    return b, a, se, n


def size_for_reach(target_latents: float, b: float, a: float) -> float:
    """Invert the fitted law: the ``N`` whose reach is ``target_latents``."""
    if not (math.isfinite(b) and math.isfinite(a)) or b <= 0:
        return float("nan")
    return math.exp((math.log(target_latents) - a) / b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="output of titans.worldmodel.retention")
    ap.add_argument("--train-json", default=None,
                    help="glob over the sweep's result jsons, for exact sizes")
    ap.add_argument("--frac", type=float, default=0.1,
                    help="reach = distance where the gain falls to frac of its gap-0 value")
    ap.add_argument("--dim", type=int, default=128,
                    help="token dim, used only when --train-json is absent")
    ap.add_argument("--seconds-per-latent", type=float, default=SECONDS_PER_LATENT,
                    dest="spl")
    ap.add_argument("--target-seconds", type=float, default=EGOLIFEQA_MEDIAN_S)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fig", action="store_true", help="also draw the figure")
    a = ap.parse_args()

    rows = json.load(open(a.json))
    sizes = sizes_from_training(a.train_json) if a.train_json else {}

    recs = []
    for r in rows:
        if "curve" not in r:
            continue
        arm, seed = r["arm"], seed_of(r.get("file", ""))
        known = sizes.get((arm, seed), {})
        hidden = known.get("hidden") or hidden_from_arm(arm)
        parsed = hidden_from_arm(arm)
        if known.get("hidden") and parsed and known["hidden"] != parsed:
            raise SystemExit(f"{arm}_s{seed}: name says hidden={parsed}, "
                             f"the run recorded {known['hidden']}")
        N = memory_params(known.get("dim") or a.dim, hidden)
        lat, cens, why = reach_of(r["curve"], a.frac)
        recs.append({"arm": arm, "seed": seed, "hidden": hidden, "N": N,
                     "reach_latents": lat, "reach_s": lat * a.spl,
                     "censored": cens, "why": why})

    if not recs:
        raise SystemExit(f"{a.json} holds no scored curve.")

    by_arm = defaultdict(list)
    for rec in recs:
        by_arm[rec["arm"]].append(rec)

    tgt_lat = a.target_seconds / a.spl
    print(f"reach = distance at which retention_gain falls to {a.frac:.0%} of its "
          f"value at distance 0, while the state is still healthy")
    print(f"1 latent = {a.spl:.3f} s ; cible = {a.target_seconds / 3600:.2f} h "
          f"= {tgt_lat:,.0f} latents\n")
    print("%-8s %6s %10s %12s %12s %9s" %
          ("bras", "hidden", "N", "portee (min)", "etendue", "censures"))
    print("-" * 62)
    order = sorted(by_arm, key=lambda k: (by_arm[k][0]["hidden"] or 0))
    for arm in order:
        g = by_arm[arm]
        mins = [x["reach_s"] / 60 for x in g if math.isfinite(x["reach_s"])]
        nc = sum(1 for x in g if x["censored"])
        med = st.median(mins) if mins else float("nan")
        rng = f"{min(mins):.1f}-{max(mins):.1f}" if mins else "-"
        print("%-8s %6s %10.0f %12.1f %12s %5d/%d" %
              (arm, g[0]["hidden"], g[0]["N"], med, rng, nc, len(g)))

    clean = [x for x in recs if not x["censored"] and math.isfinite(x["reach_latents"])
             and x["reach_latents"] > 0]
    b, a0, se, n = loglog_fit([x["N"] for x in clean], [x["reach_latents"] for x in clean])
    ncens = sum(1 for x in recs if x["censored"])
    print(f"\najustement log-log sur {n} points non censures "
          f"({ncens} censures exclues) :")
    print(f"  portee ~ N^{b:.3f}" + (f" +/- {se:.3f}" if math.isfinite(se) else ""))
    if math.isfinite(b) and math.isfinite(se) and se > 0:
        print(f"  ecart a la loi predite N^0.93 : {(b - 0.93) / se:+.1f} sigma")
    need = size_for_reach(tgt_lat, b, a0)
    if math.isfinite(need):
        cur = max(x["N"] for x in recs if math.isfinite(x["N"]))
        print(f"  pour {a.target_seconds / 3600:.2f} h a pleine cadence : "
              f"N = {need:,.0f} parametres de memoire, soit x{need / cur:,.0f} "
              f"la plus grande memoire mesuree ici (N = {cur:,.0f})")
        if ncens:
            print("  (extrapolation a lire avec les censures ci-dessus : la pente est "
                  "un minorant tant que les grandes memoires ne sont pas mesurees "
                  "jusqu'a leur mort)")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        csv_path = os.path.join(a.out_dir, "capacity_reach.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bras", "graine", "hidden", "N_parametres_memoire",
                        "portee_latents", "portee_s", "portee_min", "censure", "motif"])
            for x in sorted(recs, key=lambda x: (x["hidden"] or 0, x["seed"] or 0)):
                w.writerow([x["arm"], x["seed"], x["hidden"], x["N"],
                            round(x["reach_latents"], 1), round(x["reach_s"], 1),
                            round(x["reach_s"] / 60, 2), x["censored"], x["why"]])
        print("\ncsv:", csv_path)
        if a.fig:
            print("figure:", _figure(recs, by_arm, order, b, a0, a, tgt_lat))


def _figure(recs, by_arm, order, b, a0, args, tgt_lat) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    obs = [x for x in recs if not x["censored"]]
    cen = [x for x in recs if x["censored"]]
    ax.scatter([x["N"] for x in obs], [x["reach_s"] / 60 for x in obs],
               s=26, color="#1f4e79", label="portée mesurée", zorder=3)
    if cen:
        ax.scatter([x["N"] for x in cen], [x["reach_s"] / 60 for x in cen],
                   s=40, facecolors="none", edgecolors="#1f4e79", marker="^",
                   label="censurée (portée ≥ borne)", zorder=3)
    if math.isfinite(b):
        xs = [min(x["N"] for x in recs), size_for_reach(tgt_lat, b, a0)]
        if all(math.isfinite(x) and x > 0 for x in xs):
            ax.plot(xs, [math.exp(a0 + b * math.log(x)) * args.spl / 60 for x in xs],
                    "--", color="#c0392b", lw=1.4,
                    label=f"ajustement  portée ∝ N^{b:.2f}")
    ax.axhline(args.target_seconds / 60, color="k", lw=1.0, ls=":")
    ax.text(0.01, args.target_seconds / 60 * 1.08,
            f"EgoLifeQA, question médiane ({args.target_seconds / 3600:.1f} h)",
            transform=ax.get_yaxis_transform(), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N — poids de la mémoire (dim × hidden × 2)")
    ax.set_ylabel("portée épisodique (minutes de vidéo)")
    ax.set_title("Ce qu'une mémoire plus grande achète en portée")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "capacity_reach.png")
    fig.savefig(png, dpi=160)
    return png


if __name__ == "__main__":
    main()
