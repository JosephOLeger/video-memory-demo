"""The reactivity/stability trade-off as one figure, on a shared time axis.

Four panels, and the pairing is the point:

  A  reactivity   ``mem_effect`` at distance 0 against the accumulation horizon
  B  stability    tokens held before the state stops being finite, same axis
  C  divergence   ``||S||`` along a stream -- geometric growth, log scale
  D  collapse     ``pr`` along the same stream -- rank falling to a floor

A and B share their x axis so the trade-off is read by looking down a column: the
arms that sit highest in A sit lowest in B. C and D show *why* the two families end
differently, on the two observables that separate them -- reading either alone
conflates the failure modes (see :mod:`titans.worldmodel.tradeoff`).

Deliberately **not** a dual-axis chart. Reactivity and survival differ by four
orders of magnitude and share no unit; putting them on one pair of axes would make
the trade-off look like whatever the two scalings happened to imply.

    python -m titans.worldmodel.plot_tradeoff --root results/box --out-dir figs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOK = 64                       # latents per repetition of the length probe
PROBE_CAP = 1000 * TOK         # the long probe's own ceiling: 64 000 tokens

# Categorical slots 1 and 2 of the validated default palette, plus its ink steps.
# Two series only, so the all-pairs check applies and passes (CVD dE 24.7).
C_SCALAR, C_CASCADE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
# Marker shape carries identity as well as hue, so the pair never depends on colour.
M_SCALAR, M_CASCADE = "o", "s"


def tau_acc(cascade: bool, eta: float | None, leak: str | None,
            rate: float | None, theta_bar: float) -> float | None:
    """Accumulation horizon in tokens; see :mod:`titans.worldmodel.tradeoff`."""
    if cascade:
        if leak == "none" or rate == 0:
            return math.inf
        return None if not rate else 1.0 / (rate * theta_bar)
    return 1.0 / (1.0 - (eta if eta is not None else 0.5))


def load_arms(root: str, theta_bar: float) -> list[dict]:
    """One row per arm, pooling every seed that has a survival probe.

    Two probe sources, deliberately pooled: ``vvlong_<arm>.json`` (1000 repetitions,
    one seed per arm -- the earlier sweep never probed more) and
    ``taucrit/<arm>_s<seed>.json`` (200 repetitions, every seed). Their ceilings
    differ, so ``tokens_held`` is censored differently between them and is reported
    as a median with the ceiling noted; **the divergence *fraction* is not censored**,
    because every arm that ever diverged did so before 400 tokens and both ceilings
    are far past that. That is why the headline panel is the fraction and not the
    survival time.

    The fraction is the quantity ``tau*`` is defined on: the divergence is documented
    as seed-dependent, so a single seed cannot locate a threshold -- it can only say
    which side of it that seed fell.
    """
    react = defaultdict(list)
    for p in glob.glob(os.path.join(root, "retention_out", "*.json")):
        try:
            recs = json.load(open(p))
        except Exception:
            continue
        for r in recs if isinstance(recs, list) else [recs]:
            if "curve" not in r or r.get("T") != 64:
                continue
            for c in r["curve"]:
                if c["gap_latents"] == 0:
                    react[r["arm"]].append(c["l1_repeat_memoff"] - c["l1_repeat"])

    probes = defaultdict(list)
    for pat in ("vvlong_*.json", os.path.join("taucrit", "*.json")):
        for p in sorted(glob.glob(os.path.join(root, pat))):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            sm = (d.get("summary") or {}).get("loop")
            if not sm:
                continue
            probes[d["label"]].append({
                "cascade": bool(d.get("cascade")), "eta_init": d.get("eta_init"),
                "leak": d.get("leak"), "leak_rate": d.get("leak_rate"),
                "held": sm["reps_finite"] * TOK, "cap": (d.get("reps") or 0) * TOK,
                "diverged": not sm["all_finite"],
            })

    rows = []
    for arm, ps in probes.items():
        d0 = ps[0]
        tau = tau_acc(d0["cascade"], d0["eta_init"], d0["leak"], d0["leak_rate"],
                      theta_bar)
        if tau is None:
            continue
        held = sorted(x["held"] for x in ps)
        me = react.get(arm)
        rows.append({
            "arm": arm, "cascade": d0["cascade"], "tau": tau,
            "mem_effect": st.mean(me) if me else None,
            "n_seeds": len(ps),
            "n_diverged": sum(x["diverged"] for x in ps),
            "frac_diverged": sum(x["diverged"] for x in ps) / len(ps),
            "held": held[len(held) // 2],
            "cap": max(x["cap"] for x in ps),
        })
    return rows


def load_traces(root: str, arms: list[str]) -> dict:
    """``(tokens, ||S||, pr)`` from the fine-resolution probes, per arm."""
    out = {}
    for a in arms:
        p = os.path.join(root, "tau", f"{a}.json")
        if not os.path.exists(p):
            continue
        tr = (json.load(open(p)).get("streams") or {}).get("loop") or []
        pts = [x for x in tr if "b0" in x and x["b0"].get("finite")]
        out[a] = ([x["rep"] * TOK for x in pts],
                  [x["b0"]["S_norm"] for x in pts],
                  [x["b0"]["pr"] for x in pts])
    return out


def _log_x_with_infinity(ax, fin: float):
    """Log x-axis where the last decade stands for ``tau_acc = infinity``.

    A leakless cascade never forgets, so its horizon has no finite position on a log
    axis. Drawing it at the right edge is fine only if the axis says so: the tick is
    relabelled and a rule separates it, otherwise the reader takes 1e4 tokens for a
    measurement.
    """
    ax.set_xscale("log")
    ax.axvline(fin / 3.2, color=GRID, linewidth=1, linestyle=(0, (2, 3)))
    ticks = [t for t in (1, 10, 100, 1000) ] + [fin]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "10", "100", "1000", "$\\infty$"])
    ax.minorticks_off()


def _style(ax, xlabel, ylabel, title):
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def build(root: str, out_dir: str, theta_bar: float) -> str:
    rows = load_arms(root, theta_bar)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    FIN = 1e4                      # where an infinite horizon is drawn on a log axis

    def xy(rows, key):
        return ([FIN if r["tau"] == math.inf else r["tau"] for r in rows],
                [r[key] for r in rows])

    # --- A: reactivity ------------------------------------------------------
    ax = axes[0][0]
    for cas, col, mk, lab in ((False, C_SCALAR, M_SCALAR, "momentum scalaire"),
                              (True, C_CASCADE, M_CASCADE, "cascade")):
        sub = [r for r in rows if r["cascade"] == cas and r["mem_effect"] is not None]
        x, y = xy(sub, "mem_effect")
        ax.plot(x, y, mk, color=col, markersize=8, linestyle="none", label=lab,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    _style(ax, "horizon d'accumulation $\\tau_{acc}$ (tokens)",
           "mem_effect à distance 0", "A · réactivité — plus haut = mieux")
    _log_x_with_infinity(ax, FIN)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

    # --- B: where the transition is ----------------------------------------
    ax = axes[0][1]
    for cas, col, mk, lab in ((False, C_SCALAR, M_SCALAR, "momentum scalaire"),
                              (True, C_CASCADE, M_CASCADE, "cascade")):
        sub = sorted([r for r in rows if r["cascade"] == cas],
                     key=lambda r: (FIN if r["tau"] == math.inf else r["tau"]))
        x, y = xy(sub, "frac_diverged")
        ax.plot(x, y, mk, color=col, markersize=8, linestyle="none", label=lab,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        for r, xi, yi in zip(sub, x, y):      # n is not decoration: the fraction of
            ax.annotate("%d" % r["n_seeds"],  # 1 seed is 0 or 1 and means nothing
                        xy=(xi, yi), xytext=(0, 9), textcoords="offset points",
                        color=INK2, fontsize=7.5, ha="center")
    ax.set_ylim(-0.08, 1.18)
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("$\\tau^*$ : la moitié des graines rompt", xy=(1.4, 0.5),
                xytext=(1.4, 0.56), color=INK2, fontsize=8)
    _style(ax, "horizon d'accumulation $\\tau_{acc}$ (tokens)",
           "fraction des graines qui divergent",
           "B · le seuil — plus bas = mieux  (chiffre = graines)")
    _log_x_with_infinity(ax, FIN)

    # --- C and D: the two signatures ---------------------------------------
    tr = load_traces(root, ["nocas_e0.95", "cas_none"])
    pairs = [("nocas_e0.95", C_SCALAR, "scalaire $\\eta$=0.95"),
             ("cas_none", C_CASCADE, "cascade sans fuite")]

    ax = axes[1][0]
    for a, col, lab in pairs:
        if a in tr:
            t, sn, _ = tr[a]
            ax.plot(t, sn, "-", color=col, linewidth=2, label=lab)
            if a.startswith("cas"):
                ax.plot(t[-1], sn[-1], "x", color=col, markersize=9, markeredgewidth=2)
                ax.annotate("NaN", xy=(t[-1], sn[-1]), xytext=(8, -2),
                            textcoords="offset points", color=col, fontsize=8.5)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 6)          # headroom: the cascade exits the frame otherwise
    _style(ax, "tokens", "$\\|S\\|$ (échelle log)",
           "C · signature de la divergence — la norme explose")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    ax = axes[1][1]
    for a, col, lab in pairs:
        if a in tr:
            t, _, pr = tr[a]
            ax.plot(t, pr, "-", color=col, linewidth=2, label=lab)
            if a.startswith("cas"):
                ax.plot(t[-1], pr[-1], "x", color=col, markersize=9, markeredgewidth=2)
                ax.annotate("NaN", xy=(t[-1], pr[-1]), xytext=(8, -2),
                            textcoords="offset points", color=col, fontsize=8.5)
    _style(ax, "tokens", "rang effectif de $w_1$ (PR)",
           "D · signature du collapse — le rang tombe")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

    fig.suptitle("Compromis réactivité / stabilité de la mémoire-poids",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "compromis_reactivite_stabilite.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())

    # The table view the accessibility pass requires, and the numbers to cite.
    csv = os.path.join(out_dir, "compromis_reactivite_stabilite.csv")
    with open(csv, "w") as fh:
        fh.write("bras,famille,tau_acc_tokens,mem_effect_d0,graines,"
                 "graines_divergentes,fraction,tokens_tenus_median,plafond_sonde\n")
        for r in sorted(rows, key=lambda r: (r["cascade"], r["tau"])):
            fh.write("%s,%s,%s,%s,%d,%d,%.2f,%d,%d\n" % (
                r["arm"], "cascade" if r["cascade"] else "scalaire",
                "inf" if r["tau"] == math.inf else "%.1f" % r["tau"],
                "" if r["mem_effect"] is None else "%.4f" % r["mem_effect"],
                r["n_seeds"], r["n_diverged"], r["frac_diverged"],
                r["held"], r["cap"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="folder holding the pulled JSONs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--theta-bar", type=float, default=0.05, dest="theta_bar",
                    help="mean realised theta_t, for the cascade's tau_acc")
    a = ap.parse_args()
    print(build(a.root, a.out_dir, a.theta_bar))


if __name__ == "__main__":
    main()
