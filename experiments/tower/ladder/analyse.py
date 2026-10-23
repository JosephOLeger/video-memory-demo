"""Tables + figures for the ladder runs.

    python analyse.py runs/*.jsonl --fig figs
Per (family|kind, cond): item accuracy (SE by group bootstrap), pair accuracy (pairs of variants
with different answers, both right), group accuracy (all variants right).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

CH = {2: 50.0, 4: 25.0}
MEMS = ["fifo", "fifo448", "key", "mean"]
ORDER = (["solo", "oracle", "tail"] + [f"{m}@{d}" for m in MEMS for d in (0, 30, 90, 150)]
         + [f"{m}@30L3" for m in MEMS] + ["empty", "blind", "other"])


def load(paths):
    rows, seen = [], set()
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            k = (r["family"], r["group"], r["variant"], r["cond"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    return rows


def stats(sel, nboot=1000, seed=0):
    """sel: rows of one cell. Returns item acc, SE, pair acc, SE, group acc, n items, n groups."""
    by = defaultdict(list)
    for r in sel:
        by[(r["family"], r["group"])].append(r)
    keys = list(by)
    def g_stats(rs):
        it = [r["correct"] for r in rs]
        pr = [a["correct"] * b["correct"] for a, b in itertools.combinations(rs, 2)
              if a["answer"] != b["answer"]]
        return sum(it), len(it), sum(pr), len(pr), int(all(it) and len(rs) > 1)
    gs = np.array([g_stats(by[k]) for k in keys], float)
    def agg(a):
        return (100 * a[:, 0].sum() / a[:, 1].sum(), 100 * a[:, 2].sum() / max(1, a[:, 3].sum()),
                100 * a[:, 4].mean())
    point = agg(gs)
    rng = np.random.default_rng(seed)
    boots = np.array([agg(gs[rng.integers(0, len(gs), len(gs))]) for _ in range(nboot)])
    se = boots.std(0)
    return {"acc": point[0], "se": se[0], "pair": point[1], "pse": se[1], "grp": point[2],
            "n": int(gs[:, 1].sum()), "ng": len(gs)}


def table(rows, key, conds):
    cells = defaultdict(list)
    for r in rows:
        cells[(r[key], r["cond"])].append(r)
    groups = sorted({r[key] for r in rows})
    out = []
    for g in groups:
        sub = [r for r in rows if r[key] == g]
        ch = np.mean([CH[r["n_opt"]] for r in sub])
        out.append(f"\n### {key} = {g}  (hasard {ch:.0f} %)\n")
        out.append("| condition | n | acc % | SE | paire % | SE | groupe % |")
        out.append("|---|---|---|---|---|---|---|")
        for c in conds:
            if (g, c) not in cells:
                continue
            s = stats(cells[(g, c)])
            extra = ""
            if c == "other":
                dh = [r["donor_hit"] for r in cells[(g, c)] if "donor_hit" in r]
                if dh:
                    extra = f" (réponse du donneur choisie : {100 * np.mean(dh):.1f} %)"
            out.append(f"| {c}{extra} | {s['n']} | {s['acc']:.1f} | {s['se']:.1f} | {s['pair']:.1f} | "
                       f"{s['pse']:.1f} | {s['grp']:.1f} |")
    return "\n".join(out)


def compact(rows, key, conds):
    """One line per family: item acc of each cond."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r[key], r["cond"])].append(r)
    groups = sorted({r[key] for r in rows})
    head = "| " + key + " | n | " + " | ".join(conds) + " |"
    out = [head, "|" + "---|" * (len(conds) + 2)]
    for g in groups:
        n = len(cells.get((g, "oracle"), []))
        vals = []
        for c in conds:
            s = cells.get((g, c))
            vals.append(f"{100 * np.mean([r['correct'] for r in s]):.0f}" if s else "")
        out.append(f"| {g} | {n} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def figure(rows, key, path, metric="acc"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = sorted({r[key] for r in rows})
    D = sorted({r["D"] for r in rows if "D" in r and r.get("L", 0) == 0})
    colors = {"oracle": "#222222", "fifo448": "#17becf", "tail": "#ff7f0e", "fifo": "#1f77b4", "key": "#2ca02c", "mean": "#d62728",
              "empty": "#999999", "other": "#9467bd"}
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), squeeze=False)
    for ax, g in zip(axes[0], groups):
        sub = [r for r in rows if r[key] == g]
        ch = np.mean([CH[r["n_opt"]] for r in sub])
        for m in ("fifo", "fifo448", "key", "mean"):
            if not any(r["cond"].startswith(m + "@") for r in sub):
                continue
            ys, es = [], []
            for d in D:
                s = stats([r for r in sub if r["cond"] == f"{m}@{d}"], nboot=300)
                ys.append(s[metric]); es.append(s["se" if metric == "acc" else "pse"])
            ax.errorbar(D, ys, yerr=es, marker="o", color=colors[m], label=m, capsize=3)
        for m, ls in (("oracle", "-"), ("tail", "-."), ("empty", ":"), ("other", "--")):
            sel = [r for r in sub if r["cond"] == m]
            if sel:
                s = stats(sel, nboot=300)
                ax.axhline(s[metric], color=colors[m], ls=ls, lw=1.2, label=m)
        if metric == "acc":
            ax.axhline(ch, color="k", lw=0.6, alpha=0.4)
        ax.axvspan(-5, 80 - 24, color="#1f77b4", alpha=0.06)
        ax.set_title(f"{g}")
        ax.set_xlabel("D (s) entre fin d'épisode et fenêtre")
        ax.set_ylim(-3, 103)
        ax.set_xticks(D)
    axes[0][0].set_ylabel("exactitude par question %" if metric == "acc" else "exactitude par paire %")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--families", default=None)
    a = ap.parse_args()
    rows = load(a.paths)
    if a.families:
        keep = set(a.families.split(","))
        rows = [r for r in rows if r["family"] in keep]
    conds = [c for c in ORDER if any(r["cond"] == c for r in rows)]
    print("## Résumé par famille (exactitude par question %)\n")
    print(compact(rows, "family", conds))
    print("\n## Par catégorie\n")
    print(table(rows, "kind", conds))
    print("\n## Par famille\n")
    print(table(rows, "family", conds))
    if a.fig:
        os.makedirs(a.fig, exist_ok=True)
        figure(rows, "kind", os.path.join(a.fig, "acc_vs_D_kind.png"))
        figure(rows, "family", os.path.join(a.fig, "acc_vs_D_family.png"))
        figure(rows, "kind", os.path.join(a.fig, "pair_vs_D_kind.png"), metric="pair")


if __name__ == "__main__":
    main()
