"""Tables + figures for the v3 ladder (runs3/*.jsonl).

Scores per (family, condition), SE by bootstrap over GROUPS:
  item  per-question accuracy
  grp   group accuracy (all 4 counterfactual variants right)  <- certification metric
  cgrp  CONTRASTIVE group accuracy: letter logits minus their mean over the 4 variants of
        the group, then argmax (removes any constant option preference)
    python analyse3.py runs3/main_s*.jsonl --fig figs3 [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

GIST = ["count", "colour", "big_object", "room_state", "side"]
FAM_ORDER = ["count", "colour", "big_object", "sign_number", "hat_color", "room_state", "side"]
COND_ORDER = ["oracle", "tail", "oracle_shuf", "fifo@0", "fifo224@0", "fifo@120", "fifo224@120",
              "key@120", "mean@120", "mean_shuf@120", "empty", "blind", "other"]


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


def per_group(sel):
    """-> list of (item_correct_list, contrastive_correct_list) per complete group."""
    by = defaultdict(list)
    for r in sel:
        by[r["group"]].append(r)
    out = []
    for g, rs in by.items():
        nv = len({r["variant"] for r in rs})
        if len(rs) < 4 or nv < 4:
            continue
        L = np.array([r["lg"] for r in rs])
        Lc = L - L.mean(0, keepdims=True)
        it = [r["correct"] for r in rs]
        ct = [int(np.argmax(Lc[i]) == rs[i]["answer"]) for i in range(len(rs))]
        out.append((it, ct, [r.get("donor_hit") for r in rs]))
    return out


def stats(sel, nboot=1000, seed=0):
    gs = per_group(sel)
    if not gs:
        return None
    A = np.array([[np.mean(i), float(all(i)), np.mean(c), float(all(c))] for i, c, _ in gs])
    rng = np.random.default_rng(seed)
    bs = np.array([A[rng.integers(0, len(A), len(A))].mean(0) for _ in range(nboot)])
    m, se = 100 * A.mean(0), 100 * bs.std(0)
    dh = [d for _, _, ds in gs for d in ds if d is not None]
    return {"item": m[0], "item_se": se[0], "grp": m[1], "grp_se": se[1], "citem": m[2],
            "cgrp": m[3], "cgrp_se": se[3], "ng": len(gs),
            "donor": 100 * np.mean(dh) if dh else None}


def cells(rows):
    c = defaultdict(list)
    for r in rows:
        c[(r["family"], r["cond"])].append(r)
    return c


def fmt(s, key="grp"):
    if s is None:
        return ""
    return f"{s[key]:.0f} ± {s[key + '_se']:.0f}"


def main_table(rows, fams, conds, key):
    C = cells(rows)
    head = "| mémoire | " + " | ".join(fams) + " |"
    out = [head, "|---|" + "---|" * len(fams)]
    for c in conds:
        vals = []
        for f in fams:
            s = stats(C[(f, c)]) if (f, c) in C else None
            if s is None:
                vals.append("")
            elif key == "both":
                vals.append(f"{s['grp']:.0f} ({s['item']:.0f}) n={s['ng']}")
            else:
                vals.append(fmt(s, key) + f" (n={s['ng']})")
        if any(vals):
            out.append(f"| {c} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def donor_table(rows, fams):
    C = cells(rows)
    out = ["| famille | other item % | other grp % | donneur choisi % |", "|---|---|---|---|"]
    for f in fams:
        s = stats(C.get((f, "other"), []))
        if s:
            out.append(f"| {f} | {s['item']:.1f} | {s['grp']:.0f} | {s['donor']:.0f} |")
    return "\n".join(out)


def kind_table(rows, conds, key="grp"):
    out = ["| mémoire | gist grp | détail grp | gist item | détail item |", "|---|---|---|---|---|"]
    for c in conds:
        sg = stats([r for r in rows if r["cond"] == c and r["family"] in GIST[:3]])
        sd = stats([r for r in rows if r["cond"] == c and r["family"] not in GIST])
        if sg or sd:
            out.append(f"| {c} | {fmt(sg)} | {fmt(sd)} | {sg['item']:.0f} | {sd['item']:.0f} |"
                       if sg and sd else f"| {c} | {fmt(sg)} | {fmt(sd)} | | |")
    return "\n".join(out)


def fig_main(rows, fams, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = cells(rows)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.3 * len(fams), 3.4), squeeze=False, sharey=True)
    col = {"fifo": "#1f77b4", "fifo224": "#17becf", "key": "#2ca02c", "mean": "#d62728",
           "mean_shuf": "#ff9896"}
    for ax, f in zip(axes[0], fams):
        for m in ("fifo", "fifo224"):
            xs, ys, es = [], [], []
            for D in (0, 120):
                s = stats(C.get((f, f"{m}@{D}"), []), nboot=300)
                if s:
                    xs.append(D); ys.append(s["grp"]); es.append(s["grp_se"])
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", color=col[m], label=m, capsize=3)
        for j, m in enumerate(("key", "mean", "mean_shuf")):
            s = stats(C.get((f, f"{m}@120"), []), nboot=300)
            if s:
                ax.errorbar([120 + 6 * (j + 1)], [s["grp"]], yerr=[s["grp_se"]], marker="s",
                            color=col[m], label=m, capsize=3, ls="none")
        for m, ls, c in (("oracle", "-", "k"), ("tail", "-.", "#ff7f0e"), ("empty", ":", "#999999"),
                         ("other", "--", "#9467bd")):
            s = stats(C.get((f, m), []), nboot=300)
            if s:
                ax.axhline(s["grp"], color=c, ls=ls, lw=1.1, label=m)
        ax.set_title(f + (" (gist)" if f in GIST else " (détail)"))
        ax.set_xlabel("D (s)")
        ax.set_xticks([0, 120])
        ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel("exactitude par GROUPE %")
    axes[0][-1].legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def fig_load(rows, fams, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = cells(rows)
    Ls = sorted({r["L"] for r in rows if "L" in r})
    fig, axes = plt.subplots(1, len(fams), figsize=(3.3 * len(fams), 3.4), squeeze=False, sharey=True)
    col = {"oracle_all": "k", "oracle_gap": "#8c564b", "oracle_sep": "#e377c2", "oracle": "#7f7f7f", "key": "#2ca02c", "mean": "#d62728", "empty": "#bbbbbb"}
    for ax, f in zip(axes[0], fams):
        for m in col:
            xs, ys, es = [], [], []
            for L in Ls:
                s = stats(C.get((f, f"{m}|L{L}"), []), nboot=300)
                if s:
                    xs.append(L); ys.append(s["grp"]); es.append(s["grp_se"])
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", color=col[m], label=m, capsize=3)
        ax.set_title(f)
        ax.set_xlabel("L (tableaux distracteurs)")
        ax.set_xticks(Ls)
        ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel("exactitude par GROUPE %")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()
    rows = load(a.paths)
    fams = [f for f in FAM_ORDER if any(r["family"] == f for r in rows)]
    if a.load:
        Ls = sorted({r["L"] for r in rows if "L" in r})
        conds = [f"{m}|L{L}" for m in ("oracle_all", "oracle_gap", "oracle_sep", "oracle", "key", "mean", "empty") for L in Ls]
        print("## Charge : groupe % (item %)\n")
        print(main_table(rows, fams, conds, "both"))
        print("\n## Charge : contrastif groupe %\n")
        print(main_table(rows, fams, conds, "cgrp"))
        if a.fig:
            os.makedirs(a.fig, exist_ok=True)
            fig_load(rows, fams, os.path.join(a.fig, "load.png"))
        return
    conds = [c for c in COND_ORDER if any(r["cond"] == c for r in rows)]
    print("## Groupe % ± SE (n groupes)\n")
    print(main_table(rows, fams, conds, "grp"))
    print("\n## Groupe % (item %)\n")
    print(main_table(rows, fams, conds, "both"))
    print("\n## Contrastif, groupe % ± SE\n")
    print(main_table(rows, fams, conds, "cgrp"))
    print("\n## Contrôle autre variante\n")
    print(donor_table(rows, fams))
    print("\n## Gist (count, colour, big_object) contre détail (sign_number, hat_color)\n")
    print(kind_table(rows, conds))
    if a.fig:
        os.makedirs(a.fig, exist_ok=True)
        fig_main(rows, fams, os.path.join(a.fig, "grid_v3.png"))


if __name__ == "__main__":
    main()
