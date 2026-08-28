"""Analyse des sorties de diag_pr_collapse_probe : loi contre mesure.

Lit results/prlaw/*.jsonl (rapatries du box), applique le predicteur
pr(L) (pr_law_predictor) aux statistiques mesurees, et produit :
  - parite pr final mesure vs predit (tous artefacts x blocs),
  - trajectoires pr(L) mesure vs predit (representants),
  - scalaires interpretables par bras (g_rows, g_cols, Phi, r_eff),
  - volet D : dose-reponse alpha-override mesure vs predit.

Usage : analyze_pr_law.py <dir_jsonl> <dir_figs>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_law_predictor import predict_pr_curve, scalars_from_qp

WRITES_PER_CLIP = 8  # T=64 / seg 8 : appels d'update par clip et par bloc


def load_rows(d):
    stats, trajs = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("header") or r.get("error"):
                continue
            r["_file"] = os.path.basename(f)
            (trajs if "traj" in r else stats).append(r)
    return stats, trajs


def label_of(r):
    base = r["frozen"].split("/")[-1].replace(".pt", "")
    return base


def match_traj(trajs, r):
    for t in trajs:
        if (t["frozen"] == r["frozen"] and t["mode"] == r["mode"]
                and t.get("alpha_override") == r.get("alpha_override")
                and t["_file"] == r["_file"]):
            return t["traj"]
    return None


def predicted_curve(r, Ls):
    al = r.get("alpha_override") or 0.0
    return predict_pr_curve(
        r["norms"], r["cos_init"], r["q_d"], r["p_d"],
        r["init_norm"], r["init_pr"], r["qi_mean"], r["pi_mean"],
        alpha=al, tokens_per_call=8, Ls=Ls)[1]


def main():
    din = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/josep/projects/titans-mac/results/prlaw"
    dfig = sys.argv[2] if len(sys.argv) > 2 else \
        "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs"
    stats, trajs = load_rows(din)
    print(f"{len(stats)} lignes stats, {len(trajs)} trajectoires")

    # ---------- table + parite ----------
    rows = []
    for r in stats:
        tr = match_traj(trajs, r)
        if tr is None or not r.get("norms"):
            continue
        L = r["n_writes"]
        clip_L = np.array([WRITES_PER_CLIP * (c["clip"] + 1) for c in tr])
        clip_L = np.clip(clip_L, 1, L)
        meas = np.array([c.get(f"pr_b{r['block']}") for c in tr], float)
        pred = predicted_curve(r, clip_L)
        sc = scalars_from_qp(r["q_d"], r["p_d"], r["r_eff_mean"])
        err_final = abs(pred[-1] - meas[-1]) / meas[-1]
        err_traj = float(np.nanmedian(np.abs(pred - meas) / meas))
        rows.append(dict(label=label_of(r), block=r["block"], mode=r["mode"],
                         alpha=r.get("alpha_override") or 0.0,
                         file=r["_file"], n_writes=L,
                         pr_meas=float(meas[-1]), pr_pred=float(pred[-1]),
                         err_final=err_final, err_traj=err_traj,
                         meas_curve=meas.tolist(), pred_curve=pred.tolist(),
                         clip_L=clip_L.tolist(), **sc))
    rows.sort(key=lambda r: (r["file"], r["label"], r["block"]))
    print(f"\n{'artefact':26s} b {'mode':7s} {'alpha':>6s} {'pr mes':>7s} "
          f"{'pr pred':>8s} {'ecart':>6s} {'Phi':>6s} {'g_rows':>7s} "
          f"{'g_cols':>7s} {'r_eff':>6s}")
    for r in rows:
        print(f"{r['label']:26s} {r['block']} {r['mode']:7s} "
              f"{r['alpha']:6.3f} {r['pr_meas']:7.1f} {r['pr_pred']:8.1f} "
              f"{100*r['err_final']:5.0f}% {r['phi']:6.1f} "
              f"{r['g_rows']:7.1f} {r['g_cols']:7.1f} {r['r_eff']:6.2f}")
    med = np.median([r["err_final"] for r in rows]) if rows else float("nan")
    print(f"\nmediane |ecart final| : {100*med:.1f}%  (n={len(rows)})")

    json.dump(rows, open(os.path.join(din, "analysis_rows.json"), "w"),
              indent=1)

    # ---------- figures ----------
    base = [r for r in rows if r["alpha"] == 0.0 and r["mode"] == "cycle"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for r in base:
        ax.plot(r["pr_pred"], r["pr_meas"], "o" if r["block"] == 0 else "^",
                ms=5, alpha=0.7,
                color="tab:blue" if "unit10k" in r["label"] else
                      ("tab:orange" if "usq" in r["label"] else
                       ("tab:green" if "scl" in r["label"] else
                        ("tab:purple" if r["label"].startswith("tx_") else
                         "tab:red"))))
    lim = [0, max([r["pr_meas"] for r in base] +
                  [r["pr_pred"] for r in base]) * 1.1] if base else [0, 1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("pr predit (stats d'ecritures mesurees)")
    ax.set_ylabel("pr mesure (fin de flux)")
    ax.set_title("(a) parite, artefacts geles (o=bloc 0, ^=bloc 1)",
                 fontsize=9)
    ax = axes[1]
    shown = 0
    for r in base:
        if r["block"] != 0 or shown >= 4:
            continue
        ax.plot(r["clip_L"], r["meas_curve"], "-", lw=1.5,
                label=f"{r['label']} mesure")
        ax.plot(r["clip_L"], r["pred_curve"], "--", lw=1.2,
                label=f"{r['label']} predit")
        shown += 1
    ax.set_xlabel("ecritures L")
    ax.set_ylabel("pr")
    ax.set_title("(b) trajectoires pr(L), bloc 0", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(dfig, "fig_pr_artifacts.pdf"),
                bbox_inches="tight")
    print("fig_pr_artifacts.pdf ecrit")

    # ---------- volet D : dose alpha ----------
    dose = [r for r in rows if r["alpha"] > 0 or
            (r["alpha"] == 0 and r["mode"] == "cycle"
             and "unit10k_s0" in r["label"] and "_at" not in r["label"]
             or "unit10k_s1" in r["label"] and "_at" not in r["label"])]
    dd = [r for r in rows if r["alpha"] > 0]
    if dd:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for blk, mk in [(0, "o"), (1, "^")]:
            xs = sorted({r["alpha"] for r in dd})
            for s in ["s0", "s1"]:
                sel = [r for r in dd if r["block"] == blk and s in r["label"]]
                sel.sort(key=lambda r: r["alpha"])
                if not sel:
                    continue
                ax.plot([r["alpha"] for r in sel], [r["pr_meas"] for r in sel],
                        mk + "-", ms=5, alpha=0.7,
                        label=f"mesure b{blk} {s}" if s == "s0" else None)
                ax.plot([r["alpha"] for r in sel], [r["pr_pred"] for r in sel],
                        mk + "--", ms=4, alpha=0.5, color="k",
                        label=f"predit b{blk}" if s == "s0" else None)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$ force au rejeu")
        ax.set_ylabel("pr final")
        ax.set_title("dose-reponse alpha : le terme de compte mord",
                     fontsize=9)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(dfig, "fig_pr_alpha_dose.pdf"),
                    bbox_inches="tight")
        print("fig_pr_alpha_dose.pdf ecrit")


if __name__ == "__main__":
    main()
