"""Figures du programme collapse (28/08), depuis results/prlaw2/.

fig_collapse_training.pdf :
  (a) le collapse se construit a l'ENTRAINEMENT : pr(bloc 0) au fil des
      checkpoints 2k..10k — unit s'effondre, usq stable, decouple stable
      (bloc 1 pour dec, son bloc 0 n'ecrit pas).
  (b) la loterie des graines : pr final contre r_eff du flux d'ecriture,
      tous bras cycle — la variable cachee du collapse.

fig_collapse_dc.pdf :
  (a) dose alpha-override : pr mesure et pr predit par la formule DC
      (zero parametre libre, non tautologique).
  (b) deux canaux : pr(L) mesure contre la rampe DC (m plein-flux) pour
      un bras qui collapse (unit10k_s0 b0), un bras stable (usq10k_s0
      b0) et decouple (dec_s1 b1).
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "/home/josep/projects/titans-mac/results/prlaw2"
OUT = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs"

stats, trajs = [], {}
for f in sorted(glob.glob(f"{DIR}/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("header") or r.get("error"):
            continue
        if "traj" in r:
            trajs[(f, r["frozen"], r["mode"], r.get("alpha_override"))] = r["traj"]
        elif "pr_dc" in r:
            r["_f"] = f
            stats.append(r)


def lab(r):
    return r["frozen"].split("/")[-1].replace(".pt", "")


# ---------------- fig 1 : entrainement + loterie ----------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
steps = [2000, 4000, 6000, 8000, 10000]
series = {
    ("unit s0 (b0)", "tab:red", "mu_unit10k_s0_at{}", 0, "-"),
    ("unit s1 (b0)", "tab:red", "mu_unit10k_s1_at{}", 0, "--"),
    ("usq s0 (b0)", "tab:blue", "mu_usq10k_s0_at{}", 0, "-"),
    ("usq s1 (b0)", "tab:blue", "mu_usq10k_s1_at{}", 0, "--"),
    ("decouple s0 (b1)", "tab:green", "mu_unit10k_dec_s0_at{}", 1, "-"),
    ("decouple s1 (b1)", "tab:green", "mu_unit10k_dec_s1_at{}", 1, "--"),
}
for name, col, pat, blk, ls in sorted(series):
    ys = []
    for st in steps:
        m = [r for r in stats if lab(r) == pat.format(st) and r["block"] == blk]
        ys.append(m[0]["pr_final"] if m else np.nan)
    ax.plot(steps, ys, ls, marker="o", ms=4, color=col, label=name, lw=1.4)
ax.set_yscale("log")
ax.set_xlabel("pas d'entrainement du checkpoint")
ax.set_ylabel("pr de l'etat au rejeu (800 ecritures)")
ax.set_title("(a) le collapse se construit a l'entrainement", fontsize=9)
ax.legend(fontsize=6, ncol=2)

ax = axes[1]
fams = {"mu_unit_": ("unit 4k", "tab:red", "o"),
        "mu_unit10k_s": ("unit 10k", "tab:red", "^"),
        "mu_usq10k_s": ("unit_squash 10k", "tab:blue", "s"),
        "mu_scl_": ("softclip", "tab:green", "D"),
        "mu_unit10k_dec": ("decouple", "tab:purple", "*"),
        "tx_": ("texte", "tab:orange", "v")}
for r in stats:
    if r["mode"] != "cycle" or r.get("alpha_override") or "_at" in lab(r):
        continue
    for pref, (name, col, mk) in fams.items():
        if lab(r).startswith(pref):
            # bras decouple bloc 0 : ecritures eteintes, cas trivial — marque creux
            hollow = pref == "mu_unit10k_dec" and r["block"] == 0
            ax.plot(r["r_eff_mean"], r["pr_final"], mk, color=col, ms=7,
                    mfc="none" if hollow else col, alpha=0.75)
            break
for pref, (name, col, mk) in fams.items():
    ax.plot([], [], mk, color=col, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("r_eff moyen du flux d'ecriture (participation par ecriture)")
ax.set_ylabel("pr final de l'etat")
ax.set_title("(b) la loterie des graines EST le r_eff appris", fontsize=9)
ax.legend(fontsize=6, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_collapse_training.pdf", bbox_inches="tight")
print("fig_collapse_training.pdf")

# ---------------- fig 2 : validation DC ----------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
dose = [r for r in stats if r.get("alpha_override")]
base = [r for r in stats if r["mode"] == "cycle" and not r.get("alpha_override")
        and lab(r) in ("mu_unit10k_s0", "mu_unit10k_s1") and "_at" not in lab(r)]
for blk, mk in [(0, "o"), (1, "^")]:
    for s, ls in [("s0", "-"), ("s1", "--")]:
        rows = sorted([r for r in dose if r["block"] == blk and s in lab(r)],
                      key=lambda r: r["alpha_override"])
        b = [r for r in base if r["block"] == blk and s in lab(r)]
        xs = [3e-4] + [r["alpha_override"] for r in rows]  # 3e-4 ~ natif (repere)
        me = [b[0]["pr_final"]] + [r["pr_final"] for r in rows]
        pd_ = [np.nan] + [r["pr_dc"][-1] for r in rows]
        ax.plot(xs, me, mk + ls, color="tab:red", ms=5, lw=1.2,
                label=f"mesure b{blk}" if s == "s0" else None)
        ax.plot(xs, pd_, mk + ":", color="k", ms=4, alpha=0.7,
                label=f"formule DC b{blk}" if s == "s0" else None)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$ force au rejeu (point gauche : natif $\sim 10^{-4}$)")
ax.set_ylabel("pr final")
ax.set_title("(a) dose alpha : la formule DC predit sans parametre libre",
             fontsize=9)
ax.legend(fontsize=7)

ax = axes[1]
picks = [("mu_unit10k_s0", 0, "tab:red", "unit s0 b0 (collapse)"),
         ("mu_usq10k_s0", 0, "tab:blue", "usq s0 b0 (stable)"),
         ("mu_unit10k_dec_s1", 1, "tab:green", "decouple s1 b1")]
for name, blk, col, txt in picks:
    r = [x for x in stats if lab(x) == name and x["block"] == blk
         and x["mode"] == "cycle" and not x.get("alpha_override")][0]
    tr = trajs[(r["_f"], r["frozen"], r["mode"], None)]
    Ls = [(c["clip"] + 1) * 8 for c in tr]
    me = [c[f"pr_b{blk}"] for c in tr]
    dc = r["pr_dc"][:len(Ls)]
    ax.plot(Ls, me, "-", color=col, lw=1.6, label=f"{txt} — mesure")
    ax.plot([8 * (i + 1) for i in range(len(r["pr_dc"]))], r["pr_dc"], ":",
            color=col, lw=1.4, label=f"{txt} — rampe DC")
ax.set_xlabel("ecritures L (rejeu)")
ax.set_ylabel("pr")
ax.set_title("(b) deux canaux : chute precoce (fluctuations correlees)\n"
             "puis verrouillage par la derive coherente", fontsize=9)
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_collapse_dc.pdf", bbox_inches="tight")
print("fig_collapse_dc.pdf")
