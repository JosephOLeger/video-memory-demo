"""Deux figures a partir des verdicts consignes.

fig_r_dose.pdf : (a) la serie de doses D2 (maillons NS dans le graphe vs
gradients non finis au pas 0, + le regime de l'issue #29 en point rapporte)
et (b) l'echelle R (pas de mort par realisation et par barreau).

fig_homotopy.pdf : l'homotopie de la famille -- gain de retention (scoring
conjoint, 4000 pas) le long de l'axe c, de Titans (c=inf, non borne) a
ATLAS (c->0, unit), avec le piege du c annote.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs"

# ---------- figure dose-reponse R ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

ax = axes[0]
links = [4, 64, 128]
frac0 = [0.0, 0.0, 248 / 340]
ax.plot(links, frac0, "o-", color="tab:red", ms=7,
        label="mesure (gradcheck, pas 0)")
ax.plot([32], [np.nan], "s", color="tab:orange")
ax.annotate("issue #29 (SEQ 4096, batch 128)\nNaN rapportes, remede ecretage",
            xy=(32, 0.35), fontsize=7, ha="center", color="tab:orange")
ax.axvline(32, color="tab:orange", ls=":", lw=1)
for x, txt in [(4, "R0 : survit 20k pas"), (64, "ctrl256 : 0/340"),
               (128, "R1 : 248/340,\nNaN pas 2")]:
    ax.annotate(txt, xy=(x, frac0[links.index(x)] if x in links else 0),
                xytext=(x, 0.62 if x == 128 else 0.08), fontsize=7,
                ha="center")
ax.set_xscale("log", base=2)
ax.set_xticks([4, 32, 64, 128]); ax.set_xticklabels(["4", "32", "64", "128"])
ax.set_xlabel("maillons Newton-Schulz dans le graphe backward")
ax.set_ylabel("fraction de gradients non finis (pas 0)")
ax.set_ylim(-0.05, 1.0)
ax.set_title("(a) D2 : dose par longueur de chaine")
ax.legend(fontsize=7, loc="upper left")

ax = axes[1]
rungs = {"R1\n(spec-norm,\nboucle fermee)": [0, 2],
         "R2\n(+amplitude\nbrute)": [62, 1001],
         "R3\n(+MLP nu)": [274, 1001]}
for i, (name, deaths) in enumerate(rungs.items()):
    for d in deaths:
        if d > 1000:
            ax.plot(i, 1000, "^", color="tab:green", ms=9)
        else:
            ax.plot(i, max(d, 0.7), "x", color="tab:red", ms=10, mew=2.5)
ax.set_yscale("log")
ax.set_xticks(range(len(rungs))); ax.set_xticklabels(rungs.keys(), fontsize=7)
ax.set_ylabel("pas de mort (x) / survie 1000 pas (^)")
ax.set_title("(b) echelle R : toutes les morts backward-nees")
ax.set_ylim(0.5, 1500)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_r_dose.pdf", bbox_inches="tight")
print("saved fig_r_dose.pdf")

# ---------- figure homotopie ----------
fig, ax = plt.subplots(figsize=(6.2, 3.8))
# (c effectif, gain 4k, label, borne?)
# NOTE 27/08 : le point c=inf (bras non borne) est retire — il venait de
# l'hybride maison retire (archive_rescale_2026-08-27.md). L'extremite
# non bornee sera re-mesuree sur l'echelle Titans-litterale.
pts = [
    (1.0, 0.206, "clip c=1\n(cap jamais actif)", False),
    (0.07, 0.318, "clip c=0.07", True),
    (0.03, 0.330, "softclip c=0.03", True),
    (4.5e-4, 0.001, "softclip c=mediane\n(le piege du c)", True),
    (1e-5, 0.086, "unit (c->0)", True),
]
for c, g, lab, borne in pts:
    ax.plot(c, g, "o" if borne else "s", ms=9,
            color="tab:blue" if borne else "tab:red")
    dy = -0.045 if g > 0.05 else 0.03
    ax.annotate(lab, xy=(c, g), xytext=(c, g + dy), fontsize=7, ha="center")
ax.plot([1e-5], [0.426], "o", ms=9, mfc="none", color="tab:blue")
ax.annotate("unit, 10k pas", xy=(1e-5, 0.426), xytext=(1e-5, 0.45),
            fontsize=7, ha="center", color="tab:blue")
ax.axhline(0.206, color="tab:red", lw=0.6, ls="--", alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("constante de compression $c$ (echelle log ; homotopie ATLAS $\\leftarrow$ Titans)")
ax.set_ylabel("gain de retention (dist. 0, 4000 pas)")
ax.set_title("L'homotopie : au bon $c$, la borne bat le non-borne")
ax.set_ylim(-0.03, 0.52)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_homotopy.pdf", bbox_inches="tight")
print("saved fig_homotopy.pdf")
