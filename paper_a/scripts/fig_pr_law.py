"""Figure de la loi pr : mesure (sim_pr_law.json) contre prediction.

    1/pr = (alpha/r_eff) . Phi(rho_k, rho_v) + 1/d1 + 1/g
    Phi = 1 + 2 rho_k^2/(1-rho_k^2) + 2 rho_v^2/(1-rho_v^2)

(a) parite pr mesure vs pr predit, toutes configs ; (b) 1/pr contre le
terme de compte alpha.Phi/r : la loi predit une droite de pente 1 et
d'ordonnee 1/d_h — la version rang de la loi de portee.
"""
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs"
D1 = D2 = 128

rows = json.load(open("/home/josep/projects/titans-mac/results/sim_pr_law.json"))


def phi(rk, rv):
    return 1 + 2 * rk**2 / (1 - rk**2) + 2 * rv**2 / (1 - rv**2)


def pred(r):
    g = r["g"] or D2
    count = (r["alpha"] / r["r"]) * phi(r["rho_k"], r["rho_v"])
    return 1.0 / (count + 1.0 / D1 + 1.0 / g)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

ax = axes[0]
groups = [("iid", lambda r: r["rho_k"] == 0 and r["rho_v"] == 0 and not r["g"],
           "tab:blue", "o"),
          (r"cles AR(1) $\rho_k>0$", lambda r: r["rho_k"] > 0, "tab:red", "^"),
          (r"valeurs AR(1) $\rho_v>0$", lambda r: r["rho_v"] > 0,
           "tab:orange", "v"),
          (r"geometrie $g<d$", lambda r: bool(r["g"]), "tab:green", "s")]
for name, sel, col, mk in groups:
    xs = [pred(r) for r in rows if sel(r)]
    ys = [r["pr"] for r in rows if sel(r)]
    ax.plot(xs, ys, mk, color=col, ms=5, alpha=0.8, label=name)
lim = [0, 70]
ax.plot(lim, lim, "k:", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("pr predit par la loi")
ax.set_ylabel("pr mesure (simulation)")
err = [abs(r["pr"] - pred(r)) / pred(r) for r in rows]
ax.set_title(f"(a) parite --- erreur mediane "
             f"{100*float(np.median(err)):.0f}%, max "
             f"{100*max(err):.0f}%", fontsize=9)
ax.legend(fontsize=7, loc="upper left")

ax = axes[1]
for name, sel, col, mk in groups[:2]:
    xs = [(r["alpha"] / r["r"]) * phi(r["rho_k"], r["rho_v"])
          for r in rows if sel(r)]
    ys = [1.0 / r["pr"] for r in rows if sel(r)]
    ax.plot(xs, ys, mk, color=col, ms=5, alpha=0.8, label=name)
xg = np.logspace(-5, 0, 50)
ax.plot(xg, xg + 2.0 / D1, "k--", lw=1,
        label=r"loi : $1/\mathrm{pr} = x + 2/d$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"terme de compte $x = \alpha\,\Phi(\rho)/r_{\mathrm{eff}}$")
ax.set_ylabel(r"$1/\mathrm{pr}$")
ax.set_title("(b) la droite harmonique --- pendant rang de la loi de portee",
             fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pr_law.pdf", bbox_inches="tight")
print("saved fig_pr_law.pdf")

for r in rows:
    r["pred"] = pred(r)
worst = sorted(rows, key=lambda r: -abs(r["pr"] - r["pred"]) / r["pred"])[:6]
for r in worst:
    print(f"ecart {100*abs(r['pr']-r['pred'])/r['pred']:5.1f}%  "
          f"a={r['alpha']:g} rk={r['rho_k']} rv={r['rho_v']} r={r['r']} "
          f"g={r['g']} : mesure {r['pr']:.1f} predit {r['pred']:.1f}")
