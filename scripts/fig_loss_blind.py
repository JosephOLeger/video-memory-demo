"""Figure "la loss est aveugle a D1" — le barreau frontiere fine_t008_s0.

Par pas d'entrainement : w1_max et S_max (log) explosent par a-coups
jusqu'au NaN au pas 7, pendant que la loss externe reste plate (~1.0)
Y COMPRIS au pas fatal — la lecture l2-normalisee cache ||W|| a la perte.
Donnees : results/d2interne/fine_t008_s0.jsonl (rapatrie).
Sortie : figs/fig_loss_blind.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "d2interne")
steps, w1, smax, loss = [], [], [], []
for line in open(f"{S}/fine_t008_s0.jsonl"):
    r = json.loads(line)
    if "step" not in r or r.get("p_bad", 1) != 0:
        continue
    steps.append(r["step"])
    w1.append(r["w1_max"])
    smax.append(r["s_max"])
    loss.append(r["loss"])

fig, ax = plt.subplots(figsize=(6.6, 3.8))
w1p = [v if v == v else 1e18 for v in w1]
sp = [v if v == v else 1e18 for v in smax]
ax.plot(steps, w1p, "o-", color="tab:red", label=r"$\Vert W_1\Vert_{\max}$ (etat interne)")
ax.plot(steps, sp, "s--", color="tab:orange", alpha=0.7, label=r"$\Vert S\Vert_{\max}$ (momentum)")
ax.set_yscale("log")
ax.axhline(1e18, color="k", lw=0)
ax.annotate("NaN", xy=(steps[-1], 1e18), xytext=(steps[-1] - 0.6, 2e16),
            fontsize=9, color="tab:red")
ax.set_xlabel("pas d'entrainement (theta-scale 0.08, graine 0)")
ax.set_ylabel("normes d'etat (log)")
ax2 = ax.twinx()
ax2.plot(steps, loss, "^-", color="tab:blue", label="loss externe")
ax2.set_ylabel("loss externe", color="tab:blue")
ax2.set_ylim(0, 2)
ax2.tick_params(axis="y", labelcolor="tab:blue")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")
ax.set_title("La loss est aveugle a D1 : etat x$10^{15}$, loss plate (lecture normalisee)")
fig.tight_layout()
out = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs/fig_loss_blind.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("loss au pas fatal:", loss[-1], "| w1 trajectoire:", [f"{v:.3g}" for v in w1])
