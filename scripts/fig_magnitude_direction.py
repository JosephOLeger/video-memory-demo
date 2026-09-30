"""Fig. 3 — "Magnitude, not direction".

Consigned numbers (results/predictions_collapse_2026-08-28.md):
- Titans-literal (raw direction, no magnitude bound): 8/8 runs diverge
  (titans_lit ladder), no recall to report.
- rawdir_usq (raw direction S/||S||_F, psi_c magnitude bound): 2/2 seeds
  train 10k steps with zero NaN, retention gains 0.407 / 0.394
  (P-2x2-1/-2 verdicts, 31/08).
- usq (Newton-Schulz direction, same psi_c bound): stable, retention
  gain 0.533 at 10k (P-2x2-3) — orthogonalisation buys ~25% quality,
  not stability.

Output: figs/fig_magnitude_direction.pdf
"""
import matplotlib.pyplot as plt

import figstyle as st

st.apply()

fig, ax = plt.subplots(figsize=(6.0, 3.6))

bars = [
    ("published rule\nraw direction\nno magnitude bound", 0.0, st.RED, []),
    ("magnitude bound $\\psi_c$\nraw direction\n(no orthogonalisation)",
     (0.407 + 0.394) / 2, st.GREEN, [0.407, 0.394]),
    ("magnitude bound $\\psi_c$\nNewton--Schulz direction\n(full rule)",
     0.533, st.BLUE, [0.533]),
]
xs = range(len(bars))
for x, (lab, h, col, seeds) in zip(xs, bars):
    ax.bar(x, h, width=0.55, color=col, alpha=0.85, zorder=3)
    for s in seeds:
        ax.plot(x, s, "o", color="k", ms=4, mfc="white", mew=1.2, zorder=4)

# the divergent cell
ax.bar(0, 0.62, width=0.55, color="none", edgecolor=st.RED, hatch="///",
       lw=1.0, zorder=2)
ax.text(0, 0.315, "DIVERGES\n8 of 8 runs\n(no recall\nto measure)",
        ha="center", va="center", fontsize=8, color=st.RED,
        fontweight="bold")

# +25% bracket between the two stable arms
y0, y1 = 0.4005, 0.533
ax.annotate("", xy=(2, y1 + 0.025), xytext=(1, y1 + 0.025),
            arrowprops=dict(arrowstyle="-", color=st.DARK, lw=1.0))
ax.plot([1, 1], [y0 + 0.012, y1 + 0.025], color=st.DARK, lw=1.0)
ax.plot([2, 2], [y1 + 0.005, y1 + 0.025], color=st.DARK, lw=1.0)
ax.text(1.5, y1 + 0.04, "+25% recall quality\n(what the direction buys)",
        ha="center", fontsize=8, color=st.DARK)

ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("retention gain (10k steps, joint scoring)")
ax.set_ylim(0, 0.68)
fig.tight_layout()
st.save(fig, "fig_magnitude_direction")
