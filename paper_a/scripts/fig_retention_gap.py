"""Appendix figure — single-item retention at a distance (review #2, pt 1).

Paired-contrast retention at gap g (worldmodel testbed, absolute rule,
10k-step checkpoints, 5 seeds per arm): store video V, insert g blocks
of distinct distractors, probe the final block with V again (repeat) or
an unrelated U (cross).  retention_gain = l1_cross - l1_repeat.

(a) The collapsed arm (unit) OUTSCORES the healthy arm (usq) at gaps
    1..8: a rank-collapsed memory concentrates its energy on one trace
    and passes a one-item test; a distributed memory spends capacity on
    the distractors too and its per-item trace fades faster.
(b) The same states' cos_ret (mean pairwise cosine of retrievals for
    DISTINCT probe queries): unit reads out nearly the same vector for
    every query, usq differentiates.  What survives the gap in (a) is
    one direction, not a memory.

Data: results/box/retention_mu_{unit,usq}10k_absrule.json.
Output: figs/fig_retention_gap.pdf
"""
import json
import statistics as st

import matplotlib.pyplot as plt

import figstyle as fs

fs.apply()
BOX = "/home/josep/projects/titans-mac/results/box"
ARMS = [("retention_mu_unit10k_absrule.json", fs.RED,
         "unbounded momentum (unit, collapsed)"),
        ("retention_mu_usq10k_absrule.json", fs.BLUE,
         r"$\psi_c$ (usq, healthy)")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.2))

for fname, col, lab in ARMS:
    runs = json.load(open(f"{BOX}/{fname}"))
    gaps = [r["gap_blocks"] for r in runs[0]["curve"]]
    xs = list(range(len(gaps)))
    for key, axis in (("retention_gain", ax), ("cos_ret", bx)):
        per_seed = [[r[key] for r in run["curve"]] for run in runs]
        for ys in per_seed:
            axis.plot(xs, ys, "-", color=col, lw=0.8, alpha=0.35)
        med = [st.median(ys[i] for ys in per_seed) for i in range(len(xs))]
        axis.plot(xs, med, "-o", color=col, lw=2.2, ms=4,
                  label=lab if axis is ax else None)
    for axis in (ax, bx):
        axis.set_xticks(xs)
        axis.set_xticklabels([str(g) for g in gaps])
        axis.set_xlabel("gap (distractor blocks)")

ax.axhline(0.0, color=fs.GRAY, lw=0.8, ls="--")
ax.set_ylabel(r"retention gain  $\ell^{\mathrm{cross}}-\ell^{\mathrm{repeat}}$")
ax.set_title("(a)  the collapsed arm outscores at a distance", fontsize=9.6)
ax.legend(fontsize=9.4, loc="upper right")

bx.set_ylabel("cos_ret (distinct queries, same state)")
bx.set_ylim(0, 1)
bx.set_title("(b)  ...by answering every query the same way", fontsize=9.6)

fig.tight_layout()
fs.save(fig, "fig_retention_gap")

# medians for the appendix text
for fname, _, lab in ARMS:
    runs = json.load(open(f"{BOX}/{fname}"))
    gaps = [r["gap_blocks"] for r in runs[0]["curve"]]
    for key in ("retention_gain", "cos_ret", "pr"):
        med = [round(st.median(run["curve"][i][key] for run in runs), 3)
               for i in range(len(gaps))]
        print(f"{lab:40s} {key:15s} {med}")
