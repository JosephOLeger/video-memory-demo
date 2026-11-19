"""Fig. 7 — external stack (lucidrains titans-pytorch, enwik8): the 2x2
{Newton-Schulz on/off} x {magnitude rule psi on/off}, and the
loss-health decorrelation.

Each cell: state pr along replay (4096 tokens, 1152 writes), 12 streams
per seed (4 sequences x 3 memory layers), sampled every ~28 writes.
Validation loss (nats/char, median of last 10) printed per cell — the
best loss of the whole campaign (1.37) sits on memories that reach the
exact zero matrix in-sequence.

Data: results/prlaw3/{R_lad_final,MS_prlaw_s*,R2b2_psic_trained,
MS_psic_s*,ABL_sn,SNPSIC}.jsonl (P-MULTI / P-ABLRANK / P-SNPSIC
verdicts, 02-03/09).

Output: figs/fig_external_2x2.pdf
"""
import json
import math

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st

st.apply()
P = "/home/josep/projects/titans-mac/results/prlaw3"
STRIDE = 28  # pr_traj_series samples the state every len(series)//40 writes


def trajs(fname):
    out = []
    for line in open(f"{P}/{fname}.jsonl"):
        r = json.loads(line)
        ts = r["pr_traj_series"]
        xs = [i * STRIDE for i in range(len(ts))]
        fin = [(x, v) for x, v in zip(xs, ts)
               if not (isinstance(v, float) and math.isnan(v))]
        extinct = len(fin) < len(ts)
        out.append(([x for x, _ in fin], [v for _, v in fin], extinct))
    return out


cells = [
    (0, 0, "NS direction ON  /  magnitude rule $\\psi$ OFF\n(baseline, as shipped)",
     [("R_lad_final", "-"), ("MS_prlaw_s1", "--"), ("MS_prlaw_s2", ":")],
     st.RED, "val 2.36 - 2.71", "rank collapse: pr $\\to$ 1, writes alive"),
    (0, 1, "NS ON  /  $\\psi$ ON\n(trained with the bound)",
     [("R2b2_psic_trained", "-"), ("MS_psic_s1", "--"), ("MS_psic_s2", ":")],
     st.BLUE, "val 1.46 - 1.61", "alive: pr 5-13, 3/3 seeds"),
    (1, 0, "NS OFF  /  $\\psi$ OFF\n(spectral-norm removed)",
     [("ABL_sn", "-")],
     st.RED, "val 1.37  (best of campaign)",
     "extinction: writes die out,\nstate = the exact zero matrix"),
    (1, 1, "NS OFF  /  $\\psi$ ON\n(softclip magnitude, raw direction)",
     [("SNPSIC", "-")],
     st.GREEN, "val 1.37", "extinction: same signature"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), sharex=True, sharey=True)
for row, colu, title, files, col, val, verdict in cells:
    ax = axes[row][colu]
    n_ext, n_tot = 0, 0
    for fname, ls in files:
        for xs, ys, extinct in trajs(fname):
            n_tot += 1
            ax.plot(xs, ys, ls, color=col, lw=0.9, alpha=0.55)
            if extinct:
                n_ext += 1
                ax.plot(xs[-1], max(ys[-1], 1.0), "x", color=st.DARK, ms=5,
                        mew=1.4, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(0.85, 130)
    ax.set_title(title, fontsize=8, fontweight="normal", loc="center")
    ax.text(0.97, 0.90, val, transform=ax.transAxes, ha="right", fontsize=8,
            fontweight="bold", color=st.DARK)
    ax.text(0.97, 0.74, verdict, transform=ax.transAxes, ha="right",
            fontsize=7, color=col)
    if n_ext:
        ax.text(0.97, 0.58, f"{n_ext}/{n_tot} streams end at 0",
                transform=ax.transAxes, ha="right", fontsize=7, color=st.DARK)
    if row == 1:
        ax.set_xlabel("write index along the sequence")
    if colu == 0:
        ax.set_ylabel("state pr (log)")
    if row == 0 and colu == 0:
        for ls, sd in [("-", "seed 0"), ("--", "seed 1"), (":", "seed 2")]:
            ax.plot([], [], ls, color=st.DARK, lw=1, label=sd)
        ax.legend(loc="lower left", fontsize=6.5, ncol=3,
                  handlelength=1.6, columnspacing=0.9)

fig.tight_layout()
st.save(fig, "fig_external_2x2")
