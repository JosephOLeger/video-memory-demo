"""Fig. 6 — "No inference-time remedy: the read path is co-adapted".

Gap-0 mem_effect (loss with memory off minus loss with memory on;
positive = the memory helps) from the retention harness, unit seed 0,
under every replay-time intervention that rescues the write GEOMETRY.
Data: scratchpad wk/ret_*.json (P-GAIN-OV / P-GAIN-SW / P-GAIN-THCOMP
verdicts, 01/09).

Output: figs/fig_no_remedy.pdf
"""
import glob
import json

import matplotlib.pyplot as plt

import figstyle as st

st.apply()
WK = "/home/josep/projects/titans-mac/results/prlaw3"


def g0(fname):
    r = json.load(open(f"{WK}/{fname}"))[0]
    return [c for c in r["curve"] if c["gap_blocks"] == 0][0]["mem_effect"]


bars = [
    ("native usq\n(healthy arm)", g0("ret_native_usq_s0.json"), st.BLUE, ""),
    ("native unit\n(collapsed arm)", g0("ret_native_unit_s0.json"), st.RED, ""),
    ("unit,\nforced $\\eta$=0.98", g0("ret_ETAov098_unit_s0.json"),
     st.RED_LIGHT, "//"),
    ("unit, grafted\n$\\eta$ head", g0("ret_SWetaH_unit_s0.json"),
     st.RED_LIGHT, "//"),
    ("unit, $\\eta$=0.98\n$\\theta$=0.03", g0("ret_ETAov098_th0.03_unit_s0.json"),
     st.RED_LIGHT, "//"),
    ("unit, $\\eta$=0.98\n$\\theta$=0.08", g0("ret_ETAov098_th0.08_unit_s0.json"),
     st.RED_LIGHT, "//"),
]

fig, ax = plt.subplots(figsize=(6.6, 3.5))
for x, (lab, v, col, hat) in enumerate(bars):
    ax.bar(x, v, width=0.6, color=col, hatch=hat,
           edgecolor="white" if not hat else st.RED, lw=0.6, zorder=3)
    ax.text(x, v + (0.035 if v > 0 else -0.035), f"{v:+.2f}", ha="center",
            va="bottom" if v > 0 else "top", fontsize=7.5, color=st.DARK)
ax.axhline(0, color="k", lw=0.9)
ax.set_xticks(range(len(bars)))
ax.set_xticklabels([b[0] for b in bars], fontsize=7.3)
ax.set_ylabel("mem_effect at gap 0\n(positive = memory helps the read-out)")
ax.set_ylim(-1.0, 0.62)
ax.text(3.5, 0.44, "replay-time interventions rescue the write\n"
        "geometry (pr, $r_{\\mathrm{eff}}$) — the FUNCTION gets worse:\n"
        "the read path is co-adapted to writes without momentum",
        fontsize=7.6, ha="center", color=st.DARK)
ax.text(-0.35, 0.52, "memory helps", fontsize=7, color=st.GRAY, style="italic")
ax.text(-0.35, -0.95, "memory hurts", fontsize=7, color=st.GRAY, style="italic")
fig.tight_layout()
st.save(fig, "fig_no_remedy")
