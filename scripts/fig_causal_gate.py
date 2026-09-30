"""Fig. 5 — THE causal figure: "one learned gate carries the collapse".

Data (mechanism campaign, 31/08-01/09, all local jsonls):
  (a) transplant series: frozen unit weights, one usq component swapped
      in at replay; bar = mean r_eff of the write stream, block 0
      (block 1 as open marker). eta_head alone = full rescue, 3 seeds.
  (b) forced-eta dose at replay on native unit: monotone.
  (c) shadow measurement: both eta heads on the SAME replay inputs —
      unit's head outputs ~zero, usq's outputs ~0.99.

Output: figs/fig_causal_gate.pdf
"""
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st

st.apply()
WK = "/home/josep/projects/titans-mac/results/prlaw3"


def blocks(path):
    """{(frozen, block): record} for the per-block stat records."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "block" in r and "r_eff_mean" in r:
            out[(r["frozen"], r["block"])] = r
    return out


def re_of(path, block=0, seed=None):
    b = blocks(path)
    for (fr, blk), r in b.items():
        if blk == block and (seed is None or f"_s{seed}" in fr):
            return r["r_eff_mean"]
    return np.nan


fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(11.8, 3.7), gridspec_kw={"width_ratios": [1.5, 1, 0.9]})

# ---------------- (a) transplants ------------------------------------------
munit = blocks(f"{WK}/M_unit10k.jsonl")
musq = blocks(f"{WK}/M_usq10k.jsonl")
unit_b0 = [r["r_eff_mean"] for (fr, blk), r in munit.items() if blk == 0]
usq_b0 = [r["r_eff_mean"] for (fr, blk), r in musq.items() if blk == 0]

conds = [
    ("native\nunit", [re_of(f"{WK}/M_unit10k.jsonl", seed=0)], st.RED, ""),
    ("+proj.", [re_of(f"{WK}/SW_unitW_usqProj.jsonl")], st.GRAY, ""),
    ("+mem.\ninit", [re_of(f"{WK}/SW_unitW_usqMinit.jsonl")], st.GRAY, ""),
    ("+rule", [re_of(f"{WK}/SW_unitW_usqRule.jsonl")], st.GRAY, "xx"),
    ("+$\\theta$\nhead", [re_of(f"{WK}/SW_unitW_usqThetaH.jsonl")], st.GRAY, "xx"),
    ("+$\\alpha$\nhead", [re_of(f"{WK}/SW_unitW_usqAlphaH.jsonl")], st.GRAY, ""),
    ("+$\\eta$\nhead", [re_of(f"{WK}/SW_unitW_usqEtaH.jsonl"),
                        re_of(f"{WK}/SW_unitW_usqEtaH_s1.jsonl"),
                        re_of(f"{WK}/SW_unitW_usqEtaH_s2.jsonl")],
     st.BLUE, ""),
    ("+all\ngates", [re_of(f"{WK}/SW_unitW_usqGates.jsonl"),
                     re_of(f"{WK}/SW_unitW_usqGates_s1.jsonl"),
                     re_of(f"{WK}/SW_unitW_usqGates_s2.jsonl")],
     st.BLUE_LIGHT, ""),
]
ax.axhspan(min(usq_b0), max(usq_b0), color=st.BLUE, alpha=0.12, lw=0)
ax.text(1.0, max(usq_b0) * 1.15, "native usq band (5 seeds)", fontsize=7,
        color=st.BLUE, ha="center")
for x, (lab, vals, col, hat) in enumerate(conds):
    ax.bar(x, vals[0], width=0.62, color=col, alpha=0.9, hatch=hat,
           edgecolor="white" if not hat else st.DARK, lw=0.5, zorder=3)
    if len(vals) > 1:
        for v in vals:
            ax.plot(x, v, "o", color="k", ms=3.5, mfc="white", mew=1.1,
                    zorder=4)
# disengaged-writes guard notes (transplanted rule / theta: psi or theta
# out of calibration, write norms 0.05-0.27 — not interpretable as rescue)
for x in (3, 4):
    ax.text(x, 2.6, "writes disengage", ha="center", fontsize=6.2,
            color=st.DARK, rotation=90, va="bottom")
ax.set_yscale("log")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c[0] for c in conds], fontsize=7.2)
ax.set_ylabel(r"write-stream $r_{\mathrm{eff}}$ at replay (block 0)")
ax.set_ylim(0.9, 110)
ax.set_title(r"(a)  Transplants: the $\eta$ head ALONE rescues (3 seeds)",
             fontsize=9)

# ---------------- (b) forced-eta dose --------------------------------------
etas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
re0, re1 = [], []
for e in etas:
    b = blocks(f"{WK}/ETA_unit_e{e}.jsonl")
    re0.append([r["r_eff_mean"] for (f_, blk), r in b.items() if blk == 0][0])
    re1.append([r["r_eff_mean"] for (f_, blk), r in b.items() if blk == 1][0])
xpos = np.arange(len(etas))
bx.axhspan(min(usq_b0), max(usq_b0), color=st.BLUE, alpha=0.12, lw=0)
bx.text(0.03, 33, "native usq", fontsize=7, color=st.BLUE)
bx.plot(xpos, re0, "o-", color=st.DARK, label="block 0")
bx.plot(xpos, re1, "s--", color=st.DARK, alpha=0.55, label="block 1")
bx.plot(-0.85, np.mean([re_of(f"{WK}/M_unit10k.jsonl", 0, 0)]), "*",
        color=st.RED, ms=13,
        label=r"native unit ($\eta_{\mathrm{eff}} \approx 0$)")
bx.set_xticks(np.concatenate([[-0.85], xpos]))
bx.set_xticklabels(["native"] + [f"{e}" for e in etas], fontsize=7.5)
bx.set_xlabel(r"forced $\eta$ at replay")
bx.set_ylabel(r"write-stream $r_{\mathrm{eff}}$")
bx.set_yscale("log")
bx.set_ylim(0.9, 110)
bx.set_title(r"(b)  Forced-$\eta$ dose: monotone", fontsize=9)
bx.legend(fontsize=6.8, loc="upper left")

# ---------------- (c) shadow gates -----------------------------------------
rows = [json.loads(l) for l in open(f"{WK}/SHADOW_unitX_usqHeads.jsonl")]
ys, labels = [], []
y = 0
for blk in (0, 1):
    for who, col, name in [("base", st.RED, "unit head"),
                           ("donor", st.BLUE, "usq head")]:
        r = [x for x in rows if x["block"] == blk and x["who"] == who
             and x["gate"] == "eta"][0]
        cx.plot([max(r["p10"], 1e-10), r["p90"]], [y, y], "-", color=col,
                lw=2.4, alpha=0.85, solid_capstyle="round")
        cx.plot(max(r["med"], 1e-10), y, "o", color=col, ms=6, zorder=4)
        labels.append(f"block {blk} — {name}")
        ys.append(y)
        y -= 1
    y -= 0.6
cx.set_xscale("log")
cx.set_xlim(1e-10, 30)
cx.set_yticks(ys)
cx.set_yticklabels(labels, fontsize=7.5)
cx.axvline(1.0, color="k", lw=0.6, ls=":")
cx.set_xlabel("$\\eta$ on the same inputs\n(p10 - median - p90)")
cx.set_title("(c)  The collapsed head\noutputs zero", fontsize=9)
cx.grid(axis="x", alpha=0.22)

fig.tight_layout()
st.save(fig, "fig_causal_gate")
