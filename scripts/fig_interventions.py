"""Fig. 7 of the collapse section — the causal interventions.

All replay-time chimeras on frozen artifacts (block 0, unit = the
collapsed arm, usq = the healthy psi_c twin). Verdicts consigned in
predictions_collapse_2026-08-28.md (mecanisme day 31/08, wave 2
01/09, P-REPL-ETAH, P-DOSE-SEEDS, P-GAIN-SEEDS 09/09).

(a) Component transplants: replay unit's weights with ONE component
    taken from its healthy twin. Rule, projections and memory init
    do not restore write diversity; the gate heads do — and the
    eta head ALONE reproduces the full rescue, 3/3 seeds. Open
    markers: disengaged writes (wn < 0.3, guard) — the probe is
    unreadable there, not evidence of absence.
(b) Dose-response: unit replayed with eta forced to a constant.
    Write diversity rises monotonically with the momentum window
    1/(1-eta), 3 seeds. (Native unit's REALISED eta is ~1e-6 —
    the shadow measurement — i.e. the far left of this curve.)
(c) The catch: the same interventions that restore the geometry
    destroy the function (gap-0 mem_effect, within-arm) — the
    readout is co-adapted to momentum-free writes. The lever is
    preventive (train with the bound), not curative (replay).

Output: figs/fig_interventions.pdf
"""
import json
import math

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

fs.apply()
P2 = "/home/josep/projects/titans-mac/results/prlaw2"
P3 = "/home/josep/projects/titans-mac/results/prlaw3"
S5 = "/home/josep/projects/titans-mac/results/s5"


def b0(path, suffix=None):
    """(r_eff_mean, wn, pr_final) of block-0 rows, per frozen artifact."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if r.get("block") != 0 or "r_eff_mean" not in r:
            continue
        if suffix and not r["frozen"].endswith(suffix):
            continue
        out.append((r["r_eff_mean"], r["write_norm_mean"], r["pr_final"]))
    return out


def gap0(path):
    return json.load(open(path))[0]["curve"][0]["mem_effect"]


fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.0, 3.9),
                             gridspec_kw={"width_ratios": [1.3, 1]})

# ---------------- (a) component transplants, condensed ---------------------
# One gray row pools the five failed single-component swaps (rule,
# projections, memory init, alpha head, theta head); open markers are
# guard-fail cells (disengaged writes, wn < 0.3 — unreadable, per the
# consigned garde-fou). Full-gates swap = same rescue as eta alone
# (caption). The message is the contrast native / other / eta.
OTHER = ["SW_unitW_usqRule", "SW_unitW_usqProj", "SW_unitW_usqMinit",
         "SW_unitW_usqAlphaH", "SW_unitW_usqThetaH"]
rows = [
    ("unit native\n(5 seeds)", fs.RED,
     [b0(f"{P2}/L_mu_unit10k.jsonl", f"mu_unit10k_s{s}.pt")[0]
      for s in range(5)]),
    ("+ any other single\ncomponent of the twin", fs.GRAY,
     sum((b0(f"{P3}/{f}.jsonl") for f in OTHER), [])),
    ("+ the twin's $\\eta$ head\nALONE (3 seeds)", fs.BLUE,
     b0(f"{P3}/SW_unitW_usqEtaH.jsonl")
     + b0(f"{P3}/SW_unitW_usqEtaH_s1.jsonl")
     + b0(f"{P3}/SW_unitW_usqEtaH_s2.jsonl")),
]
usq_nat = [b0(f"{P2}/L_mu_usq10k.jsonl", f"mu_usq10k_s{s}.pt")[0][0]
           for s in range(5)]
ax.axvspan(min(usq_nat), max(usq_nat), color=fs.BLUE, alpha=0.10, lw=0)
ax.text(math.sqrt(min(usq_nat) * max(usq_nat)), 2.52,
        "healthy twin,\nnative range", fontsize=8, color=fs.BLUE,
        ha="center", va="top")
for i, (lab, col, vals) in enumerate(rows):
    y = len(rows) - 1 - i
    for re_, wn, _ in vals:
        if wn < 0.3:  # guard: disengaged writes, probe unreadable
            ax.plot(re_, y, "o", mfc="none", color=col, ms=7, mew=1.5)
        else:
            ax.plot(re_, y, "o", color=col, ms=7, alpha=0.9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.6)
ax.set_ylim(-0.55, 2.62)
ax.set_xscale("log")
ax.set_xlim(1, 90)
ax.set_xlabel(r"write diversity $r_{\mathrm{eff}}$ (block 0, log)")
ax.set_title("(a)  the $\\eta$ head alone carries the collapse",
             fontsize=9.6)

# ---------------- (b) dose-response in eta ---------------------------------
DOSES_S0 = [("0.0", f"{P3}/ETA_unit_e0.0.jsonl"),
            ("0.5", f"{P3}/ETA_unit_e0.5.jsonl"),
            ("0.8", f"{P3}/ETA_unit_e0.8.jsonl"),
            ("0.9", f"{P3}/ETA_unit_e0.9.jsonl"),
            ("0.95", f"{P3}/ETA_unit_e0.95.jsonl"),
            ("0.98", f"{P3}/ETA_unit_e0.98.jsonl")]
curves = {0: [(float(e), b0(f)[0][0]) for e, f in DOSES_S0]}
for s in (1, 2):
    curves[s] = [(float(e), b0(f"{S5}/ETA_unit_e{e}_s{s}.jsonl")[0][0])
                 for e in ("0.0", "0.5", "0.9", "0.98")]
for s in (0, 1, 2):
    pts = sorted(curves[s])
    bx.plot([1.0 / (1.0 - e) for e, _ in pts], [v for _, v in pts],
            "-", marker="o", ms=4.5, lw=1.2, color=fs.BLUE, alpha=0.75,
            label="forced $\\eta$, 3 seeds" if s == 0 else None)
nat_re = [b0(f"{P2}/L_mu_unit10k.jsonl", f"mu_unit10k_s{s}.pt")[0][0]
          for s in range(3)]
bx.plot([1.02] * 3, nat_re, "*", color=fs.RED, ms=10, zorder=5,
        label="native (measured $\\eta \\approx 10^{-6}$)")
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlabel(r"momentum window $1/(1-\eta)$ (log)")
bx.set_ylabel(r"write diversity $r_{\mathrm{eff}}$ (log)")
bx.set_title("(b)  diversity is dose-monotone in $\\eta$", fontsize=9.6)
bx.legend(fontsize=9.2, loc="upper left")

fig.tight_layout()
fs.save(fig, "fig_interventions")
