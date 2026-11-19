"""Fig. for section 4 — the state-size bracket, verified (P-ALPHAEFF).

Ten (plateau, b, alpha) triples: the 4 trained bounded arms of Fig. 3
(realized alpha from the in-replay shadow measurement, G_*.jsonl) and
6 forced-alpha replays (D_a*, alpha_override exact, both seeds). Per
measurement, the per-call forgetting is alpha_C = 1 - (1-alpha)^C with
C = 8 tokens per call; the write bound b is the measured per-call
write norm of the same replay (block 0). Lemma bracket:
floor  b / sqrt(alpha_C (2 - alpha_C))   (decorrelated RMS)
bound  b / alpha_C                       (aligned worst case).

(a) per measurement: the bracket as a vertical segment, the measured
    stationary plateau (median of the last 20 w1n_b0) as a dot ---
    every dot inside its bracket.
(b) normalized: plateau/b against alpha_C, log-log, with the two laws
    as guides --- forced-alpha points hug the RMS floor (pooled
    exponent ~ -0.44 vs -1/2), trained arms sit x4 above it, pushed
    toward the aligned bound by write correlation.

Data: results/alpha_eff/G_*.jsonl, results/prlaw2/T2_usq10k_s*.jsonl,
results/prlaw3/N_rawdir_usq.jsonl, results/prlaw2/D_a*_unit10k.jsonl
Verdict: P-ALPHAEFF in results/predictions_collapse_2026-08-28.md
Output: figs/fig_state_law.pdf
"""
import json
import math
import statistics

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st

st.apply()
BASE = "/home/josep/projects/titans-mac/results"
C = 8  # tokens per optimizer call in the replay probe


def rows(path):
    return [json.loads(l) for l in open(path)]


def plateau_and_b(path, suffix):
    plat = b = None
    for r in rows(path):
        if not r.get("frozen", "").endswith(suffix):
            continue
        if "traj" in r:
            plat = statistics.median([c["w1n_b0"] for c in r["traj"][-20:]])
        if r.get("block") == 0 and "write_norm_mean" in r:
            b = r["write_norm_mean"]
    return plat, b


def shadow_alpha(path):
    for r in rows(path):
        if r["block"] == 0 and r["who"] == "base" and r["gate"] == "alpha":
            return r["mean"]


# ---------------- assemble the ten measurements ----------------------------
pts = []  # (label, group, alpha_tok, b, plateau)
trained = [
    ("NS s0", f"{BASE}/alpha_eff/G_mu_usq10k_s0_at10000.jsonl",
     f"{BASE}/prlaw2/T2_usq10k_s0.jsonl", "mu_usq10k_s0_at10000.pt"),
    ("NS s1", f"{BASE}/alpha_eff/G_mu_usq10k_s1_at10000.jsonl",
     f"{BASE}/prlaw2/T2_usq10k_s1.jsonl", "mu_usq10k_s1_at10000.pt"),
    ("raw s0", f"{BASE}/alpha_eff/G_rawdir_usq_s0.jsonl",
     f"{BASE}/prlaw3/N_rawdir_usq.jsonl", "rawdir_usq_s0.pt"),
    ("raw s1", f"{BASE}/alpha_eff/G_rawdir_usq_s1.jsonl",
     f"{BASE}/prlaw3/N_rawdir_usq.jsonl", "rawdir_usq_s1.pt"),
]
for lab, gpath, ppath, suf in trained:
    plat, b = plateau_and_b(ppath, suf)
    pts.append((lab, "trained", shadow_alpha(gpath), b, plat))

for a in ("0.003", "0.01", "0.03"):
    for s in (0, 1):
        plat, b = plateau_and_b(f"{BASE}/prlaw2/D_a{a}_unit10k.jsonl",
                                f"mu_unit10k_s{s}.pt")
        pts.append((rf"$\alpha$={a}" if s == 0 else None, "forced",
                    float(a), b, plat))


def acall(al):
    return 1.0 - (1.0 - al) ** C


def floor_of(b, ac):
    return b / math.sqrt(ac * (2.0 - ac))


fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.8, 3.7),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

# ---------------- (a) the bracket, measurement by measurement --------------
xs_pos = [0, 1, 2, 3, 4.85, 5.15, 5.85, 6.15, 6.85, 7.15]
for x, (lab, grp, al, b, plat) in zip(xs_pos, pts):
    ac = acall(al)
    lo, hi = floor_of(b, ac), b / ac
    ax.plot([x, x], [lo, hi], "-", color=st.GRAY, lw=4.5, alpha=0.45,
            solid_capstyle="butt", zorder=2)
    col = st.BLUE if grp == "trained" else st.GREEN
    ax.plot(x, plat, "o", color=col, ms=7, zorder=4)
ax.plot([], [], "-", color=st.GRAY, lw=4.5, alpha=0.45,
        label=r"bracket $[\,b/\sqrt{\alpha_C(2-\alpha_C)},\ b/\alpha_C\,]$")
ax.plot([], [], "o", color=st.BLUE, ms=7, label="trained arms (Fig. 3)")
ax.plot([], [], "o", color=st.GREEN, ms=7, label=r"forced $\alpha$ (replay)")
ax.set_yscale("log")
ax.set_xticks([0, 1, 2, 3, 5, 6, 7])
ax.set_xticklabels(["NS\ns0", "NS\ns1", "raw\ns0", "raw\ns1",
                    ".003", ".01", ".03"], fontsize=8.3)
ax.set_xlabel(r"trained arms $\quad|\quad$ forced $\alpha$ (two seeds)")
ax.set_ylabel(r"stationary state norm $\Vert W_1\Vert$ (log)")
ax.set_ylim(0.6, 700)
ax.set_title("(a)  every stationary state falls inside the bracket",
             fontsize=9.6)
ax.legend(fontsize=9.2, loc="upper right")

# ---------------- (b) normalized: plateau/b against alpha_C ----------------
gx = np.geomspace(0.008, 0.32, 60)
bx.plot(gx, 1.0 / gx, "--", color=st.RED, lw=1.2,
        label=r"aligned bound $1/\alpha_C$")
bx.plot(gx, 1.0 / np.sqrt(gx * (2.0 - gx)), "--", color=st.DARK, lw=1.2,
        label=r"decorrelated RMS $1/\sqrt{\alpha_C(2-\alpha_C)}$")
seen = set()
for lab, grp, al, b, plat in pts:
    ac = acall(al)
    col = st.BLUE if grp == "trained" else st.GREEN
    mk = "o" if grp == "trained" else "s"
    l = None
    if grp not in seen:
        seen.add(grp)
        l = ("trained arms" if grp == "trained" else r"forced $\alpha$")
    bx.plot(ac, plat / b, mk, color=col, ms=6.5, label=l, zorder=4)
bx.annotate(r"trained: $\times$4 above the ref. scale" + "\n(correlated writes)",
            xy=(0.0145, 24), xytext=(0.033, 55), fontsize=8, color=st.BLUE,
            arrowprops=dict(arrowstyle="->", color=st.BLUE, lw=0.8))
bx.annotate(r"$\times$1.05 from the bound", xy=(0.212, 4.3),
            xytext=(0.075, 2.0), fontsize=7.7, color=st.DARK,
            arrowprops=dict(arrowstyle="->", color=st.GRAY, lw=0.7))
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlim(0.008, 0.32)
bx.set_ylim(1.5, 200)
bx.set_xlabel(r"per-call forgetting $\alpha_C$")
bx.set_ylabel(r"plateau$\,/\,b$")
bx.set_title(r"(b)  the size follows the $\alpha^{-1/2}$ law", fontsize=9.6)
bx.legend(fontsize=9.2, loc="lower left", frameon=True, framealpha=0.92,
          facecolor="white", edgecolor="none")

fig.tight_layout()
st.save(fig, "fig_state_law")

# ---------------- console: the numbers quoted in the text ------------------
print(f"{'label':10s} {'a_tok':>9s} {'a_C':>7s} {'b':>6s} {'plat':>7s} "
      f"{'floor':>6s} {'bound':>7s} {'p/fl':>5s} {'bd/p':>5s}")
for lab, grp, al, b, plat in pts:
    ac = acall(al)
    lo, hi = floor_of(b, ac), b / ac
    print(f"{str(lab):10s} {al:9.2e} {ac:7.4f} {b:6.3f} {plat:7.2f} "
          f"{lo:6.2f} {hi:7.1f} {plat/lo:5.2f} {hi/plat:5.2f}")
for s in (0, 1):
    sel = [(math.log(acall(al)), math.log(plat / b))
           for lab, g, al, b, plat in pts
           if g == "forced"][s::2]
    xs_, ys_ = zip(*sel)
    sl = np.polyfit(xs_, ys_, 1)[0]
    print(f"forced-alpha exponent s{s}: {sl:+.3f}")
xs_, ys_ = zip(*[(math.log(acall(al)), math.log(plat / b))
                 for lab, g, al, b, plat in pts if g == "forced"])
xs_, ys_ = np.array(xs_), np.array(ys_)
sl = np.polyfit(xs_, ys_, 1)[0]
print(f"forced-alpha exponent pooled: {sl:+.3f}  (laws: -0.5 / -1)")
