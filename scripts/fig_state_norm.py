"""Fig. D2 of the divergence section — bounded magnitude is sufficient
for stability without sacrificing memory utility (Figs 2+4 merged per
advisor feedback, 07/09).

(a) state norm along a replay of 800 writes, log-log, six runs:
    - unbounded rule, theta_scale=0.03 witness pair (longest-lived
      rung; higher rungs die inside their first forward, Fig. D1):
      both end non-finite (x);
    - psi_c-bounded rule after 10k training steps: NS direction (usq,
      2 seeds, blue) and raw direction (rawdir, 2 seeds, green), flat.
(b) retention gain after 10k steps, individual seeds (no bar chart):
    unbounded = diverges, no recall to measure; bounded raw n=5,
    mean 0.410 +- 0.011 (s0-1 + reinforced s2-4); bounded NS 0.533 (n=1).
    Consigned: results/predictions_collapse_2026-08-28.md (31/08).

No in-image figure title (titles live in the LaTeX caption).
Data: results/titans_lit/tl_t003_s{0,1}.jsonl,
results/prlaw2/T2_usq10k_s{0,1}.jsonl,
results/prlaw3/N_rawdir_usq.jsonl (replay w1n_b0 trajectories).
Output: figs/fig_state_norm.pdf
"""
import json
import math

import matplotlib.pyplot as plt

import figstyle as st

st.apply()
TL = "/home/josep/projects/titans-mac/results/titans_lit"
P2 = "/home/josep/projects/titans-mac/results/prlaw2"
P3 = "/home/josep/projects/titans-mac/results/prlaw3"


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(9.4, 4.3), gridspec_kw={"width_ratios": [2.3, 1.0]})

# ---------------- (a) published rule: theta 0.03 witness pair --------------
seen = set()
for lab, ls in [("tl_t003_s0", "-"), ("tl_t003_s1", "--")]:
    rows = [json.loads(l) for l in open(f"{TL}/{lab}.jsonl")]
    traces = {r["step"]: r["trace"] for r in rows if "trace" in r}
    steps = {r["step"]: r for r in rows if "loss" in r}
    summ = [r for r in rows if r.get("summary")][0]
    # merge chunk-level traces and per-step maxima; the fast state
    # resets each forward, so the honest monotone summary is the
    # running maximum of the state norm along writes
    pts = []
    for s in sorted(traces):
        if s > summ["death_step"]:
            break
        for c in traces[s]:
            if fin(c.get("w1_fro")):
                pts.append((s * 32 + c["chunk"] + 1, c["w1_fro"]))
    for s in sorted(steps):
        if s > summ["death_step"]:
            break
        if fin(steps[s].get("w1_max")):
            pts.append(((s + 1) * 32, steps[s]["w1_max"]))
    pts.sort()
    xs, ys, run = [], [], 0.0
    for x, y in pts:
        run = max(run, y)
        xs.append(x)
        ys.append(min(run, 5e17))  # display ceiling: death marker stays
        # attached to its curve inside the frame
    label = ("unbounded magnitude, raw direction (2 seeds)"
             if not seen else None)
    seen.add(lab)
    ax.plot(xs, ys, ls, color=st.RED, lw=1.5, alpha=0.9, label=label)
    ax.plot(xs[-1], min(ys[-1], 5e17), "x", color=st.RED, ms=8, mew=2.0,
            zorder=5)

# ---------------- (a) converse control: NS direction, raw magnitude --------
# d2interne echelle, readmitted SOLELY as the unbounded+NS control
# (decision 08/09; the D2-interne verdict chain stays retired)
D2 = "/home/josep/projects/titans-mac/results/d2interne"
seen_rs = False
for lab, ls in [("fine_t008_s0", "-"), ("fine_t008_s1", "--")]:
    rows = [json.loads(l) for l in open(f"{D2}/{lab}.jsonl")]
    summ = [r for r in rows if r.get("summary")][0]
    pts = sorted(((r["step"] + 1) * 32, r["w1_max"]) for r in rows
                 if "loss" in r and r["step"] <= summ["death_step"]
                 and fin(r.get("w1_max")))
    xs, ys, run = [], [], 0.0
    for x, y in pts:
        run = max(run, y)
        xs.append(x)
        ys.append(min(run, 5e17))
    label = (None if seen_rs
             else "unbounded magnitude, NS direction (2 seeds)")
    seen_rs = True
    ax.plot(xs, ys, ls, color=st.ORANGE, lw=1.5, alpha=0.9, label=label)
    ax.plot(xs[-1], ys[-1], "x", color=st.ORANGE, ms=8, mew=2.0, zorder=5)

# ---------------- (a) bounded family: replay trajectories ------------------
def replay_traj(path, frozen_suffix):
    for line in open(path):
        r = json.loads(line)
        if "traj" in r and r["frozen"].endswith(frozen_suffix):
            return ([(c["clip"] + 1) * 8 for c in r["traj"]],
                    [c["w1n_b0"] for c in r["traj"]])
    return None, None


bounded = [
    (f"{P2}/T2_usq10k_s0.jsonl", "mu_usq10k_s0_at10000.pt", st.BLUE, "-",
     r"$\psi_c$ bound, NS direction (2 seeds)"),
    (f"{P2}/T2_usq10k_s1.jsonl", "mu_usq10k_s1_at10000.pt", st.BLUE, "--",
     None),
    (f"{P3}/N_rawdir_usq.jsonl", "rawdir_usq_s0.pt", st.GREEN, "-",
     r"$\psi_c$ bound, raw direction (2 seeds)"),
    (f"{P3}/N_rawdir_usq.jsonl", "rawdir_usq_s1.pt", st.GREEN, "--", None),
]
for path, frz, col, ls, label in bounded:
    xs, ys = replay_traj(path, frz)
    if xs is None:
        print("MISSING", frz)
        continue
    ax.plot(xs, ys, ls, color=col, lw=1.8, label=label)

ax.plot([], [], "x", color=st.RED, ms=8, mew=2.0,
        label="state becomes non-finite")
ax.text(8, 1e12, "13 orders of magnitude\nin fewer than 300 writes",
        fontsize=8.3, color=st.RED)
ax.text(250, 150, "same architecture, bounded write:\nflat over a full"
        " replay, every seed", fontsize=8.3, color=st.BLUE, ha="left")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(6, 1000)
ax.set_ylim(2.2, 1e18)
ax.set_xlabel("writes (memory updates)")
ax.set_ylabel(r"running max of state norm $\Vert W_1\Vert$ (log)")
ax.set_title("(a)  numerical stability along replay", fontsize=9.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
          fontsize=8.2)

# ---------------- (b) recall: individual seeds, no bars --------------------
# consigned verdicts P-2x2-1/-2/-3 (31/08), joint scoring at 10k steps
gains = [
    (1, [0.407, 0.394, 0.4105, 0.4254, 0.4146], st.GREEN),  # raw dir, n=5
    (2, [0.533], st.BLUE),           # psi_c bound, NS direction (n=1)
]
for x, ys, col in gains:
    for y in ys:
        bx.plot(x, y, "o", color=col, ms=8, mfc=col, alpha=0.85, zorder=4)
    bx.plot([x - 0.16, x + 0.16], [sum(ys) / len(ys)] * 2, "-", color=col,
            lw=1.6, zorder=3)
bx.plot(0, 0.0, "x", color=st.RED, ms=10, mew=2.4, zorder=4)
bx.annotate("diverges\n(no recall\nto measure)", xy=(0, 0.02),
            xytext=(0, 0.09), fontsize=8.3, color=st.RED, ha="center")
# The NS retention gain still has ONE scored seed against raw's five, so
# the gap is limited by the NS side, not the raw side. Panel (a)'s "2
# seeds" describes the state-norm trajectories, a different quantity; do
# not let the reader carry it over. Label n on the axis.
bx.annotate(r"$\approx$ +30%" + "\n(1 seed vs 5:\nnot resolved)",
            xy=(1.5, 0.47), fontsize=8.3, color=st.DARK, ha="center")
bx.set_xticks([0, 1, 2])
bx.set_xticklabels(["unbounded", "bounded,\nraw dir. ($n{=}5$)",
                    "bounded,\nNS dir. ($n{=}1$)"], fontsize=7.4)
bx.set_xlim(-0.55, 2.55)
bx.set_ylim(-0.03, 0.62)
bx.set_ylabel("retention gain (10k steps)")
bx.set_title("(b)  memory utility after training", fontsize=9.6)

fig.tight_layout()
st.save(fig, "fig_state_norm")
