"""Fig. for section 3.2 — the chunk-factor experiment (P-THEFF2 data).

Scores the theff2 grid (1080 runs, 3 steps at init, Titans-literal
arm) by the pre-registered rule: death = non-finite state before the
end of step 3; theta* per (m, eta, seed) = geometric mean of (last
surviving theta_scale, first dead one), seeds geo-pooled; kappa at
the median realized eta/alpha of the two highest surviving runs.

(a) theta* against chunk length m, one curve per eta: the measured
    threshold is flat from m = 8 at every eta; dashed guides show the
    per-token law theta* = C_eta * m/kappa anchored at m = 32 --- it
    captures the eta-dependence and the large-m limit, while small m
    survives ABOVE it (the safe side; fresher gradients tolerate
    larger steps). Censored cells (alive at theta_scale = 2.605) as
    open triangles.
(b) Verification of the corollary: theta* x kappa/m per eta, with the
    saturated regime (m >= 8) shaded --- constant to within x1.5 at
    m = 32 while kappa/m spans 1.0 -> 12.1.

Data: results/theff2/tf2_m{M}_e{E}_t{T}_s{S}.jsonl
Output: figs/fig_theta_eff.pdf
"""
import json
import math

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st

st.apply()
RES = "/home/josep/projects/titans-mac/results/theff2"

THETAS = ["0.013", "0.018", "0.024", "0.033", "0.045", "0.062", "0.084",
          "0.115", "0.157", "0.215", "0.294", "0.401", "0.548", "0.749",
          "1.022", "1.396", "1.907", "2.605"]
MS = [1, 2, 4, 8, 16, 32]
ES = ["0.05", "0.3", "0.6", "0.9", "0.97"]
ECOL = {"0.05": st.BLUE, "0.3": st.GREEN, "0.6": st.ORANGE,
        "0.9": st.RED, "0.97": st.PURPLE}


def is_dead(rows):
    steps = [r for r in rows if "loss" in r]
    if len(steps) < 3:
        return True
    for r in steps:
        w = r.get("w1_max")
        if r.get("first_bad_upd") is not None or (r.get("p_bad") or 0) > 0:
            return True
        if not (isinstance(w, (int, float)) and math.isfinite(w)):
            return True
    return False


def med_gates(rows):
    es_, as_ = [], []
    for r in rows:
        for c in r.get("trace", []):
            if c.get("finite"):
                es_.append(c["eta"])
                as_.append(c["alpha"])
    return np.median(es_), np.median(as_)


def kappa(m, eta, alpha):
    a = 1 - alpha
    return sum(a ** (m - j) * (1 - eta ** j)
               for j in range(1, m + 1)) / (1 - eta)


def kappa_cont(m, eta):
    # alpha -> 0 closed form, valid for real m (guide line only)
    return (m - eta * (1 - eta ** m) / (1 - eta)) / (1 - eta)


cells = {}  # (m, e) -> (theta*, eta_med, alpha_med) or "cens"
for m in MS:
    for e in ES:
        ts, ems, ams = [], [], []
        for s in (0, 1):
            stt = []
            for t in THETAS:
                rows = [json.loads(l) for l in
                        open(f"{RES}/tf2_m{m}_e{e}_t{t}_s{s}.jsonl")]
                stt.append((float(t), is_dead(rows), rows))
            alive = [x for x in stt if not x[1]]
            dead = [x for x in stt if x[1]]
            if not dead:
                continue  # censored high for this seed
            t_lo = max(a[0] for a in alive)
            t_hi = min(d[0] for d in dead if d[0] > t_lo)
            ts.append(math.sqrt(t_lo * t_hi))
            for _, _, rows in sorted(alive, key=lambda x: -x[0])[:2]:
                em, am = med_gates(rows)
                ems.append(em)
                ams.append(am)
        if not ts:
            cells[(m, e)] = "cens"
        else:
            cells[(m, e)] = (math.exp(np.mean(np.log(ts))),
                             float(np.median(ems)), float(np.median(ams)))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.8, 3.6))

# ---------------- (a) theta* vs m, per eta ---------------------------------
for e in ES:
    col = ECOL[e]
    xs_m, ys, xs_c = [], [], []
    for m in MS:
        v = cells[(m, e)]
        if v == "cens":
            xs_c.append(m)
        else:
            xs_m.append(m)
            ys.append(v[0])
    ax.plot(xs_m, ys, "o-", color=col, ms=4.5, lw=1.4,
            label=rf"$\eta$ = {e}")
    if xs_c:
        ax.plot(xs_c, [2.605] * len(xs_c), "^", color=col, ms=6,
                mfc="none", mew=1.4)
    # per-token law, anchored at the largest measured m
    if xs_m:
        m0, t0 = xs_m[-1], ys[-1]
        _, em, am = cells[(m0, e)]
        c0 = t0 * kappa_cont(m0, em) / m0
        mm = np.geomspace(1, 32, 40)
        ax.plot(mm, [c0 * x / kappa_cont(x, em) for x in mm], "--",
                color=col, lw=0.9, alpha=0.5)
ax.plot([], [], "--", color=st.GRAY, lw=0.9,
        label=r"per-token law $\theta^{*} \propto m/\kappa$")
ax.plot([], [], "^", color=st.GRAY, mfc="none", ms=6, mew=1.4,
        label=r"censored (alive at $\theta$ = 2.6)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(MS)
ax.set_xticklabels([str(m) for m in MS])
ax.set_ylim(0.03, 4.2)
ax.set_xlabel("chunk length $m$")
ax.set_ylabel(r"divergence threshold $\theta^{*}$")
ax.set_title(r"(a)  $\theta^{*}$ is flat from $m = 8$; below it the"
             " deviation changes sign with $\eta$", fontsize=9.1)
ax.legend(fontsize=8.8, loc="lower left", ncol=2)

# ---------------- (b) verification of the corollary ------------------------
bx.axvspan(8, 40, color=st.GRAY, alpha=0.12, lw=0)
bx.text(17, 0.115, "measured flat regime " r"($m \geq 8$)",
        fontsize=8, color=st.DARK, ha="center")
prods_32 = []
for e in ES:
    col = ECOL[e]
    xs_m, y_tok = [], []
    for m in MS:
        v = cells[(m, e)]
        if v == "cens":
            continue
        t, em, am = v
        xs_m.append(m)
        y_tok.append(t * kappa(m, em, am) / m)
        if m == 32:
            prods_32.append(y_tok[-1])
    bx.plot(xs_m, y_tok, "o-", color=col, ms=4.5, lw=1.4,
            label=rf"$\eta$ = {e}")
med32 = float(np.median(prods_32))
bx.axhline(med32, color=st.DARK, ls=":", lw=0.9)
bx.text(1.05, med32 * 1.13, f"{med32:.2f}", fontsize=8, color=st.DARK)
bx.set_xscale("log", base=2)
bx.set_yscale("log")
bx.set_xticks(MS)
bx.set_xticklabels([str(m) for m in MS])
bx.set_ylim(0.1, 3.5)
bx.set_xlabel("chunk length $m$")
bx.set_ylabel(r"$\theta^{*}\,\kappa/m$")
bx.set_title(r"(b)  $\theta^{*}\kappa/m$ constant across $\eta$"
             " ($\\times$1.5 at $m$=32)", fontsize=9.1)
bx.legend(fontsize=8.8, loc="upper right", ncol=2)

fig.tight_layout()
st.save(fig, "fig_theta_eff")

meas = [(m, e, *cells[(m, e)]) for m in MS for e in ES
        if cells[(m, e)] != "cens"]
pk = [t * kappa(m, em, am) for m, e, t, em, am in meas]
pt = [t * kappa(m, em, am) / m for m, e, t, em, am in meas]
p32 = [t * kappa(m, em, am) / m for m, e, t, em, am in meas if m == 32]
print(f"cells: {len(meas)}/30  disp per-chunk x{max(pk)/min(pk):.0f}  "
      f"per-token x{max(pt)/min(pt):.1f}  at m=32 x{max(p32)/min(p32):.2f}")
