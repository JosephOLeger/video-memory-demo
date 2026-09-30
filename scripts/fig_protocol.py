"""Fig. for the Setting — the episodic retention protocol, as a schema.

The metric is a PAIRED CONTRAST (echo.py: memory_gain = second_cross
- second_repeat): a repeat stream [V; D...; V] and a matched cross
stream [U; D...; V] share the distractors and the scored recurrence;
the memory writes at every token of both; the only difference is
whether the scored item V appeared before. Delta = l(cross) -
l(repeat) is attributable to what the memory kept of V's first
occurrence — generic benefits of writing cancel in the pair.
Pure diagram, figstyle palette. Output: figs/fig_protocol.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import figstyle as st

st.apply()
fig, ax = plt.subplots(figsize=(8.8, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.55)
ax.axis("off")

BH = 0.55           # stream block height
WH = 0.20           # memory strip height
ROWS = [
    # (block_y, strip_y, first_label, first_color, row_label)
    (2.50, 2.10, "clip $V$", st.BLUE, "repeat stream"),
    (1.08, 0.68, "clip $U$", st.ORANGE,
     "cross stream --- $V$ never seen before"),
]
X0, X1, X2, X3 = 0.55, 1.85, 4.95, 6.40  # block edges


def block(x0, x1, y, fc, txt, tcol="white", hatch=None):
    ax.add_patch(plt.Rectangle((x0, y), x1 - x0, BH, facecolor=fc,
                               edgecolor="white", lw=1.2, hatch=hatch))
    ax.text((x0 + x1) / 2, y + BH / 2, txt, ha="center", va="center",
            fontsize=9, color=tcol)


READ_X = 5.68
for by, wy, first, fcol, rlab in ROWS:
    ax.text(X0, by + BH + 0.06, rlab, fontsize=8.3, color=st.DARK)
    block(X0, X1, by, fcol, first)
    block(X1, X2, by, "#c9c9c9", "same distractors", tcol=st.DARK)
    block(X2, X3, by, st.BLUE, "clip $V$ scored", hatch="//")
    # memory strip: rewritten at every token, in BOTH streams
    ax.add_patch(FancyBboxPatch((X0 + 0.03, wy), X3 - X0 - 0.06, WH,
                                boxstyle="round,pad=0.02",
                                facecolor=st.PURPLE, alpha=0.14,
                                edgecolor=st.PURPLE, lw=1.0))
    for x in np.arange(X0 + 0.17, X3 - 0.05, 0.33):
        if abs(x - READ_X) < 0.22:
            continue
        col = (fcol if x < X1 else (st.GRAY if x < X2 else st.BLUE))
        ax.annotate("", xy=(x, wy + WH + 0.07), xytext=(x, by - 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9,
                                    alpha=0.8, mutation_scale=6))
    # the read at the scored recurrence
    ax.annotate("", xy=(READ_X, by - 0.01), xytext=(READ_X, wy + WH + 0.07),
                arrowprops=dict(arrowstyle="-|>", color=st.GREEN, lw=1.6,
                                mutation_scale=11))

ax.text(3.40, 1.92, r"fast weights $W_t$ --- rewritten at every token,"
        " in both streams", fontsize=8.4, color=st.PURPLE, ha="center")
ax.text(6.62, 1.60, "$W$ read to\npredict $V$", fontsize=8.2,
        color=st.GREEN, ha="center", va="center")

# the single difference, highlighted
for y0 in (2.44, 1.02):
    ax.add_patch(plt.Rectangle((X0 - 0.08, y0), X1 - X0 + 0.16, 0.65,
                               fill=False, edgecolor=st.DARK, lw=1.0,
                               ls=(0, (4, 3))))
ax.text((X0 + X1) / 2, 0.47, "the only difference", fontsize=8.2,
        color=st.DARK, ha="center", style="italic")

# time arrow
ax.annotate("", xy=(X3 + 0.35, 3.38), xytext=(X0, 3.38),
            arrowprops=dict(arrowstyle="->", color=st.GRAY, lw=0.9))
ax.text(X3 + 0.45, 3.38, "time", fontsize=8.2, color=st.GRAY, va="center")

# ---------------- scoring panel --------------------------------------------
ax.add_patch(FancyBboxPatch((7.35, 0.68), 2.52, 2.37,
                            boxstyle="round,pad=0.05",
                            facecolor=st.GRAY, alpha=0.12,
                            edgecolor="none"))
ax.text(8.62, 2.80, "same block, scored twice", fontsize=8.8,
        color=st.DARK, ha="center", fontweight="bold")
ax.text(7.55, 2.38, r"repeat:  $\ell^{\,\mathrm{repeat}}$",
        fontsize=9.1, color=st.BLUE, va="center")
ax.text(7.55, 1.98, r"cross:   $\ell^{\,\mathrm{cross}}$",
        fontsize=9.1, color=st.ORANGE, va="center")
ax.plot([7.55, 9.72], [1.72, 1.72], color=st.DARK, lw=0.7)
ax.text(7.55, 1.44,
        r"$\Delta \,=\, \ell^{\,\mathrm{cross}} - \ell^{\,\mathrm{repeat}}$",
        fontsize=9.5, color=st.DARK, va="center")
ax.text(7.55, 1.10, r"$\Delta > 0$: the memory kept"
        "\n" r"$V$'s first occurrence", fontsize=8.4, color=st.DARK,
        va="center")
ax.text(7.55, 0.82, "generic write benefits cancel", fontsize=8,
        color=st.GRAY, va="center", style="italic")

fig.tight_layout()
st.save(fig, "fig_protocol")
