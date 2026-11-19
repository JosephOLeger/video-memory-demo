"""Shared visual identity for all paper figures (English labels).

Arm colors are constant across every figure so the reader builds the
reflex: red = published / unbounded rule (fails), blue = psi-bounded
(healthy), green = clipped / raw-direction magnitude rules, purple =
decoupled control, orange = auxiliary arms, gray = external reference.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
RED = "#C8433C"      # published rule / unbounded / collapsing arm
BLUE = "#2C6FBF"     # psi-bounded, healthy
GREEN = "#2E8B57"    # clipped / softclip / raw-direction magnitude
PURPLE = "#8064A8"   # decoupled control
ORANGE = "#E0821A"   # auxiliary (momentum, text, ...)
GRAY = "#8A8A8A"     # external reference / neutral
DARK = "#333333"

RED_LIGHT = "#E39B96"
BLUE_LIGHT = "#9DBEE4"
GREEN_LIGHT = "#96C8AC"

FIGS = os.environ.get(
    "FIG_OUT", "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs")
PREVIEW = os.environ.get(
    "FIG_PREVIEW",
    "/tmp/claude-1000/-home-josep-projects/"
    "53019517-9373-4e5d-b120-604194f089a9/scratchpad/figpreview")


def apply():
    # Figures are drawn wide (~11 in) and placed at <=0.9\linewidth (~4.3 in)
    # in the paper, an ~0.4x downscale; label sizes are set larger here so
    # they stay legible after that reduction. Styling only --- no data.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.65,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, name):
    """Save PDF to the paper figs dir + PNG preview for inspection."""
    os.makedirs(PREVIEW, exist_ok=True)
    pdf = os.path.join(FIGS, name + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=170,
                bbox_inches="tight")
    print("saved", pdf)
