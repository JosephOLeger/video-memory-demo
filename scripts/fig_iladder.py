"""Figure echelle I : ||W||(t) par barreau (log-log) + le mur I5.

Panneau (a) : les barreaux I0-I4 (2 graines, graine 0 en trait plein),
plateaux predits sous le seuil, mort I4 au plancher.
Panneau (b) : l'echelle de gain I5 (barreau I3 fidele, init x g) --
g=1 et 2 stables (erosion vers le plateau attracteur), g>=2.25 mur
en <= 8 chunks ; seuil g* dans (2.0, 2.25].
Sortie : figs/fig_iladder_wall.pdf
"""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = ("/tmp/claude-1000/-home-josep-projects/"
     "53019517-9373-4e5d-b120-604194f089a9/scratchpad/iladder2")
FLOOR = 1e-8


def series(path):
    xs, ys = [], []
    wall = None
    for line in open(path):
        r = json.loads(line)
        if "wall" in r:
            wall = r["wall"]
        if "window" not in r or "wnorm_max" not in r:
            continue
        w = r["wnorm_max"]
        xs.append(r["window"] + 1)
        ys.append(w if (w == w and w > 0) else FLOOR)
    return np.array(xs), np.array(ys), wall


fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

# (a) barreaux I0-I4
ax = axes[0]
rungs = [("I0_defaut", "I0 defauts (amnesique)", "tab:gray"),
         ("I1_alpha", "I1 alpha portee + spec-norm", "tab:blue"),
         ("I2_brut_rn", "I2 depots bruts + RN", "tab:green"),
         ("I3_fidele", "I3 fidele au papier", "tab:orange"),
         ("I4_amnesie", "I4 bruts + alpha repos (mort)", "tab:red")]
for stem, label, color in rungs:
    for seed, ls in [(0, "-"), (1, ":")]:
        p = f"{S}/{stem}_s{seed}.jsonl"
        try:
            xs, ys, _ = series(p)
        except FileNotFoundError:
            continue
        ax.plot(xs * 128, np.maximum(ys, FLOOR), ls, color=color, lw=1.2,
                label=label if seed == 0 else None)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("chunks ecrits"); ax.set_ylabel(r"$\Vert W\Vert_{\max}$")
ax.set_title("(a) echelle I v2 : plateaux predits, mort au repos")
ax.legend(fontsize=7, loc="lower left")

# (b) echelle de gain I5
ax = axes[1]
gains = [("I3_fidele", 1.0, "tab:orange"), ("I5_g2", 2.0, "tab:blue"),
         ("I5_g2.25", 2.25, "tab:purple"), ("I5_g2.5", 2.5, "tab:brown"),
         ("I5_g2.75", 2.75, "tab:pink"), ("I5_g3", 3.0, "tab:red"),
         ("I5_g4", 4.0, "tab:olive"), ("I5_g8", 8.0, "black")]
for stem, g, color in gains:
    for f in sorted(glob.glob(f"{S}/{stem}_s*.jsonl")):
        xs, ys, wall = series(f)
        if len(xs) == 0:
            continue
        seed0 = f.endswith("_s0.jsonl")
        ax.plot(xs * 128, ys, "-" if seed0 else ":", color=color, lw=1.2,
                label=(f"g={g:g}" + (" (mur)" if wall is not None else ""))
                if seed0 else None)
        if wall is not None:
            ax.plot(xs[-1] * 128, ys[-1], "x", color=color, ms=8, mew=2)
ax.axhline(8.5, color="gray", lw=0.6, ls="--")
ax.text(2e5, 9.5, "plateau attracteur", fontsize=7, color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("chunks ecrits")
ax.set_title(r"(b) echelle de gain I5 : $g^\ast \in (2.0, 2.25]$, transition rasoir")
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
out = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs/fig_iladder_wall.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
for stem, g, _ in gains:
    for f in sorted(glob.glob(f"{S}/{stem}_s*.jsonl")):
        xs, ys, wall = series(f)
        tag = f.rsplit('/', 1)[-1]
        print(f"  {tag:24s} n={len(xs):5d} wall={wall} last={ys[-1] if len(ys) else None:.3g}")
