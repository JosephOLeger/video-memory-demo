"""Deux figures depuis l'echelle Titans-litterale (results/titans_lit/).

fig_loss_blind.pdf : tl_t003_s0 -- la loss externe reste dans [0.94, 1.06]
pendant que max|W1| des blocs memoire monte de 18 a 2e14 sur 8 pas ; les
portes (theta) ne bougent pas. La cecite de la loss, sur l'objet publie
(Titans Eq 2/5, aucun NS).

fig_mechanism.pdf : (a) pas de mort par barreau theta (le rasoir en temps
de mort, mort DANS le premier forward des theta >= 0.08) ; (b) trace par
chunk du pas fatal de tl_t012_s0 : theta plat, lambda_keys borne,
theta*lambda_hidden franchit 2 puis l'etat devient non fini.

Remplacent les versions retirees (figs/retired/), qui venaient de runs de
l'hybride purge.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs"
RES = "/home/josep/projects/titans-mac/results/titans_lit"


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def load(label):
    rows = [json.loads(l) for l in open(f"{RES}/{label}.jsonl")]
    steps = {r["step"]: r for r in rows if "loss" in r}
    traces = {r["step"]: r["trace"] for r in rows if "trace" in r}
    summ = [r for r in rows if r.get("summary")][0]
    return steps, traces, summ


# ---------- fig_loss_blind : tl_t003_s0 ----------
steps, traces, summ = load("tl_t003_s0")
xs = sorted(s for s in steps if s <= summ["death_step"])
loss = [steps[s]["loss"] for s in xs]
w1 = [steps[s]["w1_max"] for s in xs]
w1 = [w if fin(w) else float("nan") for w in w1]
theta = []
for s in xs:
    tf = [c["theta"] for c in traces.get(s, []) if fin(c.get("theta"))]
    theta.append(sum(tf) / len(tf) if tf else float("nan"))

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(xs, loss, "o-", color="tab:blue", label="loss externe")
ax.plot(xs, theta, "s--", color="tab:green", ms=4,
        label=r"porte $\theta$ (moy. chunks)")
ax.set_xlabel("pas externe")
ax.set_ylabel(r"loss / porte $\theta$")
ax.set_ylim(0, 1.25)
ax2 = ax.twinx()
ax2.plot(xs, w1, "^-", color="tab:red", label=r"$\max|W_1|$ (memoire)")
ax2.set_yscale("log")
ax2.set_ylabel(r"$\max|W_1|$ (echelle log)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.annotate("etat non fini\npendant le forward\n(pas 8, update 31)",
             xy=(xs[-1], w1[-1]), xytext=(4.0, 3e10), fontsize=7,
             color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
ax.set_title(r"Titans litteral, $\theta_{scale}=0.03$ : la loss ne voit rien",
             fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_loss_blind.pdf", bbox_inches="tight")
print("saved fig_loss_blind.pdf")

# ---------- fig_mechanism ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

# (a) rasoir en temps de mort
ax = axes[0]
deaths = {}
for th, labs in [(0.03, ["tl_t003_s0", "tl_t003_s1"]),
                 (0.05, ["tl_t005_s0", "tl_t005_s1"]),
                 (0.08, ["tl_t008_s0", "tl_t008_s1"]),
                 (0.12, ["tl_t012_s0", "tl_t012_s1"])]:
    deaths[th] = [load(l)[2]["death_step"] for l in labs]
for th, ds in deaths.items():
    for d in ds:
        ax.plot(th, d + 0.5, "x", color="tab:red", ms=10, mew=2.5)
ax.axhspan(0.4, 1.6, color="tab:red", alpha=0.08)
ax.annotate("mort dans le PREMIER forward\n(<= 32 ecritures)",
            xy=(0.095, 0.75), fontsize=7, ha="center", color="tab:red")
ax.set_xscale("log")
ax.set_xticks([0.03, 0.05, 0.08, 0.12])
ax.set_xticklabels(["0.03", "0.05", "0.08", "0.12"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yscale("log")
ax.set_yticks([0.5, 1.5, 4.5, 8.5])
ax.set_yticklabels(["0", "1", "4", "8"])
ax.set_xlabel(r"$\theta_{scale}$")
ax.set_ylabel("pas de mort (2 graines)")
ax.set_title("(a) aucun barreau ne survit : rasoir en temps de mort",
             fontsize=9)

# (b) trace par chunk du pas fatal, tl_t012_s0
steps, traces, summ = load("tl_t012_s0")
tr = traces[summ["death_step"]]
ch = [c["chunk"] for c in tr if fin(c.get("theta_lambda_hidden"))]
thl = [c["theta_lambda_hidden"] for c in tr if fin(c.get("theta_lambda_hidden"))]
lk = [c["lambda_keys_w"] for c in tr if fin(c.get("lambda_keys_w"))]
th = [c["theta"] for c in tr if fin(c.get("theta"))]
ax = axes[1]
ax.plot(ch, thl, "o-", color="tab:red",
        label=r"$\theta\,\lambda_{hidden}$")
ax.plot(ch[:len(lk)], lk, "s-", color="tab:blue",
        label=r"$\lambda_{keys}$ (borne par la parametrisation)")
ax.plot(ch[:len(th)], th, "^-", color="tab:green",
        label=r"porte $\theta$ (plate)")
ax.axhline(2, color="k", ls=":", lw=1)
ax.annotate("seuil 2", xy=(0.3, 2.6), fontsize=7)
ax.axvline(max(ch) + 0.5, color="tab:red", ls="--", lw=1)
ax.annotate("premier chunk\nnon fini", xy=(max(ch) - 0.2, 1e5),
            fontsize=7, color="tab:red", ha="right")
ax.set_yscale("log")
ax.set_xlabel("chunk (ecritures de 8, dans UN forward)")
ax.set_ylabel("valeur (echelle log)")
ax.set_title(r"(b) pas fatal, $\theta_{scale}=0.12$ : "
             r"$\theta\lambda$ franchit 2, $\theta$ ne bouge pas",
             fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_mechanism.pdf", bbox_inches="tight")
print("saved fig_mechanism.pdf")
