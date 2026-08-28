"""Figure 2 — la trace mecanisme D1 (sonde B, b_t008_s0 / b_t005_s0).

(a) Pas fatal (7) par chunk : theta*lambda_hidden traverse le seuil ~2
    pendant que theta*lambda_keys reste plat ; ||h0|| et ||W1||_2 partent.
(b) Par pas : max theta*lambda_hidden — le cliquet (precurseur au pas 6)
    contre le controle theta-scale 0.05 qui vit sous le seuil 300 pas.
Sortie : figs/fig_mechanism.pdf
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = ("/tmp/claude-1000/-home-josep-projects/"
     "53019517-9373-4e5d-b120-604194f089a9/scratchpad")
THR = 2.37


def load(f):
    return [json.loads(l) for l in open(f"{S}/{f}.jsonl")]


def traces(recs):
    return {r["step"]: r["trace"] for r in recs if "trace" in r}


t8 = traces(load("b_t008_s0"))
t5 = traces(load("b_t005_s0"))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})

# --- (a) pas fatal, par chunk ------------------------------------------------
tr = [r for r in t8[7] if r.get("finite", True) and "theta" in r]
ch = [r["chunk"] for r in tr]
ax.plot(ch, [r["theta_lambda_hidden"] for r in tr], "o-", color="tab:red",
        label=r"$\theta\,\lambda_{\rm hidden}$ (profond)")
ax.plot(ch, [r["theta"] * r["lambda_keys_w"] for r in tr], "s-",
        color="tab:blue", label=r"$\theta\,\lambda_{\rm keys}$ (lineaire, qk-norm)")
ax.plot(ch, [r["h0_norm_max"] for r in tr], "^--", color="tab:orange",
        alpha=0.8, label=r"$\Vert h_0\Vert_{\max}$")
ax.plot(ch, [r["w1_spec"] for r in tr], "v--", color="tab:purple",
        alpha=0.8, label=r"$\Vert W_1\Vert_2$")
ax.axhline(THR, color="k", ls=":", lw=1)
ax.text(0.3, THR * 1.25, "seuil ~2.4", fontsize=8)
last = max(ch)
ax.axvline(last + 1, color="tab:red", ls="-", lw=8, alpha=0.25)
ax.text(last + 0.55, 1e-2, "etat non fini", rotation=90, fontsize=8,
        color="tab:red")
ax.set_yscale("log")
ax.set_xlabel("chunk (pas fatal, theta-scale 0.08, graine 0)")
ax.set_ylabel("echelle log")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("(a) La boucle au forward : theta plat, la courbure part", fontsize=9)

# --- (b) cliquet inter-pas ---------------------------------------------------
def per_step_max(td):
    xs, ys = [], []
    for s in sorted(td):
        vals = [r["theta_lambda_hidden"] for r in td[s]
                if r.get("finite", True) and "theta_lambda_hidden" in r
                and r["theta_lambda_hidden"] == r["theta_lambda_hidden"]]
        if vals:
            xs.append(s)
            ys.append(max(vals))
    return xs, ys


x8, y8 = per_step_max({s: t for s, t in t8.items() if s <= 7})
x5, y5 = per_step_max(t5)
bx.plot(x5, y5, "-", color="tab:green", alpha=0.8, lw=1,
        label="theta-scale 0.05 (300 pas, survit)")
bx.plot(x8, y8, "o-", color="tab:red",
        label="theta-scale 0.08 (meurt au pas 7)")
bx.axhline(THR, color="k", ls=":", lw=1)
bx.text(0.2, THR * 1.2, "seuil ~2.4", fontsize=8)
bx.set_yscale("log")
bx.set_xlabel("pas d'entrainement")
bx.set_ylabel(r"max chunk de $\theta\,\lambda_{\rm hidden}$")
bx.legend(fontsize=7.5, loc="upper left")
bx.set_title("(b) Le cliquet : precurseur au pas 6, mort au pas 7", fontsize=9)
bx.set_xlim(-2, max(max(x5), 60))

fig.tight_layout()
out = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs/fig_mechanism.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("fatal max theta*lambda:", max(y8), "| control max over 300 steps:", max(y5))
