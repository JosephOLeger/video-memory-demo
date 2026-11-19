"""Figure de verification NUMERIQUE de la loi de portee (section reach).

Simulation directe des hypotheses de la derivation : memoire = vecteur de
P parametres, depots decorreles de taille bornee, oubli (1-alpha) par pas,
lecture normalisee. Trois panneaux :
  (a) SNR(alpha) a plusieurs distances d, courbe analytique superposee,
      optimum alpha* = 1 - sqrt(d/(1+d)) marque ;
  (b) enveloppe SNR(alpha*) vs d, et la loi d^-1/2 de Benna-Fusi ;
  (c) l'annulation de theta : SNR vs theta sur six decades, a (d, alpha*).
Sortie : figs/fig_reach_law.pdf + valeurs cles imprimees.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
P = 10_000


def snr_emp(d, alpha, theta=1.0, trials=400):
    """SNR empirique de la lecture normalisee d'une trace a distance d."""
    T = d + int(12 / alpha) if alpha > 0 else d + 4000
    T = min(T, d + 6000)
    decay = (1 - alpha) ** np.arange(T)
    sig, noi = [], []
    for _ in range(trials):
        u = rng.standard_normal((T, 1))  # amplitudes des depots (P implicite)
        # lecture de la trace d : signal = poids restant de son depot ;
        # bruit = somme des autres depots projetes sur une direction aleatoire
        # de recouvrement 1/sqrt(P) chacun (depots decorreles en dim P).
        w_d = decay[d] * theta
        others = decay.copy() * theta
        others[d] = 0.0
        noise = (others * u[:, 0]).sum() / np.sqrt(P)
        sig.append(w_d)
        noi.append(noise)
    return np.mean(sig) / np.std(noi)


def snr_ana(d, alpha):
    return np.sqrt(P) * (1 - alpha) ** d * np.sqrt(alpha * (2 - alpha))


fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

# (a) SNR(alpha) a d fixe
ax = axes[0]
alphas = np.geomspace(2e-4, 0.5, 40)
for d, color in [(8, "tab:blue"), (64, "tab:orange"), (512, "tab:green")]:
    emp = [snr_emp(d, a, trials=150) for a in alphas]
    ax.plot(alphas, emp, "o", ms=3, color=color, alpha=0.6)
    ax.plot(alphas, snr_ana(d, alphas), "-", color=color,
            label=f"d={d}")
    astar = 1 - np.sqrt(d / (1 + d))
    ax.axvline(astar, color=color, ls=":", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("SNR")
ax.set_title(r"(a) SNR($\alpha$) : simulation (points) vs loi (traits)")
ax.legend(fontsize=8)

# (b) enveloppe en d^-1/2
ax = axes[1]
ds = np.unique(np.geomspace(2, 2048, 18).astype(int))
env_emp, env_ana = [], []
for d in ds:
    astar = 1 - np.sqrt(d / (1 + d))
    env_emp.append(snr_emp(int(d), astar, trials=150))
    env_ana.append(snr_ana(int(d), astar))
ax.plot(ds, env_emp, "o", ms=4, label="simulation a $\\alpha^\\star$")
ax.plot(ds, env_ana, "-", label="loi")
ax.plot(ds, 0.61 * np.sqrt(P / ds), "--",
        label=r"$0.61\sqrt{P/d}$ (Benna-Fusi)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("distance $d$"); ax.set_ylabel(r"SNR($\alpha^\star$)")
ax.set_title(r"(b) l'enveloppe $d^{-1/2}$")
ax.legend(fontsize=8)

# (c) annulation de theta (lecture normalisee)
ax = axes[2]
thetas = np.geomspace(1e-3, 1e3, 13)
d = 64
astar = 1 - np.sqrt(d / (1 + d))
vals = [snr_emp(d, astar, theta=t, trials=300) for t in thetas]
ax.plot(thetas, vals, "o-", ms=4)
ax.set_xscale("log")
ax.set_ylim(0, max(vals) * 1.4)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("SNR")
ax.set_title(r"(c) $\theta$ s'annule (6 decades, $d{=}64$)")

fig.tight_layout()
out = "/mnt/c/Users/josep/Desktop/Césure/ASTAR/figs/fig_reach_law.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("verifications cles :")
for d in (8, 64, 512):
    astar = 1 - np.sqrt(d / (1 + d))
    print(f"  d={d}: alpha*={astar:.4f} (1/(2d)={1/(2*d):.4f}) "
          f"SNR emp={snr_emp(d, astar, trials=400):.1f} "
          f"ana={snr_ana(d, astar):.1f}")
cv = np.std(vals) / np.mean(vals)
print(f"  theta-independance: CV(SNR) sur 6 decades = {cv:.3f}")
