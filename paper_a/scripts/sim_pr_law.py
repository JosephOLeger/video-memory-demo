"""Simulation de la loi pr_infini = f(alpha, rho, r, d) -- regime borne.

Modele : M_t = (1-alpha) M_{t-1} + U_t, ||U_t||_F = 1 (convention unit),
U_t = E_t K_t^T / ||.||_F avec K_t (d2 x r) les cles, E_t (d1 x r) les
valeurs/erreurs. Correlation AR(1) sur les cles (rho_k) et, en option,
sur les valeurs (rho_v). Geometrie : cles tirees d'une covariance a
spectre exponentiel de participation cible g (isotrope si g = d2).

pr mesure comme dans long_run.py : (sum sv^2)^2 / sum sv^4.

Loi testee (derivation moments de Marchenko-Pastur, note du 28/08) :
    1/pr = alpha / (r_eff * kappa_rho) + 1/d_geo
    kappa_rho = (1 - rho_k^2) / (1 + rho_k^2)
avec r_eff et d_geo a confronter aux valeurs mesurees.
"""
import json
import math

import numpy as np

OUTJ = "/home/josep/projects/titans-mac/results/sim_pr_law.json"

D1, D2 = 128, 128
BLOCK = 2000


def pr_of(M):
    sv = np.linalg.svd(M, compute_uv=False)
    s2 = sv ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def key_sqrt_cov(d, g, rng):
    """Racine d'une covariance a spectre exponentiel de participation ~g."""
    if g >= d:
        return None
    # spectre lam_i = exp(-i/tau) ; participation (sum lam)^2/sum lam^2
    # = tanh(1/2tau)^-1 ... resolu numeriquement sur tau
    taus = np.linspace(0.5, 4 * g, 4000)
    lam = np.exp(-np.arange(d)[None, :] / taus[:, None])
    prs = lam.sum(1) ** 2 / (lam ** 2).sum(1)
    tau = taus[np.argmin(np.abs(prs - g))]
    lam = np.exp(-np.arange(d) / tau)
    lam *= d / lam.sum()
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    return Q * np.sqrt(lam)[None, :]  # M = Q diag(sqrt(lam))


def run(alpha, rho_k, rho_v, r, seed, g=None, T=None):
    rng = np.random.default_rng(seed)
    T = T or int(math.ceil(8.0 / alpha))
    M = np.zeros((D1, D2))
    sq = key_sqrt_cov(D2, g, rng) if g else None
    K = rng.standard_normal((D2, r))
    E = rng.standard_normal((D1, r))
    c_k = math.sqrt(1 - rho_k ** 2)
    c_v = math.sqrt(1 - rho_v ** 2)
    done = 0
    while done < T:
        L = min(BLOCK, T - done)
        Ks = np.empty((L, D2, r))
        Es = np.empty((L, D1, r))
        fk = rng.standard_normal((L, D2, r))
        fv = rng.standard_normal((L, D1, r))
        for t in range(L):  # AR(1) sequentiel (leger), collapse vectorise apres
            K = rho_k * K + c_k * fk[t]
            E = rho_v * E + c_v * fv[t]
            Ks[t] = K
            Es[t] = E
        if sq is not None:
            Kg = np.einsum("ij,tjr->tir", sq, Ks)
        else:
            Kg = Ks
        U = np.einsum("tir,tjr->tij", Es, Kg)
        U /= np.linalg.norm(U, axis=(1, 2), keepdims=True)
        w = (1 - alpha) ** np.arange(L - 1, -1, -1.0)
        M = (1 - alpha) ** L * M + np.einsum("t,tij->ij", w, U)
        done += L
    return pr_of(M)


def main():
    out = []
    alphas = [1e-2, 3e-3, 1e-3, 3e-4]
    rhos = [0.0, 0.5, 0.8, 0.95, 0.99]
    for r in (1, 8):
        for rho in rhos:
            for a in alphas:
                for seed in (0, 1):
                    p = run(a, rho, 0.0, r, seed)
                    out.append(dict(alpha=a, rho_k=rho, rho_v=0.0, r=r,
                                    seed=seed, g=None, pr=p))
                    print(f"r={r} rho_k={rho} alpha={a:g} s{seed}: pr={p:.2f}",
                          flush=True)
    # controle : correlation cote VALEURS seulement (predite ~inoffensive)
    for rho in (0.8, 0.99):
        for seed in (0, 1):
            p = run(1e-3, 0.0, rho, 8, seed)
            out.append(dict(alpha=1e-3, rho_k=0.0, rho_v=rho, r=8,
                            seed=seed, g=None, pr=p))
            print(f"[valeurs] rho_v={rho} s{seed}: pr={p:.2f}", flush=True)
    # plafond geometrique : cles dans un sous-espace de participation g
    for g in (16, 64):
        for seed in (0, 1):
            p = run(1e-3, 0.0, 0.0, 8, seed, g=g)
            out.append(dict(alpha=1e-3, rho_k=0.0, rho_v=0.0, r=8,
                            seed=seed, g=g, pr=p))
            print(f"[geo] g={g} s{seed}: pr={p:.2f}", flush=True)
    # point d'ancrage au regime des artefacts (alpha efface ~1e-4)
    for seed in (0,):
        p = run(1e-4, 0.95, 0.0, 8, seed)
        out.append(dict(alpha=1e-4, rho_k=0.95, rho_v=0.0, r=8,
                        seed=seed, g=None, pr=p))
        print(f"[ancrage] alpha=1e-4 rho=0.95: pr={p:.2f}", flush=True)
    json.dump(out, open(OUTJ, "w"), indent=1)
    print("ecrit", OUTJ)


if __name__ == "__main__":
    main()
