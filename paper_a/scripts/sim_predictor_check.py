"""Validation du predicteur pr(L) sur synthetique (verite terrain exacte).

Genere des flux d'ecritures rang-1 correlees AR(1) + un init de haut rang,
mesure EXACTEMENT les memes statistiques que la sonde
(n_t, c_t, q_d, p_d, qi, pi, r_eff, Ni, pr_i), puis compare la courbe
pr(L) predite par pr_law_predictor a la courbe directe pr(M_L).

Configurations testees : avec/sans init, rho in {0, 0.8, 0.95},
norme d'ecriture constante ou variable, alpha 0 et 0.01.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/josep/projects/titans-mac/scripts")
from pr_law_predictor import predict_pr_curve

D1 = D2 = 128


def pr_of(M):
    s2 = np.linalg.svd(M, compute_uv=False) ** 2
    return s2.sum() ** 2 / (s2 ** 2).sum()


def gen_stream(L, rho, seed, var_norm=False):
    rng = np.random.default_rng(seed)
    ks = np.empty((L, D2)); es = np.empty((L, D1))
    k = rng.standard_normal(D2); e = rng.standard_normal(D1)
    ck = np.sqrt(1 - rho ** 2)
    for t in range(L):
        k = rho * k + ck * rng.standard_normal(D2)
        e = rho * e + ck * rng.standard_normal(D1)
        ks[t] = k / np.linalg.norm(k); es[t] = e / np.linalg.norm(e)
    n = rng.uniform(0.5, 2.0, L) if var_norm else np.ones(L)
    U = n[:, None, None] * es[:, :, None] * ks[:, None, :]
    return U


def measure(U, Wi):
    L = U.shape[0]
    n = np.linalg.norm(U.reshape(L, -1), axis=1)
    Un = U / n[:, None, None]
    sv2 = np.linalg.svd(Un, compute_uv=False) ** 2
    r_eff = float((sv2.sum(1) ** 2 / (sv2 ** 2).sum(1)).mean())
    D = min(64, L - 1)
    q_d, p_d = [], []
    for d in range(D + 1):
        A, B = Un[: L - d], Un[d:]
        q_d.append(float(np.einsum("tij,tik->tjk", A, B).reshape(L - d, -1)
                         .__pow__(2).sum(1).mean()))
        p_d.append(float(np.einsum("tij,tkj->tik", A, B).reshape(L - d, -1)
                         .__pow__(2).sum(1).mean()))
    Ni = np.linalg.norm(Wi)
    win = Wi / Ni
    pr_i = pr_of(Wi)
    c = np.einsum("ij,tij->t", win, Un)
    qi = float(np.einsum("ij,tik->tjk", win, Un).reshape(L, -1)
               .__pow__(2).sum(1).mean())
    pi = float(np.einsum("ij,tkj->tik", win, Un).reshape(L, -1)
               .__pow__(2).sum(1).mean())
    return dict(n=n, c=c, q_d=q_d, p_d=p_d, r_eff=r_eff,
                Ni=float(Ni), pr_i=float(pr_i), qi=qi, pi=pi)


def direct_curve(U, Wi, alpha=0.0, C=8, Ls=None):
    L = U.shape[0]
    if Ls is None:
        Ls = list(range(1, L + 1))
    out = []
    dec = (1 - alpha) ** C
    M = Wi.copy()
    j = 0
    prs = {}
    for t in range(L):
        M = dec * M + U[t]
        prs[t + 1] = None  # rempli si demande
    # recompute proprement pour chaque L demande (peu de L -> ok)
    for Lq in Ls:
        M = Wi * dec ** Lq
        for t in range(Lq):
            M = M + U[t] * dec ** (Lq - 1 - t)
        out.append(pr_of(M))
    return np.asarray(Ls), np.asarray(out)


def main():
    Ls = [10, 25, 50, 100, 200, 400, 800]
    cases = [
        ("sans init, rho=0",    0.0,  0.0, False, 0.0),
        ("sans init, rho=0.8",  0.8,  0.0, False, 0.0),
        ("init x10, rho=0",     0.0, 10.0, False, 0.0),
        ("init x10, rho=0.8",   0.8, 10.0, False, 0.0),
        ("init x10, rho=0.95",  0.95, 10.0, True,  0.0),
        ("init x10, rho=0.8, alpha=0.01", 0.8, 10.0, False, 0.01),
    ]
    rng = np.random.default_rng(42)
    Wi_base = rng.standard_normal((D1, D2))
    print(f"{'cas':34s} {'L':>5s} {'pr direct':>10s} {'pr predit':>10s} {'ecart':>7s}")
    worst = 0.0
    for name, rho, init_scale, var_norm, alpha in cases:
        U = gen_stream(800, rho, seed=1, var_norm=var_norm)
        Wi = Wi_base * (init_scale / np.linalg.norm(Wi_base)) if init_scale \
            else np.zeros((D1, D2)) + 1e-9 * Wi_base
        st = measure(U, Wi)
        _, direct = direct_curve(U, Wi, alpha=alpha, Ls=Ls)
        _, pred = predict_pr_curve(st["n"], st["c"], st["q_d"], st["p_d"],
                                   st["Ni"], st["pr_i"], st["qi"], st["pi"],
                                   alpha=alpha, Ls=Ls)
        for L, d, p in zip(Ls, direct, pred):
            e = abs(p - d) / d
            worst = max(worst, e)
            print(f"{name:34s} {L:5d} {d:10.2f} {p:10.2f} {100*e:6.1f}%")
    print(f"\npire ecart : {100*worst:.1f}%")


if __name__ == "__main__":
    main()
