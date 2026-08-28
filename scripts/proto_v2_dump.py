"""Prototype du predicteur v2 sur le dump brut (mu_unit10k_s0, cycle).

Questions, dans l'ordre :
  1. Le plateau de q_d/p_d disparait-il une fois l'ecriture moyenne m
     retiree ? (le DC est-il LA composante coherente)
  2. Quelle part de la masse est coherente : L^2 ||m||^2 vs somme ||V||^2 ?
  3. pr(L) direct : reconstruit exactement depuis le dump (verite).
  4. v2 : pr(L) predit par moments avec A_L = W_i + L m exact (spectre du
     couple (W_i, m)) + fluctuations V par stats de paires -> ecart ?
"""
import sys

import numpy as np
import torch

DUMP = "/home/josep/projects/titans-mac/results/prlaw/wr_mu_unit10k_s0_cycle_a0.pt"


def pr_of(M):
    s2 = np.linalg.svd(M, compute_uv=False) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def pair_stats(Un, D=64):
    L = Un.shape[0]
    q_d, p_d = [], []
    U = torch.from_numpy(Un)
    for d in range(D + 1):
        A, B = U[: L - d], U[d:]
        q = torch.einsum("tij,tik->tjk", A, B).flatten(1).pow(2).sum(1)
        p = torch.einsum("tij,tkj->tik", A, B).flatten(1).pow(2).sum(1)
        q_d.append(float(q.mean()))
        p_d.append(float(p.mean()))
    return np.array(q_d), np.array(p_d)


d = torch.load(DUMP, weights_only=False)
for bi in (0, 1):
    Wi = d["w1_init"][bi].numpy().astype(np.float64)
    snaps = d["snaps"][bi].numpy().astype(np.float64)
    U = np.diff(np.concatenate([Wi[None], snaps], axis=0), axis=0)
    L = U.shape[0]
    n = np.linalg.norm(U.reshape(L, -1), axis=1)
    print(f"\n===== bloc {bi} : {L} ecritures, norme moy {n.mean():.3f}")

    # 1-2. decomposition DC
    m = U.mean(0)
    V = U - m
    nm = np.linalg.norm(m)
    nV = np.linalg.norm(V.reshape(L, -1), axis=1)
    print(f"||m||={nm:.4f}  pr(m)={pr_of(m):.2f}  ||V|| moy={nV.mean():.3f}")
    print(f"masse coherente L^2||m||^2 = {L**2*nm**2:.1f}  vs  "
          f"masse fluctuante somme||V||^2 = {(nV**2).sum():.1f}  "
          f"(rapport {L**2*nm**2/max((nV**2).sum(),1e-9):.2f})")
    print(f"pr(W_i)={pr_of(Wi):.1f}  pr(W_i + L m)={pr_of(Wi + L*m):.1f}  "
          f"pr(etat final)={pr_of(Wi + U.sum(0)):.1f}")

    # plateaux avant/apres retrait du DC
    Un = U / n[:, None, None]
    q0, p0 = pair_stats(Un, D=48)
    Vn = V / np.maximum(nV, 1e-12)[:, None, None]
    qV, pV = pair_stats(Vn, D=48)
    t = slice(32, 49)
    print(f"plateau q (brut) {q0[t].mean():.4f} -> (sans DC) {qV[t].mean():.4f}")
    print(f"plateau p (brut) {p0[t].mean():.4f} -> (sans DC) {pV[t].mean():.4f}")
    print(f"r_eff brut {1/q0[0]:.2f} -> sans DC {1/qV[0]:.2f}")

    # 3-4. verite pr(L) et v2
    Ls = [8, 25, 50, 100, 200, 400, 600, 800]
    Ls = [x for x in Ls if x <= L]
    direct = []
    csum = np.cumsum(U, axis=0)
    for Lq in Ls:
        direct.append(pr_of(Wi + csum[Lq - 1]))

    # v2 : G = (A_L + S)(...)^T, A_L = W_i + somme_prefixe_mean,
    # S = somme V ; termes : exact sur A_L (petit calcul direct sur (Wi, m)),
    # S via paires de V, croises via second-moments V<->A_L.
    sv2A = {}
    pred = []
    for Lq in Ls:
        A = Wi + Lq * m
        trA2 = (A * A).sum()
        trA4 = (np.linalg.svd(A, compute_uv=False) ** 4).sum()
        w2 = nV[:Lq] ** 2
        trS2 = w2.sum()          # E tr SS^T (croises V ~ q/p, ordre 2 ignore ici)
        # croises V-V dans tr S2 : somme_{t!=s} <V_t, V_s> — mesurable :
        crossS = 0.0
        for dd in range(1, min(48, Lq - 1) + 1):
            crossS += 2.0 * float((V[:Lq - dd].reshape(Lq - dd, -1)
                                   * V[dd:Lq].reshape(Lq - dd, -1)).sum())
        trS2 += crossS
        trG = trA2 + trS2        # <A, S> ~ 0 (S centre)
        # tr G^2 :
        diag = (nV[:Lq] ** 4 / np.maximum(1 / qV[0], 1e-9)).sum() * 0 + \
               (nV[:Lq] ** 4).sum() * qV[0]
        cross = 0.0
        qpV = qV + pV
        for dd in range(1, min(48, Lq - 1) + 1):
            cross += 2.0 * float((w2[:Lq - dd] * w2[dd:Lq]).sum()) * qpV[dd]
        tail = float(qpV[36:].mean())
        tot_pairs = float(w2.sum() ** 2 - (w2 ** 2).sum())
        done_pairs = sum(2.0 * float((w2[:Lq - dd] * w2[dd:Lq]).sum())
                         for dd in range(1, min(48, Lq - 1) + 1))
        cross += max(0.0, tot_pairs - done_pairs) * tail
        # croises A-S (2e moment) : 2 tr(AA^T E[SS^T]) + 2 tr(A^T A E[S^T S])
        An = A / np.linalg.norm(A)
        qiA = np.einsum("ij,tik->tjk", An, Vn[:Lq]).reshape(Lq, -1)
        piA = np.einsum("ij,tkj->tik", An, Vn[:Lq]).reshape(Lq, -1)
        qiA = float((qiA ** 2).sum(1).mean())
        piA = float((piA ** 2).sum(1).mean())
        crossAS = 2.0 * trA2 * float(w2.sum()) * (qiA + piA)
        trG2 = trA4 + diag + cross + crossAS
        pred.append(trG ** 2 / trG2)
    print(f"{'L':>5s} {'pr direct':>10s} {'pr v2':>8s} {'ecart':>7s}")
    for Lq, dv, pv_ in zip(Ls, direct, pred):
        print(f"{Lq:5d} {dv:10.2f} {pv_:8.2f} {100*abs(pv_-dv)/dv:6.1f}%")
