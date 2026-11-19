"""Predicteur pr(L) a somme finie depuis les statistiques d'ecritures.

Entrees (ce que diag_pr_collapse_probe.py mesure) :
  n_t      normes des ecritures                       [L]
  c_t      cos signes  <W_init, U_t> / (||W_init|| n_t)  [L]
  q_d,p_d  recouvrements moyens ||Un_t^T Un_{t+d}||_F^2 (cles) et
           ||Un_t Un_{t+d}^T||_F^2 (valeurs)          [D+1]
  r_eff    participation moyenne du spectre d'une ecriture (= 1/q_0)
  Ni, pr_i norme et pr de W_init
  qi, pi   recouvrements second-moment init<->ecriture (moyennes)

Modele : M_L = w0 * W_init + somme_t w_t U_t, avec w_t les poids de
decroissance ((1-alpha)^{C(L-t)} ; tous 1 a alpha ~ 1e-4).

Moments (les memes appariements que la derivation du 27/08, note
ASTAR/derivation_loi_pr_2026-08-27.md ; termes d'ordre trois
init-ecriture-ecriture-init mixtes gardes via le carre du cos moyen,
le reste d'ordre trois neglige — valide sur synthetique par
sim_predictor_check.py) :

  trG   = w0^2 Ni^2 + sum w_t^2 n_t^2 + 2 w0 Ni sum w_t n_t c_t
  trG^2 ~ w0^4 Ni^4 / pr_i                                  (init pur)
        + sum_t w_t^4 n_t^4 * q_0                           (diagonale ; q_0 = 1/r_eff)
        + sum_{t!=s} w^2 w^2 n^2 n^2 (q_{|t-s|} + p_{|t-s|})  (croises ecritures)
        + 2 w0^2 Ni^2 sum_t w_t^2 n_t^2 (qi + pi)           (init x ecriture, 2e moment)
        + 4 w0^2 Ni^2 (sum_t w_t n_t c_t)^2 / pr_i_eff      (terme mixte au carre)

  pr(L) = trG^2 / trG2
"""
from __future__ import annotations

import numpy as np


def predict_pr_curve(n, c, q_d, p_d, Ni, pr_i, qi, pi,
                     alpha: float = 0.0, tokens_per_call: int = 8,
                     Ls=None) -> tuple[np.ndarray, np.ndarray]:
    """pr predit aux longueurs ``Ls`` (defaut : chaque ecriture)."""
    n = np.asarray(n, float)
    c = np.asarray(c, float)
    q = np.asarray(q_d, float)
    p = np.asarray(p_d, float)
    L_max = len(n)
    if Ls is None:
        Ls = np.arange(1, L_max + 1)
    D = len(q) - 1
    qp = q + p
    q0 = q[0]
    out = []
    for L in Ls:
        L = int(L)
        if alpha > 0.0:
            dec = (1 - alpha) ** tokens_per_call
            w = dec ** np.arange(L - 1, -1, -1.0)
            w0 = dec ** L
        else:
            w = np.ones(L)
            w0 = 1.0
        wn2 = (w * n[:L]) ** 2
        wnc = w * n[:L] * c[:L]
        trG = w0 ** 2 * Ni ** 2 + wn2.sum() + 2 * w0 * Ni * wnc.sum()
        # croises ecritures : somme sur les lags, stats stationnaires
        cross = 0.0
        for d in range(1, min(D, L - 1) + 1):
            cross += 2.0 * float((wn2[:L - d] * wn2[d:]).sum()) * qp[d]
        if L - 1 > D:
            # au-dela du lag mesure : plateau geometrique (moyenne de queue)
            tail = float(qp[max(1, D - 8):].mean())
            # somme des paires restantes
            tot_pairs = float(wn2.sum() ** 2 - (wn2 ** 2).sum())
            done_pairs = 0.0
            for d in range(1, D + 1):
                done_pairs += 2.0 * float((wn2[:L - d] * wn2[d:]).sum())
            cross += max(0.0, tot_pairs - done_pairs) * tail
        trG2 = (w0 ** 4 * Ni ** 4 / pr_i
                + float(((w * n[:L]) ** 4).sum()) * q0
                + cross
                + 2 * w0 ** 2 * Ni ** 2 * float(wn2.sum()) * (qi + pi)
                + 4 * w0 ** 2 * Ni ** 2 * float(wnc.sum() ** 2) / max(pr_i, 1.0))
        out.append(trG ** 2 / trG2)
    return np.asarray(Ls), np.asarray(out)


def scalars_from_qp(q_d, p_d, r_eff):
    """Lecture interpretable : plafonds geometriques et facteur Phi."""
    q = np.asarray(q_d, float)
    p = np.asarray(p_d, float)
    tail_q = float(q[max(1, len(q) - 16):].mean())
    tail_p = float(p[max(1, len(p) - 16):].mean())
    g_rows = 1.0 / max(tail_q, 1e-9)      # cote cles
    g_cols = 1.0 / max(tail_p, 1e-9)      # cote valeurs
    # Phi = 1 + 2 * somme des exces de correlation au-dela du plateau,
    # normalise par la diagonale (q_0 = 1/r_eff)
    exc = float(np.clip(q[1:] - tail_q, 0, None).sum()
                + np.clip(p[1:] - tail_p, 0, None).sum())
    phi = 1.0 + 2.0 * exc / max(q[0], 1e-9)
    return {"g_rows": g_rows, "g_cols": g_cols, "phi": phi,
            "r_eff": float(r_eff)}
