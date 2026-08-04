"""Probe the inner loop's stability quantities along a forward pass.

The divergence this project chases happens **inside a single forward**: the outer
weights are still finite entering it and NaN leaving it, with 30 healthy steps
before and no precursor (README §8.5). Anything sampled per outer step therefore
misreads it as gradual. This module samples per *chunk*, which is the granularity
the recurrence actually advances at.

What it computes, and why each quantity.

Within a chunk of length ``m`` the gradients are all evaluated at the chunk-start
anchor, so with ``a = 1 - alpha`` the exact chunk map in the deviation ``(delta, s)``
is::

    delta_m = (a^m I - theta * sum_i w_i H_i) delta_0 + P_m s_0
    s_m     = -theta * (sum_i u_i H_i) delta_0        + eta^m s_0

    w_i = sum_{j=i}^{m} a^{m-j} eta^{j-i}        u_i = eta^{m-i}
    P_m = sum_{j=1}^{m} a^{m-j} eta^j            sum_i w_i = kappa = Q_m / (1 - eta)

The ``H_i`` are **per-token** Hessians. That matters: a chunk-level Gram
``lambda_max(K^T K)`` is already summed over the ``m`` tokens, so multiplying it by
``kappa`` -- which counts the same ``m`` stale applications -- double-counts by
about ``m``. The correct scalar is the *weighted* Gram
``lambda_max(2 X^T diag(w) X)``.

Which ``X``? For a matrix memory ``J_i = k_i``, so ``X = K`` and this is ATLAS's
quantity. For the deep memory the last layer's block has ``J_i = h_{0,i} (x) I``,
hence ``X = H_0``, the hidden activations. Both are reported: their ratio is the
"bound imposed on the wrong Gram matrix" of the theory note, measured.

Everything raw is kept in the returned dict, so a spectral radius can be recomputed
offline under a corrected formula without re-running anything.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def chunk_weights(m: int, a: float, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(w, u)`` of shape ``[m]``, the per-token weights of the chunk map."""
    if m < 1:
        raise ValueError("m must be >= 1")
    idx = torch.arange(1, m + 1, dtype=torch.float64)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)                  # [m,1], [1,m]
    upper = j >= i
    # eta ** (j - i) would take negative exponents below the diagonal; clamp them
    # to 0 before exponentiating and mask afterwards, so eta ~ 0 cannot overflow.
    terms = (a ** (m - j)) * (eta ** (j - i).clamp_min(0))
    w = torch.where(upper, terms, torch.zeros_like(terms)).sum(dim=1)
    u = eta ** (m - idx)
    return w, u


def kappa(m: int, a: float, eta: float) -> float:
    """``kappa = Q_m / (1 - eta) = sum_i w_i``: the effective-step inflation."""
    if abs(1.0 - eta) < 1e-12:                      # eta -> 1: kappa -> m * R_m / m
        w, _ = chunk_weights(m, a, eta)
        return float(w.sum())
    j = torch.arange(1, m + 1, dtype=torch.float64)
    R = (a ** (m - j)).sum()
    P = ((a ** (m - j)) * (eta ** j)).sum()
    return float((R - P) / (1.0 - eta))


def weighted_gram_lambda_max(X: torch.Tensor, w: torch.Tensor) -> float:
    """``lambda_max(2 X^T diag(w) X)``, via the cheap ``m x m`` side.

    ``X`` is ``[m, d]`` (one row per token). Uses ``diag(sqrt(w)) X X^T diag(sqrt(w))``
    because ``m`` (the chunk, 8) is far smaller than ``d``.
    """
    Xd = X.detach().double()
    r = w.detach().double().clamp_min(0).sqrt().unsqueeze(-1)
    G = (r * Xd) @ (r * Xd).T
    return float(2.0 * torch.linalg.eigvalsh(G).max())


def chunk_transfer(a: float, eta: float, m: int, theta: float,
                   lam_w: float, lam_u: float) -> tuple[torch.Tensor, float]:
    """The 2x2 chunk map reduced to one eigendirection, and its spectral radius.

    ``lam_w`` and ``lam_u`` are the weighted-Gram maxima above. Reducing two
    non-commuting matrices to their separate maxima is an approximation -- it is
    exact when they share a top eigenvector, which they do when ``eta`` is small
    enough that ``w`` and ``u`` are nearly proportional. Both inputs are returned
    by :func:`probe_chunk` so this can be redone properly later.
    """
    j = torch.arange(1, m + 1, dtype=torch.float64)
    P_m = float(((a ** (m - j)) * (eta ** j)).sum())
    B = torch.tensor([[a ** m - theta * lam_w, P_m],
                      [-theta * lam_u, eta ** m]], dtype=torch.float64)
    rho = float(torch.linalg.eigvals(B).abs().max())
    return B, rho


def hidden_activations(state: dict, k: torch.Tensor, activation: str = "silu",
                       normalize_hidden: bool = False) -> torch.Tensor:
    """``h_0 = sigma(W_0 k + b_0)`` for a batch of keys, as the memory computes it.

    ``state`` carries a leading batch dim (``w0`` is ``[B, d_h, d_in]``); ``k`` is
    ``[B, m, d_in]``. Returns ``[B, m, d_h]``.
    """
    h = torch.einsum("bmi,boi->bmo", k, state["w0"]) + state["b0"].unsqueeze(1)
    h = F.gelu(h) if activation == "gelu" else F.silu(h)
    if normalize_hidden:
        h = h / (h.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
    return h


@torch.no_grad()
def probe_chunk(mem, x_chunk: torch.Tensor, state: dict, S: Optional[dict] = None,
                b: int = 0) -> dict:
    """Every stability quantity for one chunk, at the state it starts from.

    ``mem`` is a :class:`~titans.memory.NeuralMemory`, ``x_chunk`` is ``[B, m, dim]``
    (one chunk of tokens) and ``state`` the memory weights entering it. Batch element
    ``b`` is probed -- the recurrence is per-sequence, so mixing them would be
    meaningless.
    """
    k, v, q, alpha, eta, theta = mem._project(x_chunk)[:6]
    m = k.shape[1]
    al = float(alpha[b].mean()); et = float(eta[b].mean()); th = float(theta[b].mean())
    a = 1.0 - al
    w, u = chunk_weights(m, a, et)
    kap = kappa(m, a, et)

    K = k[b]                                                   # [m, d_phi]
    H0 = hidden_activations({kk: vv[b:b + 1] for kk, vv in state.items()},
                            k[b:b + 1], mem.memory_activation,
                            mem.normalize_hidden)[0]            # [m, d_h]

    lam_k_w = weighted_gram_lambda_max(K, w)
    lam_k_u = weighted_gram_lambda_max(K, u)
    lam_h_w = weighted_gram_lambda_max(H0, w)
    lam_h_u = weighted_gram_lambda_max(H0, u)
    _, rho_k = chunk_transfer(a, et, m, th, lam_k_w, lam_k_u)   # ATLAS's Gram
    _, rho_h = chunk_transfer(a, et, m, th, lam_h_w, lam_h_u)   # the deep one

    w1 = state["w1"][b].double()
    w0 = state["w0"][b].double()
    return {
        "m": m, "alpha": al, "eta": et, "theta": th, "kappa": kap,
        # the curvatures, both Gram matrices
        "lambda_keys_w": lam_k_w, "lambda_keys_u": lam_k_u,
        "lambda_hidden_w": lam_h_w, "lambda_hidden_u": lam_h_u,
        "lambda_keys_plain": weighted_gram_lambda_max(K, torch.ones(m, dtype=torch.float64)),
        # the quantity the theory says governs the last layer's block
        "h0_norm_max": float(H0.norm(dim=-1).max()),
        "h0_norm_mean": float(H0.norm(dim=-1).mean()),
        "k_norm_mean": float(K.norm(dim=-1).mean()),
        # the sufficient condition from the note: ||W0||_2 + ||b0||_2 < threshold
        "w0_spec": float(torch.linalg.matrix_norm(w0, 2)),
        "b0_norm": float(state["b0"][b].double().norm()),
        "w1_spec": float(torch.linalg.matrix_norm(w1, 2)),
        "w1_fro": float(w1.norm()),
        "S_norm": (float(sum(float(t[b].double().pow(2).sum()) for t in S.values()) ** 0.5)
                   if S is not None else None),
        # spectral radii; > 1 predicts the chunk map is expansive
        "rho_keys": rho_k, "rho_hidden": rho_h,
        "theta_lambda_hidden": th * lam_h_w,
        "finite": bool(torch.isfinite(state["w1"][b]).all()),
    }


@torch.no_grad()
def trace_forward(mem, x: torch.Tensor, chunk_size: int, b: int = 0) -> list[dict]:
    """Probe every chunk of one forward pass, threading the real recurrence.

    Returns one dict per chunk, in order, each tagged with ``chunk`` and the token
    index it starts at -- the within-forward trajectory the phenomenology needs.
    """
    k, v, q, alpha, eta, theta = mem._project(x)[:6]
    gamma = mem._gamma(x)
    B, N = x.shape[0], x.shape[1]
    state = mem.memory.init_state(B)
    S = {kk: torch.zeros_like(vv) for kk, vv in state.items()}
    out = []
    for c, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        rec = probe_chunk(mem, x[:, start:end], state, S, b=b)
        rec["chunk"] = c
        rec["token"] = start
        out.append(rec)
        state, S, _ = mem._scan_chunked(
            k[:, start:end], v[:, start:end], alpha[:, start:end],
            eta[:, start:end], theta[:, start:end], state, S, end - start,
            q=None, gamma=None if gamma is None else gamma[:, start:end],
            x=x[:, start:end],
        )
        if not torch.isfinite(state["w1"]).all():
            out.append({"chunk": c + 1, "token": end, "finite": False,
                        "note": "state went non-finite in this chunk"})
            break
    return out
