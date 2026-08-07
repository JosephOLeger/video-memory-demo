"""Is there long-range structure in the latents for a memory to hold at all?

The memory can only pay off where the data has information that the attention window
does not already carry. If the latent stream has no exploitable structure beyond lag
``w``, then no memory architecture can help beyond ``w``, and a sweep comparing
architectures at long range is measuring the corpus rather than the model.

So this measures, with no training and no model:

    R2_inc(delta) = fraction of the variance of z_t that z_{t-delta} explains
                    *after* the last w latents have been projected out

That is the exact quantity a memory competes for: attention already sees
``w = segment_size`` latents, so only the residual is available to it.

Three things make the number interpretable.

**A null.** With 128 free parameters, a ridge fit to noise has R2 > 0. The same
computation is therefore run with ``z_{t-delta}`` taken from a *different sequence*,
which gives the estimator's floor. Anything at or below that floor is nothing.

**Held-out scoring.** The map is fitted on one half of the sequences and scored on
the other, so the reported R2 cannot be fit noise.

**The short-lag caveat.** Consecutive latents come from a sliding window with
``clip_len=64, stride=32``: they share half their input frames *by construction*.
Short-lag correlation is therefore partly an encoding artifact, not scene dynamics.
The first genuinely disjoint pair is lag 2.

Linear, hence a **lower bound**: a memory could exploit nonlinear structure this
misses. But a lower bound that already sits at the null is decisive in the direction
that matters -- it says there is nothing linear left to hold.

Note on what a lag *is* here. ``load_video_frames`` samples ``num_frames`` uniformly
across each whole ``.mp4``, so one sequence spans one entire segment and a lag of 1
is (segment duration / T) of wall-clock -- roughly 15 s for EgoLife's ~16 min
segments, not one frame. Lags are therefore reported in latents, and the wall-clock
meaning varies with segment duration.

    python -m titans.worldmodel.latent_structure --cache cache/egolife_pca128_train.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .train import load_latent_cache


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """``argmin_W ||XW - Y||^2 + lam ||W||^2``, solved on the feature Gram."""
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(A, X.T @ Y)


def _r2(Y: torch.Tensor, P: torch.Tensor) -> float:
    """Fraction of ``Y``'s total variance explained by prediction ``P``."""
    ss_res = float(((Y - P) ** 2).sum())
    ss_tot = float((Y ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def windows(z: torch.Tensor, w: int, lag: int):
    """``(X_win, X_lag, Y)`` over every valid ``t``, flattened across sequences.

    ``t`` ranges over the positions where both the ``w``-latent window and the lagged
    latent exist, so every lag is scored on the same kind of position.
    """
    B, T, D = z.shape
    t0 = max(w, lag)
    if t0 >= T:
        return None
    ts = torch.arange(t0, T)
    Y = z[:, ts, :].reshape(-1, D)
    X_win = torch.cat([z[:, ts - k, :] for k in range(1, w + 1)], dim=-1)
    X_win = X_win.reshape(-1, w * D)
    X_lag = z[:, ts - lag, :].reshape(-1, D)
    return X_win, X_lag, Y


@torch.no_grad()
def incremental_r2(z: torch.Tensor, w: int, lags, lam: float = 1e-2,
                   seed: int = 0) -> list[dict]:
    """``R2_inc`` per lag, with a shuffled-sequence null, fitted and scored disjointly."""
    B = z.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B, generator=g)
    fit_i, sco_i = perm[:B // 2], perm[B // 2:]
    # a different sequence for the lagged latent: same marginal, no shared history
    z_shift = z[torch.randperm(B, generator=g)]

    out = []
    for lag in lags:
        wl = windows(z, w, lag)
        if wl is None:
            continue
        X_win, X_lag, Y = wl
        _, X_null, _ = windows(z_shift, w, lag)
        npos = Y.shape[0] // B

        def rows(idx):
            m = torch.zeros(B, dtype=torch.bool)
            m[idx] = True
            return m.repeat_interleave(npos)

        fr, sr = rows(fit_i), rows(sco_i)
        # 1. what the attention window alone explains
        W0 = _ridge_fit(X_win[fr], Y[fr], lam)
        R_fit = Y[fr] - X_win[fr] @ W0
        R_sco = Y[sr] - X_win[sr] @ W0
        r2_win = _r2(Y[sr], X_win[sr] @ W0)

        # 2. what the lagged latent adds, on the residual
        def add(Xa):
            Wl = _ridge_fit(Xa[fr], R_fit, lam)
            return _r2(R_sco, Xa[sr] @ Wl)

        out.append({"lag": int(lag), "r2_window": r2_win,
                    "r2_inc": add(X_lag), "r2_null": add(X_null),
                    "n_fit": int(fr.sum()), "n_score": int(sr.sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="PCA-reduced cache")
    ap.add_argument("--window", type=int, default=8,
                    help="latents attention already sees (segment_size)")
    ap.add_argument("--max-seqs", type=int, default=4000, dest="max_seqs")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    z, meta = load_latent_cache(a.cache)
    if "pca" not in meta:
        raise SystemExit(f"{a.cache} is not a PCA cache.")
    z = z[:a.max_seqs].double()
    B, T, D = z.shape
    lags = [l for l in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if l < T]

    rows = incremental_r2(z, a.window, lags, lam=a.lam)
    print(f"cache {os.path.basename(a.cache)}  {B} sequences x {T} latents x {D}")
    print(f"fenetre d'attention = {a.window} latents ; R2 de la fenetre seule = "
          f"{rows[0]['r2_window']:.4f}")
    print()
    print("%6s%16s%12s%18s" % ("lag", "R2 incremental", "R2 nul", "au-dessus du nul"))
    print("-" * 54)
    for r in rows:
        # The null is typically *negative* -- an unrelated latent actively hurts on
        # held-out data -- so the threshold has to be on its magnitude, not its value.
        # Scoring against max(null, 0) would let every positive number through.
        floor = 3 * max(abs(r["r2_null"]), 1e-6)
        v = r["r2_inc"]
        flag = "oui" if v > floor else ("marginal" if v > 0 else "non")
        print("%6d%16.5f%12.5f%18s" % (r["lag"], v, r["r2_null"], flag))
    print()
    print("R2 incremental = variance de z_t expliquee par z_{t-lag} APRES avoir")
    print("projete la fenetre d'attention. C'est ce que la memoire peut gagner.")
    print("Le nul est le meme calcul avec un latent d'une AUTRE sequence.")

    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"cache": a.cache, "window": a.window, "shape": [B, T, D],
                       "rows": rows}, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
