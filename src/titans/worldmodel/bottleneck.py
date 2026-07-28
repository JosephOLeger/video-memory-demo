"""Linear (PCA) bottleneck on frozen-encoder latents.

The V-JEPA-2 latents are ``D=1024``-dimensional but their intrinsic dimensionality
is far lower (on EgoLife train, ~120 components capture 90% of the variance). Working
in the full 1024-D space is wasteful in two ways the project cares about:

* **Parameters go to I/O, not memory.** With a small model width the `in_proj`
  (``D -> dim``) and `out_proj` (``dim -> D``) dominate the parameter count, so most
  of the model is just projecting in and out of the latent space rather than the
  long-term memory / attention we study.
* **The GMM head scales with ``D``.** A ``K``-component diagonal mixture needs
  ``K*(1+2D)`` outputs; at ``D=1024`` that head alone dwarfs the model.

Fitting a PCA on the *train* latents and modelling in the reduced (optionally
whitened) ``k``-dim space fixes both: smaller I/O projections, a tractable GMM head,
a lower-entropy target, and less overfitting -- at the cost of the PCA reconstruction
error (fixed, ~10-15% variance at ``k=128``). The transform is *fit on train only*
(no val leakage) and applied to both splits.
"""

from __future__ import annotations

from typing import Dict

import torch

Pca = Dict[str, object]


def fit_pca(latents: torch.Tensor, k: int, *, whiten: bool = True, eps: float = 1e-6) -> Pca:
    """Fit a per-channel-standardised PCA on ``latents`` ``[..., D]`` -> top ``k`` PCs.

    Returns a dict of float32 tensors (``mean``, ``std``, ``components`` ``[k, D]``,
    ``sv`` the per-PC score std) plus ``whiten``/``k``/``explained``. Fit in float64
    for a stable SVD, stored/applied in float32.
    """
    X = latents.reshape(-1, latents.shape[-1]).double()
    if k < 1 or k > X.shape[-1]:
        raise ValueError(f"k must be in [1, D={X.shape[-1]}]")
    mean = X.mean(0)
    std = X.std(0) + eps
    Xs = (X - mean) / std                       # per-channel standardise (centres too)
    U, S, Vt = torch.linalg.svd(Xs, full_matrices=False)
    comps = Vt[:k]                              # [k, D]
    sv = S[:k] / (X.shape[0] - 1) ** 0.5 + eps  # std of each PC score
    explained = (S[:k] ** 2).sum() / (S ** 2).sum()
    return {
        "mean": mean.float(),
        "std": std.float(),
        "components": comps.float(),
        "sv": sv.float(),
        "whiten": whiten,
        "k": k,
        "explained": float(explained),
    }


def apply_pca(latents: torch.Tensor, pca: Pca) -> torch.Tensor:
    """Project ``latents`` ``[..., D]`` into the fitted ``k``-dim space -> ``[..., k]``."""
    dev = latents.device
    xs = (latents - pca["mean"].to(dev)) / pca["std"].to(dev)
    z = xs @ pca["components"].to(dev).T        # [..., k]
    if pca["whiten"]:
        z = z / pca["sv"].to(dev)
    return z


def inverse_pca(reduced: torch.Tensor, pca: Pca) -> torch.Tensor:
    """Map reduced codes ``[..., k]`` back to the original latent space ``[..., D]``."""
    dev = reduced.device
    z = reduced * pca["sv"].to(dev) if pca["whiten"] else reduced
    xs = z @ pca["components"].to(dev)          # [..., D]
    return xs * pca["std"].to(dev) + pca["mean"].to(dev)


def pca_reduce_splits(
    train: torch.Tensor,
    val: torch.Tensor,
    k: int,
    *,
    whiten: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Pca]:
    """Fit a PCA on ``train`` and project both splits into its ``k``-dim basis.

    This is the "just change the dataset" path: the reduction is a **fixed, unlearned
    basis** applied to the data once, so a model can consume the reduced latents with
    no input/output projection at all (``WorldModel(io_projection=False)``). Contrast
    the learned ``in_proj``/``out_proj``, which put most of a small model's parameters
    into mapping in and out of the 1024-D encoder space.

    Fit is on **train only** -- the val split is projected with the train basis, so
    there is no leakage, matching the standardisation convention used elsewhere.

    Because the PCA centres the data (and, with ``whiten=True``, sets every component
    to unit variance), it *replaces* the per-channel standardisation rather than
    composing with it; applying both would rescale the components a second time.

    Returns ``(train_reduced, val_reduced, pca)``; ``pca["explained"]`` is the
    fraction of variance the ``k`` kept components carry, which is the honest cost of
    the reduction and should be reported with any result measured in this space.
    """
    pca = fit_pca(train, k, whiten=whiten)
    return apply_pca(train, pca), apply_pca(val, pca), pca
