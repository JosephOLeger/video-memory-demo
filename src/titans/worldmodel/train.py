"""Training utilities and a runnable demo for the Titans world model (Phase 3).

The loss is a plain L1 (mean absolute error) between the predicted next latent
and the true next latent -- the "outer loop" that trains the model's parameters,
distinct from the memory's own test-time "inner loop" learning.

Run the self-contained demo (no dataset download, no video encoder required)::

    python -m titans.worldmodel.train

It trains the model on a batch of smooth synthetic latent trajectories and prints
the loss going down, which is enough to confirm the whole stack is wired up and
trainable end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .model import WorldModel


def load_latents(path: str | Path) -> torch.Tensor:
    """Load a precomputed latent tensor ``[num_videos, T, latent_dim]`` from disk.

    Accepts either a bare tensor or the dict saved by
    :func:`titans.worldmodel.precompute_latents.precompute`.
    """
    obj = torch.load(path, weights_only=False)
    return obj["latents"] if isinstance(obj, dict) else obj


def latent_cache_meta(path: str | Path) -> dict:
    """Metadata stored alongside a cached latent tensor (``{}`` for a bare tensor).

    Used to recognise a cache that has already been PCA-reduced by
    :mod:`titans.worldmodel.make_pca_cache` -- such a cache carries a ``pca`` key,
    and re-standardising or re-projecting it would silently undo the reduction it
    was built to provide.
    """
    obj = torch.load(path, weights_only=False)
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items() if k != "latents"}


def standardize_latents(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-channel standardisation (mean 0, unit variance) of latents ``[B, T, d]``.

    Frozen encoder features (V-JEPA-2 here) come out with large, uneven per-channel
    magnitudes; feeding them raw makes the memory's inner-loop update explode to
    ``NaN``. Centring/scaling each of the ``d`` channels over the ``(B, T)`` samples
    keeps both the outer optimiser and the test-time memory update well-behaved,
    and leaves the L1 next-step objective in a sane, comparable range.
    """
    mean = z.mean(dim=(0, 1), keepdim=True)
    std = z.std(dim=(0, 1), keepdim=True)
    return (z - mean) / (std + eps)


def next_step_loss(model: WorldModel, z: torch.Tensor) -> torch.Tensor:
    """L1 loss for next-latent prediction over a batch ``z`` ``[B, T, latent_dim]``.

    Position ``t`` of the prediction targets ``z[:, t + 1]``, so the last step
    (which has no successor) is dropped from both sides.
    """
    pred = model(z)
    return F.l1_loss(pred[:, :-1], z[:, 1:])


def synthetic_latents(
    batch: int = 4,
    steps: int = 16,
    latent_dim: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Smooth random latent trajectories, a stand-in for encoder outputs.

    Each trajectory is a slow random walk (cumulative sum of small steps), so
    consecutive latents are correlated and next-step prediction is learnable --
    unlike pure i.i.d. noise. Shape ``[batch, steps, latent_dim]``.
    """
    increments = 0.1 * torch.randn(
        batch, steps, latent_dim, generator=generator
    )
    return increments.cumsum(dim=1)


def train_demo(
    steps: int = 200,
    seq_len: int = 16,
    latent_dim: int = 32,
    dim: int = 64,
    depth: int = 2,
    segment_size: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    latents: Optional[torch.Tensor] = None,
    standardize: bool = True,
) -> WorldModel:
    """Train on one fixed batch of latents; print loss periodically.

    By default the batch is synthetic (``synthetic_latents``); pass ``latents``
    ``[B, T, latent_dim]`` (e.g. from :func:`load_latents`) to train on real
    precomputed V-JEPA-2 features instead, in which case ``latent_dim`` and
    ``seq_len`` are taken from the tensor. Real latents are per-channel
    standardised by default (``standardize``) -- raw encoder features otherwise
    drive the memory's inner loop to ``NaN``. Overfitting one batch is
    intentional: it is the quickest sanity check that gradients flow and the model
    can fit next-step structure. Returns the trained model.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    if latents is not None:
        z = standardize_latents(latents) if standardize else latents
        latent_dim = z.shape[-1]
    else:
        z = synthetic_latents(
            batch=4, steps=seq_len, latent_dim=latent_dim, generator=gen
        )
    model = WorldModel(
        latent_dim=latent_dim,
        dim=dim,
        depth=depth,
        segment_size=segment_size,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = next_step_loss(model, z)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0:
            print(f"step {step:4d}  L1 = {loss.item():.4f}")

    return model


if __name__ == "__main__":
    train_demo()
