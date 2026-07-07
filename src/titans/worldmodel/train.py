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

import torch
import torch.nn.functional as F

from .model import WorldModel


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
) -> WorldModel:
    """Train on one fixed batch of synthetic latents; print loss periodically.

    Overfitting a single batch is intentional here: it is the quickest sanity
    check that gradients flow and the model can fit next-step structure. Returns
    the trained model.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

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
