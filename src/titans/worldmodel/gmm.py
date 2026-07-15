"""Gaussian-mixture (MDN) output head for the world model.

Motivation
----------
The default world model predicts one latent per step and is trained with L1
(:func:`titans.worldmodel.train.next_step_loss`). L1 (like L2) assumes a *unimodal*
target: when the future is genuinely multimodal -- several plausible next latents --
its minimiser is the (conditional) average, so the model is pushed to predict a blur
*between* the modes rather than committing to any of them. That blurring also makes
the loss nearly blind to whatever the long-term memory contributes, since the memory
mostly helps by *disambiguating which mode* the sequence is in.

A Mixture Density Network (Bishop, 1994) instead predicts a full conditional
distribution ``p(z_{t+1} | z_{<=t})`` as a mixture of ``K`` diagonal Gaussians and is
trained by negative log-likelihood. Separate components can sit on distinct futures
instead of averaging them, and the NLL rewards reducing predictive *uncertainty* --
which is exactly what better memory should buy.

Why this is well-posed here: the V-JEPA-2 encoder is **frozen**, so the target
latents are a fixed continuous space. This is plain density estimation over fixed
targets -- there is no jointly-learned target encoder to collapse (the failure mode a
JEPA-style *training* objective has to guard against). See the project notes.

Design
------
* **Diagonal covariance** per component. A full covariance in ``D=1024`` dims is
  intractable (``D^2`` params/component); a diagonal Gaussian factorises over
  channels, and the *mixture* still captures cross-channel structure by placing
  different joint modes. This is the standard MDN trade-off.
* **Numerical stability**: mixture weights via ``log_softmax``; per-channel
  ``log_sigma`` is **clamped** to ``[min_log_sigma, max_log_sigma]`` -- the floor
  stops a component collapsing onto a memorised point (variance -> 0 => NLL -> -inf),
  the cap stops it blowing up. NLL uses ``logsumexp`` over components.
* The head maps the model's internal features ``[..., dim]`` (the pre-output-proj
  representation, obtained via ``WorldModel(z, return_features=True)``) to the mixture
  parameters, so it is a drop-in replacement for the linear ``out_proj`` and leaves
  the L1 path untouched.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)


class GMMHead(nn.Module):
    """Diagonal-covariance Gaussian-mixture head.

    Maps features ``[..., dim]`` to the parameters of a ``K``-component mixture over
    ``out_dim`` target channels::

        logits    [..., K]        mixture logits   (-> log_softmax for log-weights)
        means     [..., K, D]     component means
        log_sigma [..., K, D]     per-channel log std-devs (clamped for stability)

    Parameters
    ----------
    dim: input feature width (the world model's internal ``dim``).
    out_dim: ``D``, the target latent dimension.
    num_components: ``K``, number of mixture components.
    min_log_sigma, max_log_sigma: clamp range for ``log_sigma`` (variance floor/cap).
        Defaults suit standardised latents (per-channel unit variance): floor
        ``exp(-7)~=9e-4``, cap ``exp(3)~=20``.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_components: int = 5,
        min_log_sigma: float = -7.0,
        max_log_sigma: float = 3.0,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components (K) must be >= 1")
        if min_log_sigma >= max_log_sigma:
            raise ValueError("min_log_sigma must be < max_log_sigma")
        self.dim = dim
        self.out_dim = out_dim
        self.num_components = num_components
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # K logits + K*D means + K*D log_sigma
        self.proj = nn.Linear(dim, num_components * (1 + 2 * out_dim))

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``feat`` ``[..., dim]`` -> ``(logits [...,K], means [...,K,D], log_sigma [...,K,D])``."""
        K, D = self.num_components, self.out_dim
        o = self.proj(feat)
        logits = o[..., :K]
        rest = o[..., K:].reshape(*o.shape[:-1], K, 2 * D)
        means = rest[..., :D]
        log_sigma = rest[..., D:].clamp(self.min_log_sigma, self.max_log_sigma)
        return logits, means, log_sigma


def gmm_log_prob(
    logits: torch.Tensor,
    means: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-example log-likelihood ``log p(target)`` under the mixture.

    Shapes: ``logits [...,K]``, ``means``/``log_sigma`` ``[...,K,D]``,
    ``target [...,D]`` -> returns ``[...]``.
    """
    log_pi = F.log_softmax(logits, dim=-1)                 # [...,K]
    x = target.unsqueeze(-2)                               # [...,1,D]
    # log N(x; mu, sigma^2) per channel, summed over D:
    #   -0.5 * [ (x-mu)^2 / sigma^2 + 2*log_sigma + log(2*pi) ]
    inv_var = torch.exp(-2.0 * log_sigma)                  # [...,K,D]
    log_comp = -0.5 * (
        (x - means) ** 2 * inv_var + 2.0 * log_sigma + _LOG_2PI
    ).sum(dim=-1)                                          # [...,K]
    return torch.logsumexp(log_pi + log_comp, dim=-1)      # [...]


def gmm_nll(
    logits: torch.Tensor,
    means: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean negative log-likelihood of ``target`` under the mixture (a scalar loss)."""
    return -gmm_log_prob(logits, means, log_sigma, target).mean()


def mixture_mean(logits: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """MMSE point estimate ``E[z] = sum_k pi_k mu_k`` -> ``[...,D]``.

    Note: under a genuinely multimodal posterior this can land *between* modes (the
    same averaging L1 suffers from); use :func:`dominant_mean` for a committed,
    single-mode point prediction.
    """
    pi = F.softmax(logits, dim=-1).unsqueeze(-1)           # [...,K,1]
    return (pi * means).sum(dim=-2)                        # [...,D]


def dominant_mean(logits: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Mean of the highest-weight component (a committed, non-averaged prediction) -> ``[...,D]``."""
    k = logits.argmax(dim=-1)                              # [...]
    idx = k.unsqueeze(-1).unsqueeze(-1).expand(*k.shape, 1, means.shape[-1])
    return means.gather(-2, idx).squeeze(-2)               # [...,D]
