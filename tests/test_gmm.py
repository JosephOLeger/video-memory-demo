"""Tests for the Gaussian-mixture (MDN) output head."""

import math

import torch
import torch.nn.functional as F

from titans.worldmodel.gmm import (
    GMMHead,
    dominant_mean,
    gmm_log_prob,
    gmm_nll,
    mixture_mean,
)
from titans.worldmodel.model import WorldModel


def test_head_shapes():
    B, T, dim, D, K = 2, 5, 16, 8, 4
    head = GMMHead(dim, D, num_components=K)
    logits, means, log_sigma = head(torch.randn(B, T, dim))
    assert logits.shape == (B, T, K)
    assert means.shape == (B, T, K, D)
    assert log_sigma.shape == (B, T, K, D)
    # log_sigma is clamped into range
    assert log_sigma.min() >= head.min_log_sigma - 1e-6
    assert log_sigma.max() <= head.max_log_sigma + 1e-6


def test_nll_finite_and_differentiable():
    torch.manual_seed(0)
    head = GMMHead(16, 8, num_components=3)
    feat = torch.randn(4, 16, requires_grad=True)
    target = torch.randn(4, 8)
    logits, means, log_sigma = head(feat)
    loss = gmm_nll(logits, means, log_sigma, target)
    assert torch.isfinite(loss)
    loss.backward()
    # gradient reaches the head and the input features
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_single_gaussian_matches_closed_form():
    # K=1 mixture NLL must equal the plain diagonal-Gaussian NLL.
    torch.manual_seed(1)
    D = 5
    means = torch.randn(3, 1, D)
    log_sigma = torch.randn(3, 1, D) * 0.3
    logits = torch.zeros(3, 1)
    target = torch.randn(3, D)
    lp = gmm_log_prob(logits, means, log_sigma, target)  # [3]
    var = torch.exp(2 * log_sigma[:, 0])
    ref = -0.5 * (((target - means[:, 0]) ** 2 / var) + 2 * log_sigma[:, 0]
                  + math.log(2 * math.pi)).sum(-1)
    assert torch.allclose(lp, ref, atol=1e-5)


def test_nll_prefers_the_correct_mode():
    # A 2-component mixture whose components sit on two distinct targets should
    # assign higher likelihood to those targets than a single averaged Gaussian.
    D = 4
    m0, m1 = torch.full((D,), -2.0), torch.full((D,), 2.0)
    means = torch.stack([m0, m1]).view(1, 2, D)
    log_sigma = torch.full((1, 2, D), math.log(0.5))
    logits = torch.zeros(1, 2)
    # likelihood at a real mode vs at the blurred average (0):
    ll_mode = gmm_log_prob(logits, means, log_sigma, m1.view(1, D))
    ll_avg = gmm_log_prob(logits, means, log_sigma, torch.zeros(1, D))
    assert (ll_mode > ll_avg).all()


def test_point_estimates():
    D, K = 3, 3
    means = torch.tensor([[[1.0, 1, 1], [2, 2, 2], [3, 3, 3]]])  # [1,K,D]
    logits = torch.tensor([[0.0, 10.0, 0.0]])  # component 1 dominates
    dm = dominant_mean(logits, means)
    assert torch.allclose(dm, means[:, 1])
    mm = mixture_mean(logits, means)
    # mixture mean is a convex combo -> within the components' range
    assert (mm >= means.min()).all() and (mm <= means.max()).all()


def test_worldmodel_return_features():
    torch.manual_seed(0)
    B, T, D, dim = 2, 16, 10, 32
    z = torch.randn(B, T, D)
    model = WorldModel(latent_dim=D, dim=dim, depth=2, segment_size=8)
    feat = model(z, return_features=True)
    assert feat.shape == (B, T, dim)
    # default path is unchanged: prediction has the latent shape
    pred = model(z)
    assert pred.shape == (B, T, D)


def test_gmm_head_overfits_tiny_batch():
    # Wiring sanity: WorldModel features + GMM head + NLL is trainable end to end.
    torch.manual_seed(0)
    B, T, D, dim = 2, 16, 6, 32
    z = torch.randn(B, T, D).cumsum(dim=1) * 0.3  # smooth-ish so it is learnable
    model = WorldModel(latent_dim=D, dim=dim, depth=2, segment_size=8)
    head = GMMHead(dim, D, num_components=3)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        feat = model(z, return_features=True)
        logits, means, log_sigma = head(feat[:, :-1])
        loss = gmm_nll(logits, means, log_sigma, z[:, 1:])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]  # NLL goes down
