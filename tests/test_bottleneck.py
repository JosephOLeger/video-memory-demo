"""Tests for the PCA latent bottleneck."""

import torch

from titans.worldmodel.bottleneck import fit_pca, apply_pca, inverse_pca


def _lowrank(N=500, D=32, r=6, noise=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, r, generator=g)
    B = torch.randn(r, D, generator=g)
    return A @ B + noise * torch.randn(N, D, generator=g)


def test_shapes_and_explained():
    X = _lowrank()
    pca = fit_pca(X, k=6)
    assert pca["components"].shape == (6, 32)
    assert apply_pca(X, pca).shape == (500, 6)
    assert 0.0 < pca["explained"] <= 1.0 + 1e-6
    # rank-6 data -> 6 components capture almost all variance
    assert pca["explained"] > 0.99


def test_whitening_unit_variance():
    X = _lowrank()
    z = apply_pca(X, fit_pca(X, k=6, whiten=True))
    assert torch.allclose(z.std(0), torch.ones(6), atol=0.1)


def test_reconstruction_improves_with_k():
    X = _lowrank(r=10)
    errs = []
    for k in (2, 6, 12):
        pca = fit_pca(X, k=k)
        rec = inverse_pca(apply_pca(X, pca), pca)
        errs.append((rec - X).pow(2).mean().item())
    assert errs[0] > errs[1] > errs[2]


def test_preserves_batch_dims():
    X = torch.randn(4, 16, 32)  # [B, T, D]
    pca = fit_pca(X, k=8)
    assert apply_pca(X, pca).shape == (4, 16, 8)
    assert inverse_pca(apply_pca(X, pca), pca).shape == (4, 16, 32)


def test_full_rank_roundtrip():
    X = _lowrank(D=16, r=16, noise=0.5)
    pca = fit_pca(X, k=16, whiten=True)
    rec = inverse_pca(apply_pca(X, pca), pca)
    assert torch.allclose(rec, X, atol=1e-3)
