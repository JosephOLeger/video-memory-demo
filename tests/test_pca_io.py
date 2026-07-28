"""PCA-reduced latents with no learned I/O projection.

The point of `io_projection=False` is that the dimensionality reduction becomes a
property of the *dataset* (a fixed PCA basis) rather than a learned layer, so these
tests guard the two things that would silently break that contract: that no learned
input/output map survives, and that the reduction is fitted on train only.
"""

import pytest
import torch

from titans.worldmodel.bottleneck import apply_pca, fit_pca, pca_reduce_splits
from titans.worldmodel.model import WorldModel


def _latents(n=6, t=16, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    # low-rank + noise, so a PCA actually has something to keep
    basis = torch.randn(8, d, generator=g)
    coef = torch.randn(n, t, 8, generator=g)
    return coef @ basis + 0.05 * torch.randn(n, t, d, generator=g)


def test_no_learned_io_when_disabled():
    m = WorldModel(latent_dim=16, dim=16, depth=1, segment_size=4, io_projection=False)
    assert isinstance(m.in_proj, torch.nn.Identity)
    assert isinstance(m.out_proj, torch.nn.Identity)
    assert isinstance(m.out_norm, torch.nn.Identity)
    named = {n for n, _ in m.named_parameters()}
    assert not any(n.startswith(("in_proj", "out_proj.", "out_norm")) for n in named)


def test_dim_mismatch_is_rejected():
    with pytest.raises(ValueError, match="latent_dim == dim"):
        WorldModel(latent_dim=64, dim=16, depth=1, segment_size=4, io_projection=False)


def test_forward_shape_and_trainability_without_io():
    k = 16
    z = _latents(n=4, t=8, d=k)
    m = WorldModel(latent_dim=k, dim=k, depth=1, segment_size=4, io_projection=False)
    out = m(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()
    torch.nn.functional.l1_loss(out[:, :-1], z[:, 1:]).backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_pca_reduce_splits_is_fitted_on_train_only():
    tr, va = _latents(seed=0), _latents(seed=1)
    k = 8
    tr_r, va_r, pca = pca_reduce_splits(tr, va, k)
    assert tr_r.shape == (*tr.shape[:-1], k)
    assert va_r.shape == (*va.shape[:-1], k)
    ref = fit_pca(tr, k)
    assert torch.allclose(pca["components"], ref["components"])
    # val is projected with the train basis, not one fitted on itself
    assert torch.allclose(va_r, apply_pca(va, ref), atol=1e-5)


def test_whitened_components_have_unit_scale():
    tr, va = _latents(n=40, t=16, seed=2), _latents(n=8, seed=3)
    tr_r, _, pca = pca_reduce_splits(tr, va, k=8, whiten=True)
    s = tr_r.reshape(-1, 8).std(0)
    assert torch.allclose(s, torch.ones_like(s), atol=0.15)
    assert 0.0 < pca["explained"] <= 1.0


def test_end_to_end_pca_then_model():
    """The intended pipeline: reduce to k, then run a model whose dim IS k."""
    k = 16
    tr, va = _latents(n=8, t=8, d=48, seed=4), _latents(n=4, t=8, d=48, seed=5)
    tr_r, va_r, _ = pca_reduce_splits(tr, va, k)
    m = WorldModel(latent_dim=k, dim=k, depth=1, segment_size=4, io_projection=False)
    assert m(va_r).shape == va_r.shape


def test_io_projection_default_unchanged():
    """Default path keeps the learned projections, so old results stay comparable."""
    m = WorldModel(latent_dim=48, dim=16, depth=1, segment_size=4)
    assert isinstance(m.in_proj, torch.nn.Linear)
    assert isinstance(m.out_proj, torch.nn.Linear)
    z = _latents(n=2, t=8, d=48)
    assert m(z).shape == z.shape


def test_prereduced_cache_is_autodetected(tmp_path):
    """A cache written by make_pca_cache must not be standardised or re-projected."""
    from titans.worldmodel.make_pca_cache import build
    from titans.worldmodel.harness import run_config

    tr, va = _latents(n=8, t=8, d=48, seed=6), _latents(n=4, t=8, d=48, seed=7)
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": tr}, tp)
    torch.save({"latents": va}, vp)
    info = build(tp, vp, k=8, out_prefix=tmp_path / "red")

    r = run_config(info["train"], info["val"], seeds=[0], steps=2,
                   depth=1, segment_size=4, batch_size=2, device="cpu")
    # the harness recognised the reduction and configured the model for it
    assert r["pca"]["prereduced"] is True and r["pca"]["k"] == 8
    assert r["config"]["dim"] == 8 and r["config"]["io_projection"] is False
    assert r["val_l1_mean"] == r["val_l1_mean"]  # finite, not NaN
