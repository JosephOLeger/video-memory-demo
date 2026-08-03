"""Sharded reading and the streaming PCA.

The streaming PCA exists so a 19 GB corpus never has to be materialised, but that
is only worth anything if it gives the *same* basis as the direct fit. These tests
pin that equivalence, plus the properties a sweep would silently lose: that the
basis is fitted on train only, and that the written cache is recognisable as
pre-reduced.
"""

import torch

from titans.worldmodel.bottleneck import apply_pca, fit_pca
from titans.worldmodel.sharded import (
    fit_pca_sharded, iter_shards, shard_paths, shard_summary, write_reduced_cache,
)


def _write_shards(tmp_path, split, n_shards=3, per=5, T=8, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(6, D, generator=g)
    all_z = []
    for i in range(n_shards):
        coef = torch.randn(per, T, 6, generator=g)
        z = coef @ basis + 0.05 * torch.randn(per, T, D, generator=g)
        segs = [f"P{split[0].upper()}{i}/DAY1/s{j}.mp4" for j in range(per)]
        torch.save({"latents": z, "segments": segs}, tmp_path / f"{split}_{i:04d}.pt")
        all_z.append(z)
    return torch.cat(all_z, 0)


def test_iteration_order_and_summary(tmp_path):
    z = _write_shards(tmp_path, "train")
    assert len(shard_paths(tmp_path, "train")) == 3
    seen = torch.cat([s for s, _ in iter_shards(tmp_path, "train")], 0)
    assert torch.equal(seen, z)          # sorted order == concatenation order
    s = shard_summary(tmp_path, "train")
    assert s["shards"] == 3 and s["sequences"] == 15 and s["latents"] == 120


def test_streaming_pca_matches_the_direct_fit(tmp_path):
    """The whole point: exact, not approximate."""
    z = _write_shards(tmp_path, "train", n_shards=4, per=20, T=8, D=16)
    k = 6
    a = fit_pca_sharded(tmp_path, "train", k)
    b = fit_pca(z, k)
    assert torch.allclose(a["mean"], b["mean"], atol=1e-4)
    assert torch.allclose(a["std"], b["std"], atol=1e-4)
    assert torch.allclose(a["sv"], b["sv"], rtol=1e-3, atol=1e-5)
    assert abs(a["explained"] - b["explained"]) < 1e-4
    # components match up to a per-component sign flip
    for i in range(k):
        ca, cb = a["components"][i], b["components"][i]
        assert min((ca - cb).abs().max(), (ca + cb).abs().max()) < 1e-3
    # and the projections agree up to that same sign
    pa, pb = apply_pca(z, a), apply_pca(z, b)
    assert torch.allclose(pa.abs(), pb.abs(), atol=1e-3)


def test_whitened_components_have_unit_scale(tmp_path):
    z = _write_shards(tmp_path, "train", n_shards=4, per=40, T=8, D=16)
    pca = fit_pca_sharded(tmp_path, "train", 6, whiten=True)
    s = apply_pca(z, pca).reshape(-1, 6).std(0)
    assert torch.allclose(s, torch.ones_like(s), atol=0.15)


def test_basis_is_fitted_on_train_only(tmp_path):
    """Val must be projected with the train basis -- otherwise the split leaks."""
    _write_shards(tmp_path, "train", seed=0)
    _write_shards(tmp_path, "val", seed=99)
    a = fit_pca_sharded(tmp_path, "train", 4)
    b = fit_pca_sharded(tmp_path, "val", 4)
    assert not torch.allclose(a["components"], b["components"], atol=1e-3)


def test_written_cache_is_detected_as_pre_reduced(tmp_path):
    from titans.worldmodel.train import load_latent_cache

    _write_shards(tmp_path, "train", n_shards=3, per=5, T=8, D=16)
    pca = fit_pca_sharded(tmp_path, "train", 4)
    out = tmp_path / "red_train.pt"
    info = write_reduced_cache(tmp_path, "train", pca, out)
    assert info["shape"] == (15, 8, 4)
    z, meta = load_latent_cache(out)
    assert z.shape[-1] == 4
    assert "pca" in meta and meta["k"] == 4        # what the harness keys on
    assert len(meta["segments"]) == 15
