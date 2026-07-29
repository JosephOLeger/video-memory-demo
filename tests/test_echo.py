"""The [V;V]/[V;W] echo benchmark.

This module produces the project's headline memory metric, so the tests guard the
parts whose silent breakage would invalidate a whole sweep: the doubling/scoring
geometry, that the PCA path really removes the learned projections, and that
pointing at a PCA cache is detected from the data rather than trusted to a flag.
"""

import json

import pytest
import torch

from titans.worldmodel.echo import (
    build_parser, double, per_copy_l1_K, prepare_data, run,
)
from titans.worldmodel.make_pca_cache import build


def _latents(n=6, t=16, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(8, d, generator=g)
    coef = torch.randn(n, t, 8, generator=g)
    return coef @ basis + 0.05 * torch.randn(n, t, d, generator=g)


def _args(train, val, **over):
    a = build_parser().parse_args(
        ["--label", "t", "--train", str(train), "--val", str(val),
         "--horizon", "4", "--steps", "3", "--n-eval", "4", "--dim", "16",
         "--batch", "2", "--seg", "4"]
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_double_builds_VV_and_VW():
    v = torch.arange(2 * 3 * 2).float().reshape(2, 3, 2)
    w = torch.zeros_like(v)
    vv, vw = double(v), double(v, w)
    assert vv.shape == (2, 6, 2) and vw.shape == (2, 6, 2)
    assert torch.equal(vv[:, :3], v) and torch.equal(vv[:, 3:], v)
    assert torch.equal(vw[:, 3:], w)


def test_per_copy_scores_the_two_halves_separately():
    """A model that is perfect on the 1st copy and wrong on the 2nd must show it."""
    T, K, margin = 8, 2, 1

    class Fake(torch.nn.Module):
        blocks: list = []

        def forward(self, z):
            out = z.roll(-K, dims=1).clone()   # exact K-step prediction
            out[:, T:] += 1.0                  # ...but corrupted on the 2nd copy
            return out

    seq = torch.randn(2, 2 * T, 3)
    first, second = per_copy_l1_K(Fake(), seq, T, K, reset=False, margin=margin)
    assert first == pytest.approx(0.0, abs=1e-5)
    assert second == pytest.approx(1.0, abs=1e-5)


def test_prepare_data_standardizes_a_raw_cache(tmp_path):
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": _latents() * 50 + 10}, tp)
    torch.save({"latents": _latents(seed=1) * 50 + 10}, vp)
    tr, va, space = prepare_data(tp, vp)
    assert space["kind"] == "standardized"
    assert tr.mean().abs() < 0.1 and abs(tr.std() - 1.0) < 0.2


def test_prepare_data_detects_a_pca_cache(tmp_path):
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": _latents(n=8, d=48)}, tp)
    torch.save({"latents": _latents(n=4, d=48, seed=1)}, vp)
    info = build(tp, vp, k=8, out_prefix=tmp_path / "red")
    tr, va, space = prepare_data(info["train"], info["val"])
    assert space["kind"] == "pca" and space["k"] == 8
    assert tr.shape[-1] == 8 and va.shape[-1] == 8


def test_pca_run_has_no_learned_io_and_dim_follows_k(tmp_path):
    """The whole point: on a PCA cache, dim = k and the I/O projections are gone."""
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": _latents(n=8, t=16, d=48)}, tp)
    torch.save({"latents": _latents(n=4, t=16, d=48, seed=1)}, vp)
    info = build(tp, vp, k=8, out_prefix=tmp_path / "red")

    r = run(_args(info["train"], info["val"], dim=999), "cpu")
    assert r["io_projection"] is False
    assert r["dim"] == 8          # --dim 999 is ignored; k wins
    assert r["space"]["kind"] == "pca"
    for v in (r["memory_gain"], r["mem_effect"], r["pr"]):
        assert v == v             # not NaN


def test_raw_run_keeps_the_learned_io(tmp_path):
    """Non-PCA path is unchanged, so earlier sweep cells stay comparable."""
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": _latents(n=8, t=16, d=48)}, tp)
    torch.save({"latents": _latents(n=4, t=16, d=48, seed=1)}, vp)
    r = run(_args(tp, vp), "cpu")
    assert r["io_projection"] is True and r["dim"] == 16
    assert r["space"]["kind"] == "standardized"


def test_result_is_json_serialisable(tmp_path):
    """Results are written as JSON by the sweep; a non-serialisable field breaks it."""
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": _latents(n=8, t=16, d=48)}, tp)
    torch.save({"latents": _latents(n=4, t=16, d=48, seed=1)}, vp)
    r = run(_args(tp, vp), "cpu")
    assert json.loads(json.dumps(r))["label"] == "t"


def test_prefix_swap_scores_identical_content():
    """[V1;V1] and [V2;V1] must be scored on the very same frames.

    This is the whole point of the prefix-swap probe: the earlier metric compared
    halves holding *different* content, so its gap confounded "the memory helps"
    with "one video is harder than the other".
    """
    va = _latents(n=4, t=8, d=6, seed=11)
    other = torch.roll(va, 1, 0)
    vv, cross = double(va), double(other, va)
    T = va.shape[1]
    assert torch.equal(vv[:, T:], cross[:, T:])       # same second half
    assert not torch.equal(vv[:, :T], cross[:, :T])   # different prefix
    # and the prefix really is another video, not a shift of the same one
    assert torch.equal(cross[:, :T], torch.roll(va, 1, 0))


def test_memory_gain_is_zero_when_the_prefix_cannot_matter():
    """A memoryless predictor must score identically under both prefixes."""
    T, K, margin = 8, 2, 1

    class Memoryless(torch.nn.Module):
        blocks: list = []

        def forward(self, z):
            return z.roll(-K, dims=1)     # depends only on the local window

    va = _latents(n=4, t=T, d=5, seed=12)
    other = torch.roll(va, 1, 0)
    m = Memoryless()
    _, rep = per_copy_l1_K(m, double(va), T, K, reset=False, margin=margin)
    _, cro = per_copy_l1_K(m, double(other, va), T, K, reset=False, margin=margin)
    assert rep == pytest.approx(cro, abs=1e-6)        # memory_gain == 0


def test_run_reports_the_prefix_swap_fields(tmp_path):
    tp, vp = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save({"latents": _latents(n=8, t=16, d=48)}, tp)
    torch.save({"latents": _latents(n=4, t=16, d=48, seed=1)}, vp)
    r = run(_args(tp, vp), "cpu")
    assert r["second_repeat"] == r["second"]
    assert r["memory_gain"] == pytest.approx(r["second_cross"] - r["second_repeat"])
    assert r["second_cross"] == r["second_cross"]     # not NaN
