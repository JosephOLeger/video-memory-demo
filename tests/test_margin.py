"""The stability-margin probe: it must read the model the memory actually sees."""

import pytest
import torch

from titans.worldmodel.frozen import build_cfg, load_frozen, save_frozen
from titans.worldmodel.margin import THRESHOLD, measure, memory_input_stream
from titans.worldmodel.model import WorldModel


def _artifact(tmp_path, name="f.pt", **kw):
    cfg = build_cfg("sane_ns", **kw)
    torch.manual_seed(0)
    m = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4, memory_num_layers=2,
                   **cfg)
    p = save_frozen(m, cfg, str(tmp_path / name), meta={"label": "t", "seed": 3})
    return load_frozen(p)


def _z(n=4, t=16, d=8):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, t, d, generator=g)


def test_stream_is_the_attention_output_not_the_latents(tmp_path):
    """The memory sees y_i, which depends on its own retrieval -- probing the raw
    latents would measure a curvature it never experiences."""
    model, _ = _artifact(tmp_path)
    z = _z()
    x = memory_input_stream(model, z)
    assert x.shape == z.shape                 # same [B, T, dim]...
    assert not torch.allclose(x, z)           # ...but not the same tensor


def test_spy_is_removed_even_though_it_wrapped_a_bound_method(tmp_path):
    """A leaked spy would make every later forward accumulate into a stale list."""
    model, _ = _artifact(tmp_path)
    mem = model.blocks[0].memory
    before = mem.update
    memory_input_stream(model, _z())
    assert mem.update == before


def test_margin_is_the_threshold_over_the_worst_chunk(tmp_path):
    model, info = _artifact(tmp_path)
    r = measure(model, _z(), info)
    assert r["margin"] == pytest.approx(THRESHOLD / r["theta_lambda_hidden_max"])
    assert r["theta_lambda_hidden_max"] >= r["theta_lambda_hidden_mean"]
    assert r["all_finite"] and r["chunks"] > 0


def test_health_is_reported_alongside_so_a_dead_memory_cannot_pass(tmp_path):
    """A huge margin reached by having no memory left is not stability."""
    model, info = _artifact(tmp_path)
    r = measure(model, _z(), info)
    assert r["pr"] >= 1.0
    assert 0.0 <= abs(r["cos_ret"]) <= 1.0 + 1e-6
    assert r["w1_norm"] > 0.0


def test_config_travels_into_the_record(tmp_path):
    """The record must be self-describing: it is joined to echo results by arm."""
    model, info = _artifact(tmp_path, cascade=True, leak="surprise", leak_rate=0.3)
    r = measure(model, _z(), info)
    assert r["cascade"] is True
    assert r["leak"] == "surprise" and r["leak_rate"] == 0.3
    assert r["label"] == "t" and r["seed"] == 3


def test_a_larger_theta_scale_shrinks_the_margin(tmp_path):
    """Sanity: the margin must move the right way with the one knob that sets it."""
    lo, ilo = _artifact(tmp_path, "lo.pt", theta_scale=0.03)
    hi, ihi = _artifact(tmp_path, "hi.pt", theta_scale=0.30)
    z = _z()
    assert measure(hi, z, ihi)["margin"] < measure(lo, z, ilo)["margin"]
