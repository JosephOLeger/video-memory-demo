"""The frozen artifact: config travels with the weights, and loading is strict."""

import pytest
import torch

from titans.worldmodel.frozen import build_cfg, load_frozen, save_frozen
from titans.worldmodel.model import WorldModel


def _model(**cfg):
    torch.manual_seed(0)
    return WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                      memory_num_layers=2, **cfg)


def test_roundtrip_reproduces_the_architecture_and_the_outputs(tmp_path):
    cfg = build_cfg("sane_ns")
    m = _model(**cfg)
    p = save_frozen(m, cfg, str(tmp_path / "f.pt"), meta={"label": "x"})

    back, info = load_frozen(p)
    z = torch.randn(2, 8, 8)
    m.eval()
    with torch.no_grad():
        ref = m(z)
    # Not bit-exact: the loaded model runs with grad disabled everywhere, and the
    # inner loop's vmap'd gradients pick different kernels with and without autograd
    # (measured ~3e-6 here). A config mismatch would be orders of magnitude larger,
    # and the strict load already rules out a different parameter set.
    assert torch.allclose(ref, back(z), atol=1e-4, rtol=1e-4)
    assert info["meta"]["label"] == "x"
    assert info["arch"] == {"dim": 8, "depth": 1, "segment_size": 4}


def test_loaded_model_is_frozen_and_in_eval(tmp_path):
    cfg = build_cfg("sane_ns")
    p = save_frozen(_model(**cfg), cfg, str(tmp_path / "f.pt"))
    back, _ = load_frozen(p)
    assert not back.training
    assert not any(q.requires_grad for q in back.parameters())


def test_cascade_config_survives_the_roundtrip(tmp_path):
    """The six cascade flags are exactly what a plain checkpoint cannot carry."""
    cfg = build_cfg("sane_ns", cascade=True, reservoirs=5, leak="surprise",
                    leak_rate=0.01, readout="gated", selective=True)
    m = _model(**cfg)
    p = save_frozen(m, cfg, str(tmp_path / "c.pt"))
    back, info = load_frozen(p)
    assert info["cfg"]["memory_use_cascade"] is True
    assert info["cfg"]["cascade_reservoirs"] == 5
    assert info["cfg"]["cascade_leak"] == "surprise"
    assert info["cfg"]["cascade_leak_rate"] == 0.01
    assert info["cfg"]["cascade_selective"] is True
    assert back.blocks[0].memory.use_cascade_momentum


def test_a_plain_checkpoint_is_refused_with_a_pointer(tmp_path):
    p = tmp_path / "plain.pt"
    torch.save({"model": _model().state_dict(), "step": 10}, p)
    with pytest.raises(ValueError, match="not a frozen artifact"):
        load_frozen(str(p))


def test_wrong_config_fails_loudly_rather_than_loading_partially(tmp_path):
    """The failure mode the artifact exists to prevent, checked directly."""
    cfg = build_cfg("sane_ns", cascade=True, reservoirs=5)
    p = save_frozen(_model(**cfg), cfg, str(tmp_path / "c.pt"))
    blob = torch.load(p, map_location="cpu", weights_only=False)
    blob["cfg"] = build_cfg("sane_ns")           # cascade silently dropped
    torch.save(blob, p)
    with pytest.raises(RuntimeError):
        load_frozen(str(p))


def test_build_cfg_leaves_the_baseline_untouched_when_nothing_is_asked():
    from titans.worldmodel.atlas import CONFIGS
    assert build_cfg("sane_ns") == dict(CONFIGS["sane_ns"])
