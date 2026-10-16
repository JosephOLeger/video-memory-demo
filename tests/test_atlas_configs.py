"""Tests for the named world-model configurations (`titans.worldmodel.atlas`).

The module exists so an experiment is reproducible from a name rather than from
ad-hoc kwargs -- which is how four flags ended up silently inert. These tests
guard the two properties that would make it useless: that the ladder really is
cumulative, and that each named config actually reaches the model.
"""

import pytest
import torch

from titans.worldmodel.atlas import CONFIGS, describe, run
from titans.worldmodel.model import WorldModel


def test_ladder_is_cumulative():
    """Each rung must be a strict superset of the one above it, so a difference
    between two adjacent configs isolates exactly one ingredient."""
    order = ["titans", "phi", "table7", "omega", "atlas"]
    for lower, upper in zip(order, order[1:]):
        a, b = CONFIGS[lower], CONFIGS[upper]
        assert set(a) <= set(b), f"{upper} drops flags from {lower}"
        for k, v in a.items():
            assert b[k] == v, f"{upper} changes {k} instead of only adding"
        assert len(b) > len(a), f"{upper} adds nothing to {lower}"


def test_titans_is_the_empty_config():
    """The baseline must turn nothing on -- otherwise it is not the paper's."""
    assert CONFIGS["titans"] == {}


def test_each_config_builds_and_differs_from_the_baseline():
    torch.manual_seed(0)
    z = torch.randn(2, 16, 16)
    outs = {}
    for name, cfg in CONFIGS.items():
        torch.manual_seed(1)
        m = WorldModel(latent_dim=16, dim=16, depth=1, segment_size=8, **cfg)
        out = m(z)
        assert out.shape == z.shape
        assert torch.isfinite(out).all(), f"{name} produced non-finite output"
        outs[name] = out
    for name in CONFIGS:
        if name == "titans":
            continue
        assert not torch.allclose(outs[name], outs["titans"]), \
            f"{name} is numerically identical to the baseline -- a flag is inert"

def test_describe_names_the_flags():
    assert describe("titans").endswith("(paper default)")
    assert "feature_degree=2" in describe("atlas")


def test_run_smoke(tmp_path):
    """End-to-end through the harness on a tiny fake cache, including the
    epochs->steps conversion and the mini-batched eval."""
    g = torch.Generator().manual_seed(0)
    tr, va = tmp_path / "tr.pt", tmp_path / "va.pt"
    torch.save(torch.randn(8, 16, 12, generator=g), tr)
    torch.save(torch.randn(4, 16, 12, generator=g), va)

    # dim must match the cache width: the model has no input projection to absorb
    # a mismatch, it raises instead.
    res = run(tr, va, ["titans", "phi"], seeds=[0], epochs=1, batch_size=4,
              dim=12, depth=1, segment_size=8)
    assert set(res) == {"titans", "phi"}
    for r in res.values():
        assert r["val_l1_mean"] == r["val_l1_mean"]        # finite
        assert r["per_seed"][0]["steps"] == 2              # ceil(8/4) x 1 epoch
        assert "baseline_l1" in r


def test_unknown_config_is_rejected_by_name():
    with pytest.raises(KeyError):
        describe("not_a_config")


# --- the head count is a real axis, not an accepted-and-inert flag ----------
# Four options in this repo were once found to be accepted and to change nothing.
# --heads exists because every run so far silently used a single head, so it gets
# the same guard as the rest.

def test_heads_flag_changes_the_attention_and_defaults_to_one():
    from titans.worldmodel.frozen import build_cfg
    from titans.worldmodel.model import WorldModel

    assert "num_heads" not in build_cfg("sane_ns")
    assert build_cfg("sane_ns", num_heads=8)["num_heads"] == 8

    torch.manual_seed(0)
    one = WorldModel(latent_dim=16, dim=16, depth=1, segment_size=4,
                     **build_cfg("sane_ns"))
    torch.manual_seed(0)
    many = WorldModel(latent_dim=16, dim=16, depth=1, segment_size=4,
                      **build_cfg("sane_ns", num_heads=4))
    assert one.blocks[0].attn.num_heads == 1
    assert many.blocks[0].attn.num_heads == 4
    assert many.blocks[0].attn.head_dim == 4

    z = torch.randn(2, 8, 16)
    assert not torch.allclose(one(z), many(z), atol=1e-6)
