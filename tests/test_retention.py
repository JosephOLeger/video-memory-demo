"""The spaced echo: sequence construction, scoring window, and the n=0 reduction."""

import pytest
import torch

from titans.worldmodel.echo import double, per_copy_l1_K
from titans.worldmodel.frozen import build_cfg
from titans.worldmodel.model import WorldModel
from titans.worldmodel.retention import (arm_of, last_block_l1, retention_curve,
                                         spaced)


def _model(d=8):
    torch.manual_seed(0)
    return WorldModel(latent_dim=d, dim=d, depth=1, segment_size=4,
                      memory_num_layers=2).eval()


def _va(n=12, t=16, d=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, d, generator=g)


def test_spaced_puts_v_last_and_first_first():
    va = _va(n=3, t=4, d=2)
    u, d1 = torch.roll(va, 1, 0), torch.roll(va, 2, 0)
    seq = spaced(va, u, [d1])
    assert seq.shape[1] == 3 * 4
    assert torch.equal(seq[:, :4], u)          # the prefix that differs
    assert torch.equal(seq[:, 4:8], d1)        # the distractor
    assert torch.equal(seq[:, 8:], va)         # the scored block is always V


def test_only_the_first_block_differs_between_the_two_conditions():
    """The gap must be attributable to the prefix, not to different content."""
    va = _va(n=6, t=4, d=2)
    distr = [torch.roll(va, k, 0) for k in (2, 3)]
    rep = spaced(va, va, distr)
    crs = spaced(va, torch.roll(va, 1, 0), distr)
    assert not torch.equal(rep[:, :4], crs[:, :4])
    assert torch.equal(rep[:, 4:], crs[:, 4:])


def test_last_block_l1_scores_the_final_block_only():
    """A doubled sequence must give exactly what per_copy_l1_K calls the 2nd copy."""
    m, va = _model(), _va()
    seq = double(va)
    _, second = per_copy_l1_K(m, seq, T=16, K=2, reset=False)
    got = last_block_l1(m, seq, T=16, K=2, reset=False)
    assert got == pytest.approx(second, rel=1e-9)


def test_reset_flag_is_restored_even_if_scoring_raises():
    """A leaked reset flag would silently disable memory in every later measurement."""
    m = _model()
    with pytest.raises(Exception):
        last_block_l1(m, _va(), T=16, K=0, reset=True)   # K=0 -> empty slice, raises
    assert all(not b.reset_memory_each_segment for b in m.blocks)


def test_curve_reports_gap_in_latents_and_grows_the_sequence():
    m, va = _model(), _va()
    rows = retention_curve(m, va, T=16, K=2, gaps=[0, 1, 3])
    assert [r["gap_blocks"] for r in rows] == [0, 1, 3]
    assert [r["gap_latents"] for r in rows] == [0, 16, 48]
    assert [r["seq_len"] for r in rows] == [32, 48, 80]


def test_memory_off_control_is_reported_at_every_gap():
    """The floor: with the carry dropped, nothing can cross the gap."""
    m, va = _model(), _va()
    for r in retention_curve(m, va, T=16, K=2, gaps=[0, 2]):
        assert "gain_memoff" in r
        assert abs(r["gain_memoff"]) < 0.2      # no memory -> no retention


def test_a_gap_wider_than_the_batch_is_skipped_not_wrapped():
    """torch.roll would silently reuse V itself as a distractor past B."""
    m, va = _model(), _va(n=6)
    rows = retention_curve(m, va, T=16, K=2, gaps=[2, 8])
    assert [r["gap_blocks"] for r in rows] == [2]


def test_arm_name_drops_the_seed_suffix():
    assert arm_of("frozen_vv/cas_r0.1_s3.pt") == "cas_r0.1"
    assert arm_of("frozen_cv/nocas_e0.90.pt") == "nocas_e0.90"


def test_collect_pools_seeds_and_reports_a_sem():
    """Per-seed spread on this benchmark is ~0.03 on a gain of ~0.25, so a curve
    drawn from one seed would suggest orderings that do not survive."""
    from titans.worldmodel.plot_retention import collect

    rows = [
        {"arm": "a", "cascade": True,
         "curve": [{"gap_latents": 0, "retention_gain": 0.20, "gain_memoff": 0.0},
                   {"gap_latents": 64, "retention_gain": 0.10, "gain_memoff": 0.0}]},
        {"arm": "a", "cascade": True,
         "curve": [{"gap_latents": 0, "retention_gain": 0.30, "gain_memoff": 0.0},
                   {"gap_latents": 64, "retention_gain": 0.20, "gain_memoff": 0.0}]},
        {"arm": "b", "cascade": False, "error": "boom"},
    ]
    curves, floor, meta = collect(rows)
    assert set(curves) == {"a"}                 # the failed artifact is dropped
    m, sem, n = curves["a"][0]
    assert m == pytest.approx(0.25) and n == 2 and sem == pytest.approx(0.05)
    assert floor[0][0] == pytest.approx(0.0)
    assert meta["a"]["cascade"] is True
