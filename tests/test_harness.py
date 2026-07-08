"""Tests for the experiment harness (baseline + a tiny train/eval run)."""

import torch

from titans.worldmodel.harness import persistence_baseline_l1, train_eval_once


def test_persistence_baseline_matches_manual():
    z = torch.randn(3, 10, 8, generator=torch.Generator().manual_seed(0))
    expected = (z[:, 1:] - z[:, :-1]).abs().mean().item()
    assert abs(persistence_baseline_l1(z) - expected) < 1e-6


def test_train_eval_once_runs_and_is_finite():
    g = torch.Generator().manual_seed(0)
    train_z = torch.randn(4, 16, 8, generator=g)
    val_z = torch.randn(2, 16, 8, generator=g)
    r = train_eval_once(train_z, val_z, seed=0, steps=5, device="cpu",
                        batch_size=2, dim=8, depth=1, segment_size=8)
    assert set(r) == {"train_l1", "val_l1"}
    assert all(v == v for v in r.values())  # finite, not NaN
