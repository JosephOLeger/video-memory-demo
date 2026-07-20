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
    assert {"train_l1", "val_l1", "steps"} <= set(r)
    assert r["steps"] == 5
    for k in ("train_l1", "val_l1"):
        assert r[k] == r[k]                       # finite, not NaN


def test_epochs_converts_to_the_right_number_of_steps():
    """epochs is the meaningful unit once a cache holds thousands of sequences."""
    g = torch.Generator().manual_seed(0)
    train_z = torch.randn(10, 16, 8, generator=g)
    val_z = torch.randn(2, 16, 8, generator=g)
    r = train_eval_once(train_z, val_z, seed=0, epochs=2, device="cpu",
                        batch_size=4, dim=8, depth=1, segment_size=8)
    assert r["steps"] == 6                        # ceil(10/4)=3 per epoch, x2


def test_steps_and_epochs_are_mutually_exclusive():
    z = torch.randn(4, 16, 8, generator=torch.Generator().manual_seed(0))
    for kw in ({}, {"steps": 5, "epochs": 1}):
        try:
            train_eval_once(z, z, seed=0, device="cpu", batch_size=2,
                            dim=8, depth=1, segment_size=8, **kw)
        except ValueError:
            continue
        raise AssertionError(f"should have raised for {kw}")


def test_eval_is_minibatched_and_matches_the_full_set():
    """The mini-batched eval must equal a single full-set forward exactly --
    otherwise fixing the OOM would silently change every reported number."""
    from titans.worldmodel.harness import _eval_l1
    import torch.nn.functional as F
    from titans.worldmodel.model import WorldModel

    torch.manual_seed(0)
    z = torch.randn(7, 16, 8)
    m = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=8)
    with torch.no_grad():
        full = F.l1_loss(m(z)[:, :-1], z[:, 1:]).item()
    for bs in (1, 2, 3, 7, 16):
        assert abs(_eval_l1(m, z, bs) - full) < 1e-5, f"batch_size={bs} disagrees"
