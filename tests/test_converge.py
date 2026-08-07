"""The convergence check: snapshot naming, ordering, and that eval needs no training."""

import pytest
import torch

from titans.worldmodel.converge import arm_of, step_of
from titans.worldmodel.echo import evaluate
from titans.worldmodel.frozen import build_cfg
from titans.worldmodel.model import WorldModel


def test_step_and_arm_are_recovered_from_the_snapshot_name():
    p = "frozen_cv/cas_none_s0_at3750.pt"
    assert step_of(p) == 3750
    assert arm_of(p) == "cas_none_s0"


def test_an_unnumbered_artifact_sorts_before_every_snapshot():
    """A plain frozen file caught by the glob must not masquerade as step 0."""
    assert step_of("frozen_vv/cas_none_s0.pt") == -1


def test_snapshots_sort_numerically_not_lexicographically():
    paths = ["a_at10000.pt", "a_at1250.pt", "a_at2500.pt"]
    assert sorted(paths, key=step_of) == ["a_at1250.pt", "a_at2500.pt", "a_at10000.pt"]


def test_evaluate_scores_a_model_without_training_it():
    """The whole point: the same metric at several steps, re-training none of them."""
    torch.manual_seed(0)
    m = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4, memory_num_layers=2,
                   **build_cfg("sane_ns")).eval()
    before = [p.detach().clone() for p in m.parameters()]
    g = torch.Generator().manual_seed(1)
    va = torch.randn(6, 8, 8, generator=g)

    r = evaluate(m, va, T=8, K=2, n_eval=4)

    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert r["memory_gain"] == pytest.approx(r["second_cross"] - r["second_repeat"])
    assert r["second_repeat"] == r["second"]
    assert r["pr"] >= 1.0
