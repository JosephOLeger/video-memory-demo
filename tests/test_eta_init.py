"""`eta_init`: setting the momentum gate's resting value.

`alpha_init` fixed the forget gate, whose default `sigmoid(0)=0.5` is a half-life
of one token. The momentum gate has exactly the same defect and never got the same
treatment: at rest `eta=0.5` gives an effective horizon `1/(1-eta)` of ~2 tokens,
so the memory's *accumulation* stays short-range however long its retention is.
These tests pin that the knob exists, does what it claims, and changes the
computation rather than merely being accepted.
"""

import math

import pytest
import torch

from titans.mac import TitansMACBlock
from titans.memory import NeuralMemory
from titans.worldmodel.model import WorldModel


def _resting_eta(mem: NeuralMemory) -> float:
    return torch.sigmoid(mem.eta_head(torch.zeros(1, 1, mem.dim))).item()


@pytest.mark.parametrize("eta", [0.9, 0.98, 0.995])
def test_eta_init_sets_the_resting_value(eta):
    m = NeuralMemory(dim=16, eta_init=eta)
    assert _resting_eta(m) == pytest.approx(eta, abs=1e-5)


def test_default_leaves_the_two_token_horizon():
    """The documented status quo: no init -> eta ~ 0.5 -> horizon ~2 tokens."""
    m = NeuralMemory(dim=16)
    eta = _resting_eta(m)
    assert 0.4 < eta < 0.6
    assert 1.0 / (1.0 - eta) < 4.0


def test_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="eta_init"):
            NeuralMemory(dim=8, eta_init=bad)


def test_eta_init_changes_the_memory_state():
    """Not merely accepted: a different momentum horizon must move the weights."""
    x = torch.randn(2, 16, 16)

    def final(eta):
        torch.manual_seed(0)
        m = NeuralMemory(dim=16, eta_init=eta)
        state, S = m.init_memory(2)
        state, _ = m.update(x, state, S, chunk_size=8)
        return state["w1"]

    assert not torch.allclose(final(0.5), final(0.99), atol=1e-6)


def test_plumbed_through_block_and_world_model():
    blk = TitansMACBlock(16, eta_init=0.99)
    assert _resting_eta(blk.memory) == pytest.approx(0.99, abs=1e-5)
    wm = WorldModel(latent_dim=16, dim=16, depth=1, segment_size=4,
                    eta_init=0.98)
    assert _resting_eta(wm.blocks[0].memory) == pytest.approx(0.98, abs=1e-5)
    assert torch.isfinite(wm(torch.randn(2, 8, 16))).all()


def test_horizon_matches_the_formula():
    """The knob is documented in tokens; check the arithmetic it advertises."""
    for horizon in (10, 50, 100):
        eta = 1.0 - 1.0 / horizon
        m = NeuralMemory(dim=8, eta_init=eta)
        assert 1.0 / (1.0 - _resting_eta(m)) == pytest.approx(horizon, rel=1e-3)
