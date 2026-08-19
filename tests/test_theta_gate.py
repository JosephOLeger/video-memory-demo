"""The inner learning rate's gate, bounded or not.

Why this exists as its own file: ``CV(theta_t)`` -- how much the learned inner
learning rate varies over tokens -- predicts ``memory_gain`` at ``r = -0.992``
across ten trained world models, and at ``r = -0.95`` *within* each arm taken
separately. That is the strongest single predictor of memory quality this project
has measured, and Titans' own gate (an unbounded softplus) is what lets the
dispersion run.

``theta_gate="sigmoid"`` is the causal test: the correlation alone is equally
consistent with a good memory *producing* smooth gradients, so the dispersion has
to be imposed rather than observed.
"""

from __future__ import annotations

import math

import pytest
import torch

from titans.memory import NeuralMemory


def _mem(gate="softplus", **kw):
    torch.manual_seed(0)
    return NeuralMemory(dim=16, hidden_dim=8, theta_scale=0.03, theta_gate=gate, **kw)


def test_the_two_gates_agree_exactly_at_initialisation():
    """The sigmoid's gain is ``2 log 2`` precisely so that ``h = 0`` gives
    ``softplus(0) = log 2``. A run that switches gates therefore starts from the
    same inner learning rate and only the tail differs."""
    a, b = _mem("softplus"), _mem("sigmoid")
    h = torch.zeros(3, 1)
    assert torch.allclose(a._theta(h), b._theta(h), atol=1e-7)
    assert float(a._theta(h)[0]) == pytest.approx(math.log(2.0))


def test_softplus_is_unbounded_and_sigmoid_is_not():
    a, b = _mem("softplus"), _mem("sigmoid")
    big = torch.full((4, 1), 40.0)
    assert float(a._theta(big).max()) > 39.0            # tracks h with no ceiling
    assert float(b._theta(big).max()) <= 2.0 * math.log(2.0) + 1e-6


def test_the_bound_is_twice_the_initial_value():
    """Stated as a number because it is the experiment's knob: theta can still
    double, it just cannot run away."""
    b = _mem("sigmoid")
    ceiling = float(b._theta(torch.full((1, 1), 1e3)).max()) * b.theta_scale
    init = float(b._theta(torch.zeros(1, 1))[0]) * b.theta_scale
    assert ceiling == pytest.approx(2.0 * init, rel=1e-6)


def test_sigmoid_compresses_the_dispersion_on_real_inputs():
    """The quantity the whole experiment is about: CV(theta) over tokens."""
    x = torch.randn(2, 64, 16)
    cvs = {}
    for gate in ("softplus", "sigmoid"):
        th = _mem(gate)._project(x)[5].detach()
        cvs[gate] = float(th.std() / th.mean())
    assert cvs["sigmoid"] < cvs["softplus"], cvs


def test_validation_and_default():
    assert _mem().theta_gate == "softplus"          # Titans' own, unchanged
    with pytest.raises(ValueError, match="theta_gate"):
        NeuralMemory(dim=8, theta_gate="relu")


def test_runs_end_to_end_under_unit():
    """The configuration the experiment actually launches."""
    mem = _mem("sigmoid", normalize_qk="l2", eta_init=0.995,
               orthogonalize_momentum="unit", update_bias=False)
    x = torch.randn(2, 32, 16)
    state, S = mem.init_memory(2)
    state, S = mem.update(x, state, S, 8)
    y = mem.retrieve(state, mem.q_proj(x))
    assert torch.isfinite(y).all()
    assert torch.isfinite(state["w1"]).all()
