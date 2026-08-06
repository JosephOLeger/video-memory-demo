"""The forward-only length probe: shapes, both momentum topologies, and the slope."""

import math

import pytest
import torch

from titans.memory import NeuralMemory
from titans.membench.leak_length import growth_exponent, run_length


def _mem(**kw):
    torch.manual_seed(0)
    return NeuralMemory(dim=16, hidden_dim=32, num_layers=2, **kw)


def _trace(mem, T=64, chunk=8):
    torch.manual_seed(1)
    x = torch.randn(2, T, 16)
    return run_length(mem, x, chunk, every=1)


def test_scalar_trace_covers_every_chunk():
    tr = _trace(_mem())
    assert [r["chunk"] for r in tr] == list(range(8))
    assert [r["token"] for r in tr] == [8 * (i + 1) for i in range(8)]
    assert all(r["finite"] for r in tr)


def test_cascade_path_runs():
    """Regression: S must come from init_memory, which carries the [..., K] axis.

    Building it with ``zeros_like(state)`` gives tensors without the reservoir
    dimension, and ``cascade_step`` fails on the flux broadcast rather than
    silently doing something wrong -- so this is a crash, not a wrong number.
    """
    mem = _mem(use_cascade_momentum=True, cascade_reservoirs=4)
    tr = _trace(mem)
    assert len(tr) == 8
    assert all(r["finite"] for r in tr)


def test_cascade_visible_state_is_a_strict_part_of_the_whole():
    """``read(u) = u[..., 0]`` -- only reservoir 1 reaches the weights.

    The distinction is the point of the study: the cascade can hold a large total
    mass while what the memory actually sees stays small, so reporting only one of
    the two norms would miss either the accumulation or its effect.
    """
    mem = _mem(use_cascade_momentum=True, cascade_reservoirs=4)
    r = _trace(mem)[-1]
    assert 0.0 < r["S_visible"] <= r["S_norm"] + 1e-9


def test_scalar_visible_state_equals_the_whole():
    r = _trace(_mem())[-1]
    assert r["S_visible"] == pytest.approx(r["S_norm"], rel=1e-9)


@pytest.mark.parametrize("power, expected", [(1.0, 1.0), (0.5, 0.5), (0.0, 0.0)])
def test_growth_exponent_recovers_a_power_law(power, expected):
    tr = [{"token": t, "S_norm": float(t) ** power, "finite": True}
          for t in range(1, 200)]
    assert growth_exponent(tr) == pytest.approx(expected, abs=1e-6)


def test_growth_exponent_needs_enough_finite_points():
    tr = [{"token": 1, "S_norm": 1.0, "finite": True}]
    assert math.isnan(growth_exponent(tr))


def test_trace_stops_at_the_first_non_finite_chunk():
    """A blown-up state must end the trace, not be extrapolated through NaN.

    The blow-up is injected rather than provoked by a large ``theta_scale``: the
    recurrence is robust enough that no scale reliably NaNs on random input in a
    few chunks, and a test that depends on the dynamics diverging would be testing
    the dynamics, not the break.
    """
    mem = _mem()
    real = mem._scan_chunked
    calls = []

    def poisoned(*args, **kw):
        state, S, extra = real(*args, **kw)
        calls.append(1)
        if len(calls) == 3:
            state = {k: torch.full_like(v, float("nan")) for k, v in state.items()}
        return state, S, extra

    mem._scan_chunked = poisoned
    tr = _trace(mem, T=256)
    assert [r["chunk"] for r in tr] == [0, 1, 2]
    assert not tr[-1]["finite"]
    assert tr[-1]["token"] == 24
