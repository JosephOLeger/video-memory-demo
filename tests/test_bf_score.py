"""The Benna-Fusi scorer: does it measure the chain we actually run?

The scorer's whole value is that it reports a property of `cascade_step` rather
than of a hand-written copy of the dynamics. These tests pin that link: the
transition matrix must reproduce a literal rollout, and the three reference
numbers (no areas / areas / areas+terminal) must not move.
"""

import math

import torch

from titans.metaplastic import cascade_step, geometric_betas
from titans.worldmodel.bf_score import (
    impulse_response,
    lifetime,
    loglog_slope,
    transition,
)

N_128 = 128 * 128


def _rollout(K, beta1, g, areas, terminal, steps):
    """The dynamics driven step by step, with no matrix in sight."""
    b = geometric_betas(K, beta1, g, dtype=torch.float64)
    a = (torch.tensor([areas ** k for k in range(K)], dtype=torch.float64)
         if areas is not None else None)
    tb = (torch.tensor(beta1 * g ** (-K), dtype=torch.float64)
          if terminal else None)
    zero = torch.zeros((), dtype=torch.float64)
    u = cascade_step(torch.zeros(K, dtype=torch.float64),
                     torch.tensor(1.0, dtype=torch.float64), b, a, tb)
    out = [u.clone()]
    for _ in range(steps - 1):
        u = cascade_step(u, zero, b, a, tb)
        out.append(u.clone())
    return torch.stack(out)


def test_transition_matrix_reproduces_a_literal_rollout():
    for areas, terminal in [(None, False), (2.0, False), (2.0, True)]:
        B = transition(6, 0.5, 2.0, areas=areas, terminal=terminal)
        ref = _rollout(6, 0.5, 2.0, areas, terminal, 200)
        u = ref[0].clone()
        for t in range(200):
            assert torch.allclose(u, ref[t], atol=1e-12), (areas, terminal, t)
            u = B @ u


def test_impulse_response_matches_the_rollout_readout():
    r = impulse_response(6, 0.5, 2.0, areas=2.0, terminal=True, T=500)
    ref = _rollout(6, 0.5, 2.0, 2.0, True, 500)[:, 0]
    assert torch.allclose(r, ref, atol=1e-12)


def test_the_three_reference_configurations_do_not_move():
    """The numbers the Benna-Fusi confrontation is written around."""
    broken = impulse_response(6, 0.5, 2.0, areas=None, terminal=False)
    assert loglog_slope(broken, 10, 1e3) > -0.2          # flat: an accumulator
    assert lifetime(broken, N_128) < 10

    areas = impulse_response(6, 0.5, 2.0, areas=2.0, terminal=False)
    assert abs(loglog_slope(areas, 10, 1e3) + 0.5) < 0.02

    full = impulse_response(6, 0.5, 2.0, areas=2.0, terminal=True)
    assert abs(loglog_slope(full, 10, 1e3) + 0.5) < 0.02
    # The terminal boundary is what turns a correct slope into a usable trace.
    assert lifetime(full, N_128) > 5 * lifetime(areas, N_128)


def test_the_shipped_leak_rate_destroys_the_power_law():
    """`--leak-rate 0.1` with the theta a trained model emits (~0.03)."""
    eff = 1.0 - math.exp(-0.1 * 0.03)
    r = impulse_response(6, 0.5, 2.0, areas=2.0, terminal=True, uniform_leak=eff)
    assert loglog_slope(r, 10, 1e3) < -0.7

    # The rate the sweep moved to keeps it.
    eff_slow = 1.0 - math.exp(-3e-4 * 0.03)
    slow = impulse_response(6, 0.5, 2.0, areas=2.0, terminal=True,
                            uniform_leak=eff_slow)
    assert abs(loglog_slope(slow, 10, 1e3) + 0.5) < 0.02


def test_a_pure_accumulator_scores_zero_lifetime():
    """No areas and no terminal leak: mass is conserved, so SNR never exceeds 1."""
    r = impulse_response(6, 0.5, 2.0, areas=None, terminal=False, readout="sum")
    assert lifetime(r, N_128) == 0
