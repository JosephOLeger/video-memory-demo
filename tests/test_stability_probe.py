"""The probe has to survive the divergence it exists to measure.

On 2026-08-18 a 40-minute trace died on ``linalg.eigh: The algorithm failed to
converge`` and took every row with it -- the run had reached ~1e19 activations, the
Gram matrix squared them past what ``eigvalsh`` can factor, and rows were only
returned at the end. Both halves of that are tested here.
"""

from __future__ import annotations

import torch

from titans.memory import NeuralMemory
from titans.stability import chunk_transfer, trace_forward, weighted_gram_lambda_max


def _w(m: int) -> torch.Tensor:
    return torch.ones(m, dtype=torch.float64)


def test_gram_returns_inf_instead_of_raising_on_a_runaway_state():
    assert weighted_gram_lambda_max(torch.full((4, 8), float("inf")), _w(4)) == float("inf")
    assert weighted_gram_lambda_max(torch.full((4, 8), torch.nan), _w(4)) == float("inf")
    # Finite input whose *square* overflows: the squaring is where it actually breaks.
    big = torch.full((4, 8), 1e200, dtype=torch.float64)
    assert weighted_gram_lambda_max(big, _w(4)) == float("inf")


def test_gram_is_unchanged_on_healthy_input():
    torch.manual_seed(0)
    X = torch.randn(6, 32)
    w = torch.rand(6, dtype=torch.float64)
    G = (w.sqrt().unsqueeze(-1) * X.double()) @ (w.sqrt().unsqueeze(-1) * X.double()).T
    assert weighted_gram_lambda_max(X, w) == float(2.0 * torch.linalg.eigvalsh(G).max())


def test_chunk_transfer_reports_inf_rather_than_raising():
    _, rho = chunk_transfer(0.99, 0.9, 8, 0.03, float("inf"), 1.0)
    assert rho == float("inf")
    _, rho_ok = chunk_transfer(0.99, 0.9, 8, 0.03, 1.0, 1.0)
    assert 0.0 < rho_ok < 10.0


def test_trace_streams_every_row_as_it_is_produced():
    """``on_row`` must see each row before the scan can crash on a later one."""
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03, normalize_qk="l2",
                       eta_init=0.995, update_bias=False)
    seen = []
    rows = trace_forward(mem, torch.randn(1, 64, 8), 8, on_row=seen.append)
    assert len(seen) == len(rows) > 0
    assert [r["chunk"] for r in seen] == [r["chunk"] for r in rows]


def test_a_probe_that_throws_is_recorded_and_ends_the_trace():
    """A failure inside probe_chunk becomes a row, not a lost run."""
    import titans.stability as st

    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, normalize_qk="l2", update_bias=False)
    real, calls = st.probe_chunk, {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("ill-conditioned")
        return real(*args, **kwargs)

    st.probe_chunk = boom
    try:
        rows = trace_forward(mem, torch.randn(1, 64, 8), 8)
    finally:
        st.probe_chunk = real
    assert len(rows) == 3 and rows[-1]["finite"] is False
    assert "ill-conditioned" in rows[-1]["note"]
