"""The long-range structure probe: it must find real structure and reject noise."""

import pytest
import torch

from titans.worldmodel.latent_structure import incremental_r2, windows


def _white(B=200, T=40, D=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g, dtype=torch.float64)


def test_windows_aligns_the_target_with_its_own_history():
    z = torch.arange(2 * 10 * 1, dtype=torch.float64).reshape(2, 10, 1)
    X_win, X_lag, Y = windows(z, w=2, lag=5)
    # first valid t is 5; Y[0] is z[0, 5], the window is z[0,4] then z[0,3]
    assert Y[0].item() == z[0, 5].item()
    assert X_win[0, 0].item() == z[0, 4].item()
    assert X_win[0, 1].item() == z[0, 3].item()
    assert X_lag[0].item() == z[0, 0].item()


def test_windows_returns_none_when_the_lag_does_not_fit():
    assert windows(_white(T=8), w=4, lag=8) is None


def test_white_noise_gives_no_incremental_structure_above_the_null():
    """The floor: with no structure, the lagged latent must not beat the null."""
    rows = incremental_r2(_white(), w=4, lags=[6, 12, 20])
    for r in rows:
        assert r["r2_inc"] < 3 * max(r["r2_null"], 1e-9)
        assert abs(r["r2_inc"]) < 0.05


def test_a_planted_long_lag_dependency_is_recovered():
    """Sanity in the other direction: real structure at lag 15 must show up there."""
    z = _white(B=400, T=40, D=6, seed=1)
    z[:, 15:, 0] += 3.0 * z[:, :-15, 1]           # z_t[0] depends on z_{t-15}[1]
    rows = {r["lag"]: r for r in incremental_r2(z, w=4, lags=[10, 15, 20])}
    assert rows[15]["r2_inc"] > 10 * max(rows[15]["r2_null"], 1e-9)
    assert rows[15]["r2_inc"] > rows[10]["r2_inc"]
    assert rows[15]["r2_inc"] > rows[20]["r2_inc"]


def test_the_null_uses_a_different_sequence_not_a_different_time():
    """The null must break the sequence pairing, or it would inherit the structure."""
    z = _white(B=400, T=40, D=6, seed=2)
    z[:, 15:, 0] += 3.0 * z[:, :-15, 1]
    r = {x["lag"]: x for x in incremental_r2(z, w=4, lags=[15])}[15]
    assert r["r2_null"] < 0.05          # planted structure must not leak into the null


def test_fit_and_score_rows_are_disjoint_and_cover_everything():
    rows = incremental_r2(_white(B=100, T=20, D=4), w=4, lags=[6])
    r = rows[0]
    assert r["n_fit"] + r["n_score"] == 100 * (20 - 6)
    assert r["n_fit"] == 50 * (20 - 6)


def test_window_r2_is_reported_and_is_the_baseline_to_beat():
    z = _white(B=300, T=30, D=4, seed=3)
    z[:, 1:, 0] += 2.0 * z[:, :-1, 1]            # strong lag-1 structure
    r = incremental_r2(z, w=4, lags=[8])[0]
    assert r["r2_window"] > 0.1                  # the window already explains a lot
