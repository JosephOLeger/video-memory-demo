"""The two failure modes must be told apart, not merely detected.

Reading only ``finite`` calls a collapsed memory healthy; reading only ``pr`` calls a
diverging one collapsed, because its rank falls on the way out too. These tests pin
the pair of signatures that separates them.
"""

import math

import pytest

from titans.worldmodel.tradeoff import (classify, collapse_time, divergence_rate,
                                        rows_from, tau_acc)


def _pts(prs, snorms=None, w1=None):
    snorms = snorms or [1.0] * len(prs)
    w1 = w1 or [8.0] * len(prs)
    return [{"rep": i, "finite": True,
             "b0": {"pr": p, "S_norm": s, "w1_norm": w, "cos_ret": 0.2, "finite": True}}
            for i, (p, s, w) in enumerate(zip(prs, snorms, w1))]


# --- the shared accumulation axis ------------------------------------------
def test_tau_acc_is_the_momentum_horizon_for_the_scalar():
    assert tau_acc({"eta_init": 0.9}) == pytest.approx(10)
    assert tau_acc({"eta_init": 0.95}) == pytest.approx(20)


def test_an_uninitialised_eta_is_read_as_its_resting_value():
    """No eta_init means sigmoid(0) = 0.5, a horizon of 2 tokens -- not 'unknown'."""
    assert tau_acc({}) == 2.0


def test_a_leakless_cascade_is_an_infinite_horizon():
    assert tau_acc({"cascade": True, "leak": "none"}) == math.inf


def test_the_cascade_horizon_uses_the_leak_rate_and_theta():
    t = tau_acc({"cascade": True, "leak": "surprise", "leak_rate": 0.1}, theta_bar=0.05)
    assert t == pytest.approx(200.0)


# --- collapse ---------------------------------------------------------------
def test_collapse_time_is_the_e_folding_of_the_rank():
    # 60 -> 20, e-folding threshold at 20 + 40/e = 34.7, first crossed at rep 2.
    col = collapse_time(_pts([60, 45, 30, 22, 20, 20, 20, 20, 20, 20]))
    assert col["collapsed"] is True
    assert col["tau_collapse"] == 2 * 64
    assert col["pr_inf"] == 20


def test_a_stable_rank_reports_no_collapse_time_rather_than_zero():
    """The memory-off controls hold their rank for 64k tokens; a time there is noise."""
    col = collapse_time(_pts([60] * 10))
    assert col["collapsed"] is False
    assert col["tau_collapse"] is None


# --- divergence -------------------------------------------------------------
def test_a_geometric_blowup_gives_a_positive_rate_and_a_finite_efolding_time():
    div = divergence_rate(_pts([60] * 6, snorms=[1, 2, 4, 8, 16, 32]))
    assert div["rate_per_token"] > 0
    assert div["tau_diverge"] == 1 / div["rate_per_token"]


def test_a_bounded_norm_gives_no_divergence_time():
    """The collapsing arms keep ||S|| small -- that is what separates them."""
    div = divergence_rate(_pts([60] * 6, snorms=[1.0, 0.9, 1.1, 1.0, 0.95, 1.05]))
    assert div["tau_diverge"] is None or abs(div["rate_per_token"]) < 1e-3


# --- the classifier ---------------------------------------------------------
def test_divergence_wins_over_collapse_when_the_state_stopped_being_finite():
    col = collapse_time(_pts([60, 30, 12]))
    div = divergence_rate(_pts([60, 30, 12], snorms=[1, 10, 100]))
    assert classify({"reps_finite": 3}, div, col, reps_total=30) == "diverge"


def test_a_finite_but_rank_collapsed_run_is_not_healthy():
    pts = _pts([60, 45, 30, 22, 20, 20, 20, 20, 20, 20])
    col, div = collapse_time(pts), divergence_rate(pts)
    assert classify({"reps_finite": 10}, div, col, reps_total=10) == "collapse"


def test_a_run_that_keeps_its_rank_and_its_norm_is_healthy():
    pts = _pts([60] * 10)
    col, div = collapse_time(pts), divergence_rate(pts)
    assert classify({"reps_finite": 10}, div, col, reps_total=10) == "sain"


# --- the reader -------------------------------------------------------------
def test_rows_from_skips_a_probe_with_no_such_mode(tmp_path):
    import json
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"label": "a", "reps": 3, "streams": {"loop": []}}))
    assert rows_from(str(tmp_path / "*.json"), mode="cycle") == []


# --- the theta/alpha audit spans several sweep directories ------------------
def test_margin_accepts_several_globs_and_skips_empty_ones(tmp_path, monkeypatch):
    """The audit runs on a timer while cells are still landing, so a directory that
    is not filled yet must be skipped -- not turned into an error record."""
    import sys
    from titans.worldmodel import margin as M

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for n in ("x", "y"):
        (tmp_path / "a" / f"{n}.pt").write_bytes(b"")

    seen = []
    monkeypatch.setattr(M, "load_frozen", lambda p, d: (_ for _ in ()).throw(OSError("stub")))
    monkeypatch.setattr(M, "load_latent_cache",
                        lambda p: (__import__("torch").zeros(8, 4, 4), {"pca": True}))
    out = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", [
        "margin", "--val", "v.pt", "--device", "cpu", "--out", str(out),
        "--frozen", str(tmp_path / "a" / "*.pt"),
        "--frozen", str(tmp_path / "b" / "*.pt"),      # empty: must vanish silently
        "--frozen", str(tmp_path / "a" / "x.pt"),      # already covered: no duplicate
    ])
    M.main()
    import json
    recs = json.load(open(out))
    seen = [r["frozen"] for r in recs]
    assert seen == ["x.pt", "y.pt"]
