"""Tests for the reach-vs-capacity reading of the retention curves.

The numbers this module produces are the ones a GPU request rests on, so the
tests are about the two ways such a reading goes wrong: reading a *distance* off a
geometric grid (linear interpolation would sit high), and quietly treating a
memory that never died inside the grid as one that died at the last point tested
(which bends the exponent downwards exactly where the argument needs it).
"""

import json
import math

import pytest

from titans.worldmodel.capacity_reach import (
    EGOLIFEQA_MEDIAN_S,
    SECONDS_PER_LATENT,
    hidden_from_arm,
    loglog_fit,
    memory_params,
    reach_of,
    seed_of,
    size_for_reach,
    sizes_from_training,
)


def curve(points, statuses=None):
    """``[(gap, gain), ...]`` -> the shape ``retention.py`` writes."""
    statuses = statuses or ["sain"] * len(points)
    return [{"gap_latents": g, "retention_gain": v, "status": s}
            for (g, v), s in zip(points, statuses)]


# ------------------------------------------------------------------- the reach
def test_reach_is_interpolated_in_log_distance():
    """The grid is geometric; a linear reading between 512 and 1024 sits high."""
    c = curve([(0, 1.0), (512, 0.2), (1024, 0.05)])
    lat, censored, _ = reach_of(c, frac=0.1)
    assert not censored
    # gain falls 0.2 -> 0.05 across the octave; tau = 0.1 sits 2/3 of the way.
    f = (0.2 - 0.1) / (0.2 - 0.05)
    assert lat == pytest.approx(math.exp(math.log(512) + f * math.log(2)))
    assert 512 < lat < 1024
    assert lat < 512 + f * 512          # strictly below the linear reading


def test_a_curve_that_never_crosses_is_censored_not_extrapolated():
    c = curve([(0, 1.0), (512, 0.9), (1024, 0.8)])
    lat, censored, why = reach_of(c, frac=0.1)
    assert censored and lat == 1024
    assert "largest gap" in why


def test_the_walk_stops_where_the_state_stops_being_healthy():
    """Past a collapsed state the gain is not measuring retention any more."""
    c = curve([(0, 1.0), (512, 0.5), (1024, 0.4), (2048, 0.9)],
              ["sain", "sain", "collapse", "sain"])
    lat, censored, why = reach_of(c, frac=0.1)
    assert censored and lat == 512      # not 2048, despite the high gain there
    assert "collapse" in why


def test_a_divergence_at_the_first_gap_leaves_no_reach():
    c = curve([(0, 1.0), (512, float("nan"))], ["sain", "diverge"])
    lat, censored, why = reach_of(c, frac=0.1)
    assert lat == 0.0 and not censored and "healthy" in why


def test_no_trace_at_distance_zero_is_a_reach_of_zero():
    for g0 in (0.0, -0.05):
        lat, censored, why = reach_of(curve([(0, g0), (512, 0.4)]), frac=0.1)
        assert lat == 0.0 and not censored and "distance 0" in why


def test_the_reach_always_lands_inside_its_bracket():
    """A rise after the crossing must not push the reach out of the interval.

    The crossing interval is always decreasing -- it is the first to trigger, so
    the left end still sits at or above the threshold -- but the *curve* need not
    be, and the reported distance must stay between the two grid points either way.
    """
    c = curve([(0, 1.0), (512, 0.4), (1024, 0.02), (2048, 0.5)])
    lat, censored, _ = reach_of(c, frac=0.1)
    assert not censored and 512 < lat < 1024


def test_a_crossing_inside_the_first_interval_is_linear():
    """log-interpolation is undefined at distance 0, so that one bracket is linear."""
    c = curve([(0, 1.0), (512, 0.05)])
    lat, _, why = reach_of(c, frac=0.1)
    assert lat == pytest.approx((1.0 - 0.1) / (1.0 - 0.05) * 512)
    assert "from 0" in why


def test_the_threshold_fraction_moves_the_reach_monotonically():
    c = curve([(0, 1.0), (512, 0.5), (1024, 0.25), (2048, 0.12), (4096, 0.06)])
    lax, _, _ = reach_of(c, frac=0.05)
    strict, _, _ = reach_of(c, frac=0.3)
    assert strict < lax


# --------------------------------------------------------------------- the fit
def test_loglog_fit_recovers_a_known_exponent():
    xs = [2 ** k for k in range(4, 12)]
    ys = [3.5 * x ** 0.93 for x in xs]
    b, a, se, n = loglog_fit(xs, ys)
    assert b == pytest.approx(0.93)
    assert math.exp(a) == pytest.approx(3.5)
    assert n == len(xs) and se == pytest.approx(0.0, abs=1e-9)


def test_the_fit_refuses_to_speak_from_two_points():
    b, a, se, n = loglog_fit([1, 2], [1, 2])
    assert n == 2 and math.isnan(b)


def test_size_for_reach_inverts_the_fit():
    xs = [2 ** k for k in range(4, 12)]
    ys = [3.5 * x ** 0.93 for x in xs]
    b, a, _, _ = loglog_fit(xs, ys)
    assert size_for_reach(3.5 * 1000 ** 0.93, b, a) == pytest.approx(1000)
    assert math.isnan(size_for_reach(100, float("nan"), a))
    assert math.isnan(size_for_reach(100, -0.5, a))     # a shrinking law is not invertible


# ------------------------------------------------------------------- the sizes
def test_memory_params_counts_both_layers():
    assert memory_params(128, 256) == 128 * 256 * 2
    assert math.isnan(memory_params(None, 256))


def test_arm_and_seed_are_read_off_the_artifact_names():
    assert hidden_from_arm("h512") == 512
    assert hidden_from_arm("bf5_shallow") is None
    assert seed_of("h128_s3.pt") == 3
    assert seed_of("h128.pt") is None


def test_sizes_come_from_what_the_run_recorded(tmp_path):
    (tmp_path / "cap_h64_s2.json").write_text(json.dumps(
        {"label": "cap_h64", "seed": 2, "memory_hidden_dim": 64, "dim": 128,
         "params": 400000}))
    (tmp_path / "cap_broken.json").write_text("{not json")
    got = sizes_from_training(str(tmp_path / "cap_*.json"))
    assert got[("h64", 2)]["hidden"] == 64          # the cap_ prefix is stripped
    assert got[("h64", 2)]["dim"] == 128
    assert len(got) == 1                            # the unreadable file is skipped


# -------------------------------------------------------------------- the unit
def test_one_latent_is_a_measured_fraction_of_a_second():
    """2000 val sequences of 64 latents from 2000 distinct 30 s segments."""
    assert SECONDS_PER_LATENT == pytest.approx(0.46875)
    assert 64 * SECONDS_PER_LATENT == pytest.approx(30.0)
    # The target is external and fixed: 5.9 h is ~45 000 latents to carry.
    assert EGOLIFEQA_MEDIAN_S / SECONDS_PER_LATENT == pytest.approx(45312, rel=1e-3)
