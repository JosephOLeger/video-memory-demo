"""The judge's bookkeeping, without the 3B host.

Scoring bugs are the quiet kind: a benchmark that mis-sums is still a benchmark,
and it will happily report a number nobody can reproduce. These pin the parts that
do not need a model -- prompt shape, per-stratum accounting, and the fact that the
comparison is same-context-plus-memory rather than memory-instead-of-frames.
"""

from __future__ import annotations

from titans.graft.eval_lvb import LETTERS, build_prompt, format_report, summarise


def _item(**kw):
    d = {"question": "What did they do first?",
         "candidates": ["ran", "sat", "ate", "slept"]}
    d.update(kw)
    return d


def test_prompt_lists_every_option_with_its_letter():
    p = build_prompt(_item())
    for i, c in enumerate(["ran", "sat", "ate", "slept"]):
        assert f"{LETTERS[i]}. {c}" in p
    assert p.startswith("What did they do first?")
    assert "letter" in p.lower()


def test_prompt_handles_a_non_standard_option_count():
    p = build_prompt(_item(candidates=["a", "b", "c", "d", "e"]))
    assert "E. e" in p and "F." not in p


def test_summary_splits_by_duration_group():
    rows = [
        {"duration_group": 15, "gold": 0, "base": 0, "mem": 0},
        {"duration_group": 15, "gold": 1, "base": 0, "mem": 1},
        {"duration_group": 3600, "gold": 2, "base": 0, "mem": 2},
        {"duration_group": 3600, "gold": 3, "base": 3, "mem": 0},
    ]
    rep = summarise(rows)
    assert rep["all"]["n"] == 4
    assert rep["all"]["base"] == 0.5 and rep["all"]["mem"] == 0.75
    # the stratum split is the claim: short videos should not carry the long ones
    assert rep["15"]["base"] == 0.5 and rep["15"]["mem"] == 1.0
    assert rep["3600"]["base"] == 0.5 and rep["3600"]["mem"] == 0.5
    assert rep["15"]["delta"] == 0.5


def test_summary_without_a_graft_reports_only_the_baseline():
    rows = [{"duration_group": 60, "gold": 1, "base": 1}]
    rep = summarise(rows)
    assert "mem" not in rep["all"] and rep["all"]["base"] == 1.0
    assert "greffe" not in format_report(rep)


def test_report_renders_both_columns_when_a_graft_is_present():
    rep = summarise([{"duration_group": 600, "gold": 0, "base": 0, "mem": 1}])
    text = format_report(rep)
    assert "greffe" in text and "delta" in text and "600" in text


def test_empty_strata_are_dropped_rather_than_reported_as_zero():
    """A stratum with no question must not appear at 0 % -- that reads as a failure
    when it is an absence."""
    rep = summarise([{"duration_group": 15, "gold": 0, "base": 0, "mem": 0}])
    assert set(rep) == {"all", "15"}
