"""The comparison table: family assignment, pooling, and the filters that guard it."""

import json

import pytest

from titans.worldmodel.vv_compare import agg, arm_of, spc


def test_leak_none_is_the_cascade_s_eta_one():
    """Both families are indexed by 'how little it forgets', so none must map to 0."""
    assert arm_of({"cascade": True, "leak": "none", "leak_rate": 1.0}) == ("cascade", 0.0)
    assert arm_of({"cascade": True, "leak": None}) == ("cascade", 0.0)
    assert arm_of({"cascade": True, "leak": "surprise", "leak_rate": 0.3}) == ("cascade", 0.3)


def test_a_missing_eta_init_is_the_learned_default_not_zero():
    """eta_init=None means the head keeps its init, which sits at ~0.53 -- reading
    it as 0 would place 41 runs at the wrong end of the axis."""
    assert arm_of({"cascade": False, "eta_init": None}) == ("scalaire", 0.53)
    assert arm_of({"cascade": False, "eta_init": 0.9}) == ("scalaire", 0.9)


def test_space_survives_both_shapes_it_has_been_written_in():
    assert spc({"space": {"kind": "pca", "k": 128}}) == "pca"
    assert spc({"space": None}) == "None"


def test_agg_reports_sem_and_ignores_non_finite():
    m, sem, n = agg([1.0, 2.0, 3.0, float("nan"), None])
    assert m == pytest.approx(2.0)
    assert n == 3
    assert sem == pytest.approx(1.0 / 3 ** 0.5)


def test_agg_on_a_single_value_gives_no_sem_rather_than_zero():
    """A lone seed must not read as a perfectly precise measurement."""
    m, sem, n = agg([0.2])
    assert m == 0.2 and n == 1 and sem != sem     # NaN


def test_load_skips_the_margins_list_that_sits_in_its_own_glob(tmp_path, monkeypatch):
    """vv_margins.json is a list and matches vv_*.json -- it must not crash the reader."""
    from titans.worldmodel.vv_compare import load

    (tmp_path / "vv_a.json").write_text(json.dumps({"label": "a"}))
    (tmp_path / "vv_margins.json").write_text(json.dumps([{"margin": 3.0}]))
    (tmp_path / "vv_bad.json").write_text("{not json")
    monkeypatch.chdir(tmp_path)
    assert load("vv_*.json") == [{"label": "a"}]
