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


def test_write_count_travels_with_the_score():
    """The memory's reach is calibrated on a number of writes. A stratum scored far
    from the training write count was asked something it was never tuned for, and
    the accuracy alone does not show it."""
    rows = [{"duration_group": 15, "gold": 0, "base": 0, "mem": 0, "writes": 400},
            {"duration_group": 3600, "gold": 0, "base": 0, "mem": 0, "writes": 9000}]
    rep = summarise(rows)
    assert rep["15"]["writes"] == 400 and rep["3600"]["writes"] == 9000
    assert "ecritures" in format_report(rep)


def test_the_evaluation_builds_the_same_memory_as_the_training():
    """Nothing here changes a parameter shape, so a mismatch loads cleanly and is
    only visible in the numbers. One function, called by both scripts."""
    from titans.graft.branch import memory_kwargs_from_args

    saved = {"memory_hidden": 64, "alpha_init": 1.27e-4, "muon": "unit_squash",
             "muon_scale": 0.0, "cascade": True, "reservoirs": 6}
    kw = memory_kwargs_from_args(saved)
    assert kw["orthogonalize_momentum"] == "unit_squash"
    assert kw["orthogonalize_clip"] == 0.0, "0.0 est un clip reel, pas une absence"
    assert kw["use_cascade_momentum"] and kw["cascade_reservoirs"] == 6


def test_a_namespace_and_a_dict_give_the_same_memory():
    import argparse

    from titans.graft.branch import memory_kwargs_from_args
    ns = argparse.Namespace(memory_hidden=32, alpha_init=1e-3, muon="unit",
                            muon_scale=None, cascade=False, reservoirs=6)
    assert memory_kwargs_from_args(ns) == memory_kwargs_from_args(vars(ns))


def test_the_mismatched_control_never_pairs_a_video_with_itself():
    """LongVideoBench asks several questions about the same video, so a naive
    shift would hand some questions their own memory back -- which would be the
    measurement, not the control, and would quietly weaken the null."""
    from titans.graft.eval_lvb import donor_indices

    data = [{"video_path": f"v{i // 3}.mp4"} for i in range(12)]   # 3 questions/video
    idx = donor_indices(data)
    assert len(idx) == len(data)
    for i, j in enumerate(idx):
        assert data[j]["video_path"] != data[i]["video_path"], (i, j)


def test_the_mismatched_control_is_deterministic():
    """No seed to record, no drift between a run and its rerun."""
    from titans.graft.eval_lvb import donor_indices

    data = [{"video_path": f"v{i}.mp4"} for i in range(7)]
    assert donor_indices(data) == donor_indices(data)


def test_donor_choice_survives_a_single_video():
    """Degenerate input must not loop forever; it simply cannot find a donor."""
    from titans.graft.eval_lvb import donor_indices

    assert len(donor_indices([{"video_path": "v.mp4"}] * 3)) == 3


def test_the_context_control_holds_the_write_count_fixed():
    """Fewer writes would confound 'less information' with 'less writing': the
    memory's state magnitude grows with how often it is written."""
    import torch

    from titans.graft.eval_lvb import tile_to

    ctx = torch.arange(32 * 2 * 2).reshape(32, 2, 2)
    out = tile_to(ctx, 256)
    assert out.shape[0] == 256
    assert torch.equal(out[:32], ctx), "les images du contexte, a l'identique"
    assert torch.equal(out[32:64], ctx), "puis repetees, sans information neuve"


def test_tiling_never_grows_frames_it_was_given_enough_of():
    import torch

    from titans.graft.eval_lvb import tile_to
    assert tile_to(torch.zeros(300, 2, 2), 256).shape[0] == 256
