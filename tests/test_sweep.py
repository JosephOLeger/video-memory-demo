"""Fast smoke test for the 3-config difficulty sweep harness.

Runs the sweep at a tiny scale (few steps, few difficulty points, all 3 configs,
all task families) to confirm it executes end-to-end and produces well-formed
output. Does NOT validate architecture quality -- just harness correctness.
"""

import json

from titans.membench.sweep import CONFIGS, main, run_sweep
from titans.membench.tasks import fuzzy_recall, selective_copy


def test_new_mad_tasks_well_formed():
    for b in (selective_copy(batch=8, num_tokens=5, num_slots=20, segment_size=8, seed=0),
              fuzzy_recall(batch=8, num_pairs=5, segment_size=8, key_vocab=24, seed=0)):
        assert b.x.shape == b.y.shape == b.mask.shape
        assert b.x.shape[1] % 8 == 0
        assert b.mask.any(dim=1).all()
        assert int(b.x.max()) < b.vocab_size
        assert (b.y[~b.mask] == 0).all()


def test_sweep_smoke_runs():
    records = run_sweep(smoke=True, log=False)
    tasks = {r["task"] for r in records}
    configs = {r["config"] for r in records}
    assert tasks == {"mqar", "selective_copy", "fuzzy_recall"}
    assert configs == set(CONFIGS)
    for r in records:
        assert 0.0 <= r["acc_mean"] <= 1.0
        assert r["num_params"] > 0
        assert r["chance"] > 0.0

    # baseline and uncoupled (independent topology) have identical param counts --
    # the filter bank adds only non-trained buffers -- so the coupled config's
    # beta_head is the only param-count difference.
    by = {(r["task"], r["difficulty"], r["config"]): r for r in records}
    for (task, dval, cfg), r in by.items():
        if cfg == "baseline":
            assert r["num_params"] == by[(task, dval, "uncoupled")]["num_params"]
            assert r["num_params"] <= by[(task, dval, "coupled")]["num_params"]


def test_sweep_main_writes_json(tmp_path):
    out = tmp_path / "smoke.json"
    main(["--smoke", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert "records" in payload and "configs" in payload
    assert len(payload["records"]) > 0


def test_sweep_converge_smoke_runs():
    # Convergence mode at tiny scale: same 3 configs, but each cell stops on the
    # plateau/ceiling rule instead of a fixed step budget. Just checks the harness
    # and the extra bookkeeping fields are well-formed, not architecture quality.
    records = run_sweep(smoke=True, converge=True, log=False)
    assert {r["config"] for r in records} == set(CONFIGS)
    for r in records:
        assert r["converge"] is True
        assert r["stopped"] in {"ceiling", "plateau", "max_steps"}
        assert 0 <= r["converge_steps"] <= 6  # smoke cap
        assert r["steps"] == r["converge_steps"]
        assert 0.0 <= r["best_acc"] <= 1.0
        # trajectory starts at the step-0 (random-init) eval.
        assert r["trajectory"][0][0] == 0


def test_sweep_resume_skips_done_cells(tmp_path):
    # A checkpoint that already contains one cell should cause resume=True to skip
    # exactly that cell and re-train the rest.
    out = tmp_path / "resume.json"
    run_sweep(smoke=True, converge=True, log=False,
              tasks={"mqar"}, configs={"baseline"}, checkpoint_path=out)
    first = json.loads(out.read_text())["records"]
    assert len(first) == 1

    fresh = run_sweep(smoke=True, converge=True, log=False,
                      tasks={"mqar"}, checkpoint_path=out, resume=True)
    # baseline already on disk -> skipped (not returned); the other configs run.
    assert {r["config"] for r in fresh} == set(CONFIGS) - {"baseline"}
    merged = json.loads(out.read_text())["records"]
    assert {r["config"] for r in merged} == set(CONFIGS)
