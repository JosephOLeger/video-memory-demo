"""The queue builder must refuse to guess.

Fifty GPU-hours ride on the bracket being real, so the failure that matters here is
not a crash -- it is emitting a plausible-looking queue from probes that never
straddled the transition.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_traj_queue.py"


def _probe(d: Path, eta, seed, loop_finite, cycle_finite=True):
    (d / "taucrit").mkdir(exist_ok=True)
    p = d / "taucrit" / ("nocas_e%s_s%d.json" % (eta, seed))
    p.write_text(json.dumps({
        "eta_init": eta, "cascade": False,
        "summary": {"loop": {"all_finite": loop_finite},
                    "cycle": {"all_finite": cycle_finite}}}))


def _run(root, out):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                           "--out", str(out)],
                          capture_output=True, text=True, check=True).stdout


def test_a_bracketed_sweep_queues_three_etas_and_three_seeds(tmp_path):
    for s in (0, 1, 2):
        _probe(tmp_path, 0.95, s, loop_finite=True)
        _probe(tmp_path, 0.98, s, loop_finite=False)
    out = tmp_path / "jobs_traj.txt"
    log = _run(tmp_path, out)
    assert "tau* entre 20 et 50" in log
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 9
    # the midpoint is geometric in tau, not arithmetic in eta
    assert "--eta-init 0.9684" in out.read_text()
    for ln in lines:
        assert "--steps 10000" in ln and "--frozen-every 1250" in ln


def test_an_unbracketed_sweep_writes_nothing(tmp_path):
    """Every arm survived: the threshold is above the sweep, not inside it."""
    for eta in (0.9, 0.95):
        _probe(tmp_path, eta, 0, loop_finite=True)
    out = tmp_path / "jobs_traj.txt"
    log = _run(tmp_path, out)
    assert "pas d'encadrement" in log and "tous survivent" in log
    assert not out.exists()


def test_all_diverging_is_also_refused(tmp_path):
    for eta in (0.98, 0.99):
        _probe(tmp_path, eta, 0, loop_finite=False)
    out = tmp_path / "jobs_traj.txt"
    assert "tous divergent" in _run(tmp_path, out)
    assert not out.exists()


def test_the_harsher_cycle_stream_is_reported_separately(tmp_path):
    """tau* depends on the input statistics; a single number would hide that."""
    _probe(tmp_path, 0.70, 0, loop_finite=True, cycle_finite=True)
    _probe(tmp_path, 0.90, 0, loop_finite=True, cycle_finite=False)
    _probe(tmp_path, 0.98, 0, loop_finite=False, cycle_finite=False)
    log = _run(tmp_path, tmp_path / "jobs_traj.txt")
    assert "encadrement : tau* entre 10 et 50" in log        # loop
    assert "flux cycle, plus dur : tau* entre 3 et 10" in log


def test_a_cascade_probe_is_not_read_as_a_scalar_eta(tmp_path):
    """A cascade arm carries an eta_init too; counting it would poison that row."""
    (tmp_path / "taucrit").mkdir()
    (tmp_path / "taucrit" / "cas_r1.0_s0.json").write_text(json.dumps({
        "eta_init": 0.95, "cascade": True,
        "summary": {"loop": {"all_finite": False}}}))
    _probe(tmp_path, 0.95, 0, loop_finite=True)
    _probe(tmp_path, 0.98, 0, loop_finite=False)
    log = _run(tmp_path, tmp_path / "jobs_traj.txt")
    row = next(l.split() for l in log.splitlines() if l.startswith("0.950"))
    assert row[2] == "1"        # one seed, not two
    assert row[3] == "0.00"     # and it survived -- the cascade's NaN did not leak in
