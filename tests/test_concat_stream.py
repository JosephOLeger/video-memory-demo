"""Re-joining EgoLife segments: parsing, run splitting, and window extraction."""

import pytest
import torch

from titans.worldmodel.concat_stream import (contiguous_runs, parse_segment,
                                             windows_from_runs)


def _name(person, day, h, m, s, f=0):
    return f"{person}/{day}/{day}_{person}_{h:02d}{m:02d}{s:02d}{f:02d}.mp4"


def test_parse_reads_the_frame_field_as_frames_not_hundredths():
    """`11094208` is 11:09:42 plus 8 frames, not 11:09:42.08."""
    p, d, t = parse_segment("A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4")
    assert (p, d) == ("A1_JAKE", "DAY1")
    assert t == pytest.approx(11 * 3600 + 9 * 60 + 42 + 8 / 30.0)


def test_parse_rejects_a_name_that_does_not_match():
    assert parse_segment("junk.mp4") is None
    assert parse_segment("A1/DAY1/DAY1_A2_11000000.mp4") is None   # person mismatch


def test_a_long_break_splits_a_run_but_the_normal_cadence_does_not():
    segs = [_name("P", "D1", 10, 0, 0), _name("P", "D1", 10, 2, 0),
            _name("P", "D1", 10, 4, 0),                    # 120 s cadence
            _name("P", "D1", 13, 0, 0), _name("P", "D1", 13, 2, 0)]   # 3 h break
    runs = contiguous_runs(segs)
    assert [len(idx) for _, idx in runs] == [3, 2]


def test_the_threshold_is_per_person_not_global():
    """One recorder runs at 90 s and another at 120 s; a single global threshold
    would either split the fast one into singletons or swallow the other's breaks."""
    segs = ([_name("FAST", "D1", 10, 0, 0), _name("FAST", "D1", 10, 1, 30),
             _name("FAST", "D1", 10, 3, 0)] +
            [_name("SLOW", "D1", 10, 0, 0), _name("SLOW", "D1", 10, 2, 0),
             _name("SLOW", "D1", 10, 4, 0)])
    runs = {k[0]: idx for k, idx in contiguous_runs(segs)}
    assert len(runs["FAST"]) == 3 and len(runs["SLOW"]) == 3


def test_runs_never_cross_a_person_or_a_day():
    """The train/val split is person-disjoint; a run crossing persons would break it."""
    segs = [_name("P1", "D1", 10, 0, 0), _name("P2", "D1", 10, 2, 0),
            _name("P1", "D2", 10, 4, 0)]
    runs = contiguous_runs(segs)
    assert len(runs) == 3
    assert all(len(idx) == 1 for _, idx in runs)


def test_segments_are_ordered_by_time_not_by_file_order():
    segs = [_name("P", "D1", 10, 4, 0), _name("P", "D1", 10, 0, 0),
            _name("P", "D1", 10, 2, 0)]
    (_, idx), = contiguous_runs(segs)
    assert idx == [1, 2, 0]


def test_windows_are_cut_out_of_a_run_laid_end_to_end():
    lat = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    runs = [(("P", "D1"), [0, 1, 2])]
    win, prov = windows_from_runs(runs, lat, length=6)
    assert win.shape == (2, 6, 2)
    flat = lat.reshape(-1, 2)
    assert torch.equal(win[0], flat[:6]) and torch.equal(win[1], flat[6:12])
    assert prov[0]["run_segments"] == 3


def test_a_run_shorter_than_one_window_is_dropped_not_padded():
    """Every row must be a genuine uninterrupted stretch."""
    lat = torch.zeros(5, 4, 2)
    runs = [(("P", "D1"), [0, 1]), (("P", "D1"), [2, 3, 4])]
    win, prov = windows_from_runs(runs, lat, length=10)
    assert win.shape[0] == 1
    assert prov[0]["run_segments"] == 3


def test_no_window_straddles_two_runs():
    lat = torch.cat([torch.zeros(2, 4, 1), torch.ones(2, 4, 1)], dim=0)
    runs = [(("P", "D1"), [0, 1]), (("P", "D2"), [2, 3])]
    win, _ = windows_from_runs(runs, lat, length=8)
    assert win.shape[0] == 2
    assert (win[0] == 0).all() and (win[1] == 1).all()
