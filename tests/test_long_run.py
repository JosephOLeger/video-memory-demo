"""The unbounded-length inference probe: stream construction, carry threading, halt."""

import pytest
import torch

from titans.worldmodel.long_run import build_stream, run_stream, state_diagnostics
from titans.worldmodel.model import WorldModel


def _va(n=6, t=16, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, d, generator=g)


def _model(d=8):
    torch.manual_seed(0)
    return WorldModel(latent_dim=d, dim=d, depth=1, segment_size=4,
                      memory_num_layers=2).eval()


def test_loop_repeats_one_clip_and_shuffle_does_not():
    va = _va()
    loop = build_stream(va, reps=5, mode="loop")
    assert len(loop) == 5
    assert all(torch.equal(c, loop[0]) for c in loop)

    shuf = build_stream(va, reps=20, mode="shuffle")
    assert len({c[0, 0, 0].item() for c in shuf}) > 1


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="loop"):
        build_stream(_va(), reps=2, mode="reverse")


def test_carry_threads_across_repetitions():
    """The whole point: repetition r must see the memory built by 0..r-1.

    Compared against ``reset_each=True``, which is the same clips with the carry
    dropped -- if the two agree the carry is not threading and the probe measures
    nothing.
    """
    m, va = _model(), _va()
    clips = build_stream(va, reps=4, mode="loop")
    threaded = run_stream(m, clips)
    reset = run_stream(m, clips, reset_each=True)
    assert threaded[0]["l1"] == pytest.approx(reset[0]["l1"], rel=1e-6)
    assert threaded[-1]["l1"] != pytest.approx(reset[-1]["l1"], rel=1e-9)
    # with the carry dropped every repetition is the same computation
    assert reset[0]["l1"] == pytest.approx(reset[-1]["l1"], rel=1e-6)


def test_records_one_entry_per_repetition_with_diagnostics():
    m = _model()
    tr = run_stream(m, build_stream(_va(), reps=3, mode="loop"))
    assert [r["rep"] for r in tr] == [0, 1, 2]
    for r in tr:
        assert r["finite"] and r["b0"]["finite"]
        assert r["b0"]["pr"] >= 1.0            # participation ratio floors at rank 1
        assert 0.0 <= abs(r["b0"]["cos_ret"]) <= 1.0 + 1e-6
        assert r["b0"]["S_norm"] >= 0.0


def test_stream_halts_on_a_non_finite_state():
    """A blown-up carry must end the stream, not be carried through as NaN."""
    m = _model()
    real = m.forward
    calls = []

    def poisoned(z, carry=None, return_carry=False):
        pred, c = real(z, carry=carry, return_carry=True)
        calls.append(1)
        if len(calls) == 2:
            c = [({k: torch.full_like(v, float("nan")) for k, v in st.items()}, S)
                 for st, S in c]
        return (pred, c) if return_carry else pred

    m.forward = poisoned
    tr = run_stream(m, build_stream(_va(), reps=6, mode="loop"))
    assert len(tr) == 2
    assert not tr[-1]["b0"]["finite"]
    assert "non-finite" in tr[-1]["note"]


def test_state_diagnostics_covers_every_block():
    m = _model()
    z = _va(n=1, t=16)
    _, carry = m(z, return_carry=True)
    d = state_diagnostics(m, carry)
    assert set(d) == {f"b{i}" for i in range(len(m.blocks))}

def test_cycle_reuses_a_small_pool_round_robin():
    """The regime between loop and shuffle: repetition exists but is interleaved."""
    va = _va(n=20)
    clips = build_stream(va, reps=9, mode="cycle", pool=3)
    assert len(clips) == 9
    for r in range(9):
        assert torch.equal(clips[r], clips[r % 3])
    assert len({c[0, 0, 0].item() for c in clips}) == 3


def test_pool_larger_than_the_cache_is_clamped():
    clips = build_stream(_va(n=3), reps=6, mode="cycle", pool=99)
    assert len({c[0, 0, 0].item() for c in clips}) <= 3


def test_memoff_baseline_matches_the_uncached_control():
    """The cache must be exact, not an approximation: same numbers, fewer forwards."""
    from titans.worldmodel.long_run import memoff_baseline
    m, va = _model(), _va()
    clips = build_stream(va, reps=6, mode="cycle", pool=2)
    cached = memoff_baseline(m, clips)
    plain = run_stream(m, clips, reset_each=True)
    assert [r["rep"] for r in cached] == [r["rep"] for r in plain]
    for c, q in zip(cached, plain):
        assert c["l1"] == pytest.approx(q["l1"], rel=1e-12)


def test_diag_every_subsamples_state_but_never_the_loss():
    m = _model()
    tr = run_stream(m, build_stream(_va(), reps=7, mode="loop"), diag_every=3)
    assert all("l1" in r for r in tr)
    assert [r["rep"] for r in tr if "b0" in r] == [0, 3, 6]   # 6 is the last rep
