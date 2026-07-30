"""Checkpointing and truncated BPTT.

These guard the two properties a long run depends on and that are easy to get
subtly wrong: that a resumed run continues the *same* experiment rather than a
different one, and that windowing bounds memory without silently changing the
objective.
"""

import torch
import torch.nn.functional as F

from titans.worldmodel.model import WorldModel
from titans.worldmodel.training import (
    load_checkpoint, save_checkpoint, tbptt_step, train_loop,
)


def _model(dim=16, depth=1, seg=4):
    torch.manual_seed(0)
    return WorldModel(latent_dim=dim, dim=dim, depth=depth, segment_size=seg,
                      io_projection=False)


def _data(b=4, t=16, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, d, generator=g).cumsum(1) * 0.1


# ------------------------------------------------------------- checkpointing
def test_checkpoint_roundtrip_restores_weights_and_step(tmp_path):
    m, z = _model(), _data()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad(); F.l1_loss(m(z)[:, :-1], z[:, 1:]).backward(); opt.step()
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt, step=3, tag="unit")

    m2 = _model()
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-2)
    step, meta = load_checkpoint(p, m2, opt2)
    assert step == 3 and meta["tag"] == "unit"
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_checkpoint_write_is_atomic(tmp_path):
    """A previous checkpoint must survive; no .tmp debris is left behind."""
    p = tmp_path / "ck.pt"
    m, opt = _model(), None
    save_checkpoint(p, m, opt, step=1)
    first = p.read_bytes()
    save_checkpoint(p, m, opt, step=2)
    assert p.exists() and p.read_bytes() != first
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_continues_the_same_trajectory(tmp_path):
    """8 steps straight must equal 4 steps + resume + 4 steps.

    Without RNG restoration the resumed run draws different mini-batches and
    silently becomes a different experiment -- the failure this test exists for.
    """
    z = _data(b=8, t=16)
    p = tmp_path / "ck.pt"

    torch.manual_seed(1)
    a = _model(); oa = torch.optim.Adam(a.parameters(), lr=1e-2)
    train_loop(a, z, steps=8, opt=oa, batch_size=4)

    torch.manual_seed(1)
    b = _model(); ob = torch.optim.Adam(b.parameters(), lr=1e-2)
    train_loop(b, z, steps=4, opt=ob, batch_size=4, ckpt_path=p, ckpt_every=4)
    b2 = _model(); ob2 = torch.optim.Adam(b2.parameters(), lr=1e-2)
    train_loop(b2, z, steps=8, opt=ob2, batch_size=4, ckpt_path=p, ckpt_every=4)

    for pa, pb in zip(a.parameters(), b2.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_resume_is_a_noop_when_already_finished(tmp_path):
    z, p = _data(), tmp_path / "ck.pt"
    m = _model(); o = torch.optim.Adam(m.parameters())
    train_loop(m, z, steps=3, opt=o, ckpt_path=p, ckpt_every=1)
    r = train_loop(m, z, steps=3, opt=o, ckpt_path=p, ckpt_every=1)
    assert r["resumed_from"] == 3 and r["steps"] == 3


# --------------------------------------------------------------- truncated BPTT
def test_window_equal_to_sequence_matches_plain_backward():
    """With no truncation the windowed path must be the plain one, exactly."""
    z = _data(b=2, t=16)
    a = _model(); b = _model()
    F.l1_loss(a(z)[:, :-1], z[:, 1:]).backward()
    tbptt_step(b, z, horizon=1, window=None)
    for pa, pb in zip(a.parameters(), b.parameters()):
        ga = torch.zeros_like(pa) if pa.grad is None else pa.grad
        gb = torch.zeros_like(pb) if pb.grad is None else pb.grad
        assert torch.allclose(ga, gb, atol=1e-6)


def test_windowing_still_produces_finite_gradients_everywhere():
    z = _data(b=2, t=32)
    m = _model()
    tbptt_step(m, z, horizon=2, window=8)
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_memory_carry_crosses_windows_but_gradient_does_not():
    """The state must thread across windows -- that is the whole point --
    while the graph is cut, which is what bounds memory."""
    z = _data(b=2, t=16)
    m = _model()
    carry = None
    seen = []
    for start in (0, 8):
        _, carry = m(z[:, start:start + 8], carry=carry, return_carry=True)
        seen.append(carry[0][0]["w1"].clone())
        carry = [({k: v.detach() for k, v in s.items()},
                  {k: v.detach() for k, v in S.items()}) for s, S in carry]
    assert not torch.allclose(seen[0], seen[1])      # state evolved across windows
    assert not carry[0][0]["w1"].requires_grad       # graph cut


def test_windowing_bounds_activation_memory():
    """Smaller windows must hold fewer scan steps in the graph.

    Proxy for VRAM (which is CUDA-only): count tensors retained by autograd via
    the number of grad_fn nodes reachable from the loss is awkward, so use peak
    allocation on CUDA when present, else assert the loop simply runs.
    """
    if not torch.cuda.is_available():
        z = _data(b=2, t=32).cuda() if False else _data(b=2, t=32)
        m = _model()
        tbptt_step(m, z, horizon=1, window=8)
        return
    z = _data(b=2, t=64, d=16).cuda()
    peaks = []
    for w in (8, 64):
        m = _model().cuda()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        tbptt_step(m, z, horizon=1, window=w)
        peaks.append(torch.cuda.max_memory_allocated())
    assert peaks[0] < peaks[1], "windowing did not reduce retained activations"
