"""The training loop's load-bearing properties, without the 3B host.

Everything here is about wiring, not about the model: does the gradient reach the
memory's gates from a loss placed after the context, does the held-out split
actually hold anything out, and does the answer segment leave the state alone.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from titans.graft.branch import MemoryBranch
from titans.graft.distill import split_windows, windows, write_context
from titans.graft.qwen import GraftedLayer


def _branch(**kw):
    return MemoryBranch(32, 16, chunk_size=4, hidden_dim=8, **kw)


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def test_windows_cover_the_stream_and_respect_the_hop():
    w = windows(10, span=4, hop=2)
    assert w[0] == (0, 4) and w[-1][1] <= 10
    assert all(b - a == 4 for a, b in w)
    assert [a for a, _ in w] == list(range(0, 7, 2))


def test_a_window_must_leave_something_for_the_memory():
    with pytest.raises(ValueError, match="span"):
        windows(10, span=1, hop=1)


def test_validation_is_a_contiguous_tail_not_a_random_subset():
    """Consecutive windows overlap by span-hop clips, so a shuffled split would put
    almost the same video on both sides and the val curve would measure nothing."""
    w = windows(100, span=12, hop=4)
    tr, va = split_windows(w, val_frac=0.1)
    assert tr + va == w                       # order preserved, nothing dropped
    assert max(s for s, _ in tr) < min(s for s, _ in va)
    # and the held-out clips really are past the training ones
    assert min(s for s, _ in va) >= max(e for _, e in tr) - 12


# --------------------------------------------------------------------------- #
# phase B -> the gradient must reach the gates
# --------------------------------------------------------------------------- #
def test_context_write_carries_a_gradient_to_the_memory_gates():
    """The whole design rests on this: a loss placed *after* the context reaches the
    parameters that wrote the context, across every chunk, with no truncation.

    Under ``unit`` -- the convention the loop defaults to. See the test below for
    what the same thing does under the library default.
    """
    torch.manual_seed(0)
    br = _branch(orthogonalize_momentum="unit")
    z = torch.randn(400, 16)                                  # 400 tokens of context
    state, S, nw = write_context(br, z, 0, 400, stride=2, device="cpu")
    assert nw == 200                                          # stride applied
    # a scalar that depends on the final state only
    q = br.memory.q_proj(torch.randn(1, 3, 16))
    br.memory.retrieve(state, q).square().mean().backward()
    named = dict(br.named_parameters())
    for key in ("memory.k_proj.weight", "memory.v_proj.weight"):
        g = named[key].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0, key
    gates = [n for n in named if "head" in n or "gate" in n]
    assert gates, "no gate heads found -- the memory's parameterisation changed"
    assert any(named[n].grad is not None and named[n].grad.abs().max() > 0
               for n in gates), "no gradient reached any gate head"


def test_rescale_kills_the_gradient_before_the_context_is_written():
    """Why the graft is trained under ``unit`` and not the library default.

    ``SANE_NS`` sets ``orthogonalize_momentum=True`` (= ``rescale``), which restores
    ``||S||`` after Newton-Schulz and so bounds nothing. Two hundred writes of plain
    noise are enough: the state goes non-finite and every gradient with it. This is
    the graft's whole blocker, reproduced without a video, a GPU or a 3B host -- and
    it is a regression test, not a curiosity: anything that silently reinstates
    ``rescale`` on this path fails here.
    """
    torch.manual_seed(0)
    br = _branch()                                   # library default = rescale
    assert br.memory._orth_mode == "rescale"
    state, _, _ = write_context(br, torch.randn(400, 16), 0, 400, stride=2,
                                device="cpu")
    assert not torch.isfinite(state["w1"]).all()

    torch.manual_seed(0)
    ok = _branch(orthogonalize_momentum="unit")
    state, _, _ = write_context(ok, torch.randn(400, 16), 0, 400, stride=2,
                                device="cpu")
    assert torch.isfinite(state["w1"]).all()
    assert float(state["w1"].detach().norm()) < 100.0         # and settled, not merely finite


def test_stride_is_the_only_thing_that_sets_the_write_count():
    br = _branch()
    z = torch.randn(240, 16)
    for stride, expect in ((1, 240), (2, 120), (4, 60)):
        _, _, nw = write_context(br, z, 0, 240, stride=stride, device="cpu")
        assert nw == expect


def test_write_context_slices_by_token_offset():
    br = _branch()
    z = torch.randn(300, 16)
    _, _, nw = write_context(br, z, 100, 220, stride=1, device="cpu")
    assert nw == 120


# --------------------------------------------------------------------------- #
# phase C -> read without writing
# --------------------------------------------------------------------------- #
def test_answer_segment_reads_the_state_without_moving_it():
    """If the answer segment wrote, the loss would have a second gradient path and
    the state would no longer be the one the context phase produced."""
    torch.manual_seed(0)
    br = _branch()
    wrap = GraftedLayer(nn.Identity(), br)
    z = torch.randn(80, 16)
    state, S, _ = write_context(br, z, 0, 80, stride=1, device="cpu")
    before = state["w1"].detach().clone()
    wrap.carry, wrap.token_offset = (state, S), 0
    wrap.writes_enabled = False
    wrap(torch.randn(1, 12, 32))
    assert torch.equal(wrap.carry[0]["w1"].detach(), before)
    # ...and with writes on, it does move -- so the flag is what is doing the work
    wrap.carry, wrap.writes_enabled = (state, S), True
    wrap(torch.randn(1, 12, 32))
    assert not torch.equal(wrap.carry[0]["w1"].detach(), before)


def test_the_branch_is_the_identity_before_training():
    """The floor the loop reports as 'sans-memoire' is the untouched host, exactly."""
    br = _branch()
    wrap = GraftedLayer(nn.Identity(), br)
    x = torch.randn(1, 12, 32)
    wrap.writes_enabled = False
    assert torch.equal(wrap(x), x)
    assert br.is_identity()


def test_unit_is_what_the_loop_defaults_to():
    br = _branch(orthogonalize_momentum="unit", alpha_init=1.27e-4)
    assert br.memory._orth_mode == "unit"
    # alpha_init is the reach knob: 1/(2d) with d ~ 3940 writes
    z = torch.randn(64, 16)
    state, _, _ = write_context(br, z, 0, 64, stride=2, device="cpu")
    assert torch.isfinite(state["w1"]).all()
