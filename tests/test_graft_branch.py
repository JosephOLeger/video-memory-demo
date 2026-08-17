"""Tests for the lateral memory branch (phase A of the VLM graft).

The properties that make the graft *measurable*, in the order they matter:

  1. **exact identity at init** -- the grafted model must be bit-identical to the
     frozen VLM before training, or "does the memory help?" has no baseline;
  2. the branch never writes to the host model's parameters;
  3. shapes and the carry survive being streamed segment by segment;
  4. a lowered write cadence still retrieves, and its phase survives segmentation
     -- the whole capacity argument rests on decoupling write from read.

Everything runs in float64 on tiny widths: these are contracts, not benchmarks.
"""

import pytest
import torch
import torch.nn as nn

from titans.graft import MemoryBranch, sane_memory_kwargs, stride_for_reach


def make_branch(d_model=16, d_mem=8, *, seed=0, double=True, **kw):
    torch.manual_seed(seed)
    branch = MemoryBranch(d_model, d_mem, chunk_size=kw.pop("chunk_size", 4), **kw)
    return branch.double() if double else branch


# ------------------------------------------------------------------- identity
def test_the_branch_is_an_exact_identity_at_init():
    """Not "close to": equal. A zero up-projection adds exactly zero."""
    branch = make_branch()
    h = torch.randn(2, 12, 16, dtype=torch.float64)
    out, _, _ = branch(h)
    assert branch.is_identity()
    assert torch.equal(out, h)


def test_identity_survives_a_populated_memory():
    """Even after writes, the residual stream is untouched while ``up`` is zero.

    This is the property that lets the memory warm up on a long video before any
    of its output reaches the host model.
    """
    branch = make_branch()
    h = torch.randn(2, 12, 16, dtype=torch.float64)
    out1, state, S = branch(h)
    out2, _, _ = branch(h, state, S, token_offset=12)
    assert torch.equal(out1, h) and torch.equal(out2, h)
    # ... and the memory really did move, so the equality above is not vacuous.
    fresh, _ = branch.init_carry(2)
    assert any(not torch.equal(state[k], fresh[k]) for k in state)


def test_a_trained_up_projection_stops_being_the_identity():
    """The guard reports what it claims to report."""
    branch = make_branch()
    with torch.no_grad():
        branch.up.weight.normal_()
    h = torch.randn(2, 8, 16, dtype=torch.float64)
    out, _, _ = branch(h)
    assert not branch.is_identity()
    assert not torch.equal(out, h)


# ------------------------------------------------------- the host stays frozen
def test_the_host_model_receives_no_gradient():
    """A frozen decoder stays frozen: only branch parameters get gradients."""
    branch = make_branch()
    host = nn.Linear(16, 16, bias=False).double()
    for p in host.parameters():
        p.requires_grad_(False)
    with torch.no_grad():                      # leave identity, else grads vanish
        branch.up.weight.normal_(std=0.1)

    out, _, _ = branch(host(torch.randn(2, 8, 16, dtype=torch.float64)))
    out.square().mean().backward()

    assert all(p.grad is None for p in host.parameters())
    assert branch.up.weight.grad is not None
    assert branch.down.weight.grad is not None
    named = dict(branch.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in named.items() if n.startswith("memory."))


# ----------------------------------------------------------------- shapes/carry
@pytest.mark.parametrize("B,N,d_model,d_mem", [(1, 8, 16, 8), (3, 12, 32, 4),
                                               (2, 5, 8, 8)])
def test_shapes_are_preserved(B, N, d_model, d_mem):
    branch = make_branch(d_model, d_mem)
    out, state, S = branch(torch.randn(B, N, d_model, dtype=torch.float64))
    assert out.shape == (B, N, d_model)
    assert all(v.shape[0] == B for v in state.values())
    assert set(state) == set(S)


def test_a_ragged_segment_is_written_not_dropped():
    """N not a multiple of chunk_size must still write its tail."""
    branch = make_branch(chunk_size=4)
    h = torch.randn(1, 6, 16, dtype=torch.float64)          # 4 + 2
    _, state_6, _ = branch(h)
    _, state_4, _ = branch(h[:, :4])
    assert any(not torch.allclose(state_6[k], state_4[k]) for k in state_6)


def test_the_input_contract_is_enforced():
    branch = make_branch()
    with pytest.raises(ValueError):
        branch(torch.randn(8, 16, dtype=torch.float64))     # missing batch dim
    with pytest.raises(ValueError):
        branch(torch.randn(1, 8, 7, dtype=torch.float64))   # wrong width
    with pytest.raises(ValueError):
        MemoryBranch(16, 8, write_stride=0)


# ------------------------------------------------------------- write vs read
def test_a_strided_write_still_reads_at_every_position():
    """Writing 1 token in 4 must not cost a single retrieval."""
    branch = make_branch(write_stride=4)
    with torch.no_grad():
        branch.up.weight.normal_(std=0.1)
    h = torch.randn(1, 8, 16, dtype=torch.float64)
    out, _, _ = branch(h)
    assert out.shape == h.shape
    assert not torch.allclose(out, h)                # every position moved
    assert (out - h).abs().amin(dim=-1).gt(0).all()


def test_a_strided_write_costs_less_memory_traffic_than_a_dense_one():
    """The stride is real: fewer writes means a different, less-driven state."""
    dense = make_branch(write_stride=1, seed=1)
    strided = make_branch(write_stride=4, seed=1)
    h = torch.randn(1, 8, 16, dtype=torch.float64)
    _, s_dense, _ = dense(h)
    _, s_strided, _ = strided(h)
    assert any(not torch.allclose(s_dense[k], s_strided[k]) for k in s_dense)


def test_the_stride_phase_survives_segmentation():
    """Streaming in two halves with ``token_offset`` == one call over the whole.

    Without the offset the phase resets per call and the cadence quietly stops
    being uniform -- the failure this argument would not survive. Run at
    ``chunk_size=1`` (fully online) so the comparison isolates the phase: at
    larger chunks a cut also moves the chunk-start weights the inner gradients
    are taken at, which is the documented mini-batch semantics, not a bug.
    """
    whole = make_branch(write_stride=3, chunk_size=1, seed=2)
    split = make_branch(write_stride=3, chunk_size=1, seed=2)
    split.load_state_dict(whole.state_dict())
    h = torch.randn(1, 12, 16, dtype=torch.float64)

    _, s_whole, _ = whole(h)                       # writes 0, 3, 6, 9
    _, s1, S1 = split(h[:, :7])                    # writes 0, 3, 6
    _, s_split, _ = split(h[:, 7:], s1, S1, token_offset=7)   # writes 9, not 7
    for k in s_whole:
        assert torch.allclose(s_whole[k], s_split[k])


def test_dropping_the_offset_breaks_the_cadence():
    """The guard above is not vacuous: without ``token_offset`` the phase resets."""
    split = make_branch(write_stride=3, chunk_size=1, seed=2)
    h = torch.randn(1, 12, 16, dtype=torch.float64)
    _, s1, S1 = split(h[:, :7])
    _, s_naive, _ = split(h[:, 7:], s1, S1)                   # writes 7 and 10
    _, s_phased, _ = split(h[:, 7:], s1, S1, token_offset=7)
    assert any(not torch.allclose(s_naive[k], s_phased[k]) for k in s_naive)


def test_an_explicit_write_mask_overrides_the_stride():
    branch = make_branch(write_stride=1, seed=3)
    ref = make_branch(write_stride=1, seed=3)
    h = torch.randn(1, 8, 16, dtype=torch.float64)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[::4] = True
    _, s_masked, _ = branch(h, write_mask=mask)
    _, s_strided, _ = ref(h[:, ::4])
    for k in s_masked:
        assert torch.allclose(s_masked[k], s_strided[k])
    with pytest.raises(ValueError):
        branch(h, write_mask=torch.ones(5, dtype=torch.bool))


# --------------------------------------------------------------- config guard
def test_the_branch_is_built_on_sane_not_on_library_defaults():
    """The standing rule: every arm is a delta on ``sane_ns``.

    Library defaults would silently restore ``normalize_output=True``, which
    empties the inner associative loss -- the exact thing SANE removes.
    """
    branch = make_branch()
    assert branch.memory.normalize_qk == "l2"
    assert branch.memory.normalize_output is False
    assert branch.memory.orthogonalize_momentum is True
    assert sane_memory_kwargs()["eta_init"] == 0.995
    assert sane_memory_kwargs(theta_scale=0.01)["theta_scale"] == 0.01


def test_the_retrieval_norm_is_on_by_default_and_optional():
    assert make_branch().norm_r is not None
    assert make_branch(normalize_retrieval=False).norm_r is None


# ------------------------------------------------------------------- planning
def test_stride_for_reach_converts_a_horizon_into_a_cadence():
    # 5.9 h at 1 fps, 32 tokens/frame, a memory good for 1605 writes:
    # 21240 frames / 1605 = 13.23 frames per write -> 423 tokens.
    assert stride_for_reach(5.9 * 3600, 1.0, 32, 1605) == 423
    # A memory that already reaches the horizon needs no subsampling.
    assert stride_for_reach(100, 1.0, 32, 10_000) == 32
    with pytest.raises(ValueError):
        stride_for_reach(0, 1.0, 32, 1605)
