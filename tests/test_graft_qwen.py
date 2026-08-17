"""Tests for the wiring, against a stand-in with the host's awkward shape.

The stand-in matters more than it looks. A decoder layer in ``transformers`` takes
positional *and* keyword arguments and may return either a bare tensor or a tuple
whose tail carries caches -- and the failure mode of a wrapper is not a crash, it
is silently dropping one of those and changing the model's output. So the fake
layer here asserts it received its kwargs and returns a tuple with a tail.

Qwen itself is not downloaded in CI; the one thing that cannot be faked -- that
the real model's logits are unchanged -- is what ``identity_check`` runs on the box.
"""

import pytest
import torch
import torch.nn as nn

from titans.graft import MemoryBranch
from titans.graft.qwen import (
    GraftedLayer,
    branch_report,
    default_layer,
    disabled,
    find_decoder_layers,
    freeze_host,
    graft,
    identity_check,
    ungraft,
)


class FakeLayer(nn.Module):
    """Shaped like a decoder layer: kwargs in, tuple with a tail out."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.saw_kwargs = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kw):
        self.saw_kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
        return (self.lin(hidden_states), "cache-tail")


class FakeText(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(d) for _ in range(n)])


class FakeInner(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.language_model = FakeText(d, n)
        self.visual = nn.Linear(d, d)          # a tower we must never graft onto


class FakeVLM(nn.Module):
    """``model.model.language_model.layers``, as transformers v5 lays it out."""

    def __init__(self, d=16, n=6):
        super().__init__()
        self.model = FakeInner(d, n)
        self.d = d

    def forward(self, hidden_states, **kw):
        h = hidden_states
        for layer in self.model.language_model.layers:
            out = layer(h, **kw)
            h = out[0] if isinstance(out, tuple) else out
        return type("Out", (), {"logits": h})()


def make(d=16, d_mem=8, n=6, seed=0):
    torch.manual_seed(seed)
    return FakeVLM(d, n).double(), MemoryBranch(d, d_mem, chunk_size=4).double()


# --------------------------------------------------------------------- finding
def test_the_decoder_layers_are_found_not_guessed():
    model, _ = make()
    assert find_decoder_layers(model) is model.model.language_model.layers


def test_a_model_with_no_decoder_stack_says_so():
    with pytest.raises(AttributeError, match="could not find the decoder layers"):
        find_decoder_layers(nn.Linear(4, 4))


def test_the_default_site_is_two_thirds_of_depth():
    assert default_layer(36) == 24          # Qwen2.5-VL-3B, read off its config
    assert default_layer(28) == 19
    assert default_layer(1) == 0


# -------------------------------------------------------------------- identity
def test_the_grafted_model_is_bit_identical_at_init():
    model, branch = make()
    x = torch.randn(2, 12, 16, dtype=torch.float64)
    rep = identity_check(model, branch, {"hidden_states": x}, layer=4)
    assert rep["identical"] and rep["max_abs_diff"] == 0.0
    assert rep["layer"] == 4 and rep["wrote_tokens"] == 12
    assert rep["memory_moved"]              # the memory ran; it just contributed 0


def test_identity_check_leaves_the_model_ungrafted():
    model, branch = make()
    before = list(model.model.language_model.layers)
    identity_check(model, branch, {"hidden_states": torch.randn(1, 8, 16,
                                                               dtype=torch.float64)})
    assert list(model.model.language_model.layers) == before


def test_a_trained_branch_stops_being_identical():
    """The check is not vacuous: give `up` weight and the logits must move."""
    model, branch = make()
    with torch.no_grad():
        branch.up.weight.normal_(std=0.1)
    rep = identity_check(model, branch, {"hidden_states": torch.randn(2, 8, 16,
                                                                     dtype=torch.float64)},
                         layer=4)
    assert not rep["identical"] and rep["max_abs_diff"] > 0


# --------------------------------------------------------------------- wrapper
def test_the_wrapper_passes_the_layer_its_keyword_arguments():
    """A dropped attention_mask would not crash -- it would quietly change the model."""
    model, branch = make()
    wrapper = graft(model, branch, layer=3)
    mask = torch.ones(2, 9, dtype=torch.float64)
    model(torch.randn(2, 9, 16, dtype=torch.float64), attention_mask=mask)
    assert wrapper.layer.saw_kwargs["attention_mask"] is mask


def test_the_wrapper_preserves_the_tail_of_the_layers_output():
    model, branch = make()
    wrapper = graft(model, branch, layer=2)
    out = wrapper(torch.randn(1, 4, 16, dtype=torch.float64))
    assert isinstance(out, tuple) and out[1] == "cache-tail"


def test_a_layer_returning_a_bare_tensor_is_handled():
    model, branch = make()
    layers = model.model.language_model.layers
    layers[1].forward = lambda h, **kw: h * 2.0        # no tuple, no tail
    wrapper = graft(model, branch, layer=1)
    out = wrapper(torch.randn(1, 4, 16, dtype=torch.float64))
    assert isinstance(out, torch.Tensor)


def test_the_carry_advances_across_calls_and_resets():
    model, branch = make()
    wrapper = graft(model, branch, layer=4)
    x = torch.randn(1, 8, 16, dtype=torch.float64)
    wrapper(x)
    assert wrapper.token_offset == 8 and wrapper.carry[0] is not None
    wrapper(x)
    assert wrapper.token_offset == 16
    wrapper.reset()
    assert wrapper.token_offset == 0 and wrapper.carry == (None, None)


def test_disabling_the_branch_passes_the_layer_straight_through():
    model, branch = make()
    with torch.no_grad():
        branch.up.weight.normal_(std=0.1)
    wrapper = graft(model, branch, layer=4)
    x = torch.randn(1, 6, 16, dtype=torch.float64)
    with disabled(wrapper):
        off = wrapper(x)[0]
    assert torch.equal(off, wrapper.layer(x)[0])
    assert wrapper.enabled                              # restored on exit


# ---------------------------------------------------------------- install/undo
def test_ungrafting_restores_the_original_module():
    model, branch = make()
    original = model.model.language_model.layers[4]
    wrapper = graft(model, branch, layer=4)
    assert isinstance(model.model.language_model.layers[4], GraftedLayer)
    ungraft(model, wrapper)
    assert model.model.language_model.layers[4] is original


def test_a_site_cannot_be_grafted_twice_and_must_exist():
    model, branch = make()
    graft(model, branch, layer=4)
    with pytest.raises(RuntimeError, match="already grafted"):
        graft(model, branch, layer=4)
    with pytest.raises(IndexError):
        graft(model, MemoryBranch(16, 8).double(), layer=99)


# ------------------------------------------------------------------- freezing
def test_grafting_freezes_the_host_but_not_the_branch():
    model, branch = make()
    graft(model, branch, layer=4)
    host = model.model.language_model.layers
    assert all(not p.requires_grad for i, l in enumerate(host) if i != 4
               for p in l.parameters())
    assert not any(p.requires_grad for p in model.model.visual.parameters())
    assert any(p.requires_grad for p in branch.parameters())


def test_only_the_branch_receives_gradient_through_the_whole_model():
    model, branch = make()
    with torch.no_grad():
        branch.up.weight.normal_(std=0.1)
    graft(model, branch, layer=2)
    model(torch.randn(2, 8, 16, dtype=torch.float64)).logits.square().mean().backward()
    for name, p in model.named_parameters():
        if "branch." in name:
            continue
        assert p.grad is None, name
    assert branch.up.weight.grad.abs().sum() > 0


def test_freeze_host_reports_what_it_froze():
    model, _ = make()
    n = freeze_host(model)
    assert n == len(list(model.parameters()))
    assert freeze_host(model) == 0           # idempotent


# ------------------------------------------------------------------ accounting
def test_the_parameter_bill_is_reported_as_a_share_of_the_host():
    _, branch = make(d=2048, d_mem=128)
    rep = branch_report(branch, host_params=3_000_000_000)
    assert rep["branch_trainable"] > 0
    assert rep["pct_of_host"] == pytest.approx(
        100 * rep["branch_trainable"] / 3e9)
    assert rep["pct_of_host"] < 0.1          # the claim: well under a tenth of a percent
