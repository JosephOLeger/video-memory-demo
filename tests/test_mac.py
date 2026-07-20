"""Tests for the Phase-2 MAC (Memory As Context) block.

The three core properties (per the task spec):
  1. shape preservation across several configs;
  2. causality -- outputs before a modified position are unchanged, including
     across segment boundaries (the key correctness check);
  3. gradient flow -- every outer parameter receives a real (non-zero) gradient,
     proving the functional inner-loop update does not detach the outer graph.
"""

import math

import pytest
import torch

from titans import TitansMACBlock


def make_block(dim=8, num_heads=2, segment_size=4, num_persistent_tokens=3,
               memory_hidden_dim=12, memory_num_layers=2, memory_chunk_size=None,
               seed=0, double=True):
    torch.manual_seed(seed)
    block = TitansMACBlock(
        dim=dim,
        num_heads=num_heads,
        segment_size=segment_size,
        num_persistent_tokens=num_persistent_tokens,
        memory_hidden_dim=memory_hidden_dim,
        memory_num_layers=memory_num_layers,
        memory_chunk_size=memory_chunk_size,
    )
    if double:
        block.double()
    return block


# ---------------------------------------------------------------- shape test
@pytest.mark.parametrize(
    "B,N,C,d,heads,Np",
    [
        (2, 8, 4, 8, 2, 3),
        (1, 12, 6, 12, 3, 0),
        (3, 6, 2, 16, 4, 5),
        (2, 16, 16, 8, 1, 2),
    ],
)
def test_output_shape(B, N, C, d, heads, Np):
    block = make_block(dim=d, num_heads=heads, segment_size=C,
                       num_persistent_tokens=Np, memory_hidden_dim=d, seed=B + N)
    x = torch.randn(B, N, d, dtype=torch.float64)
    out, state, S = block(x)
    assert out.shape == x.shape
    # state / S carry has a leading batch dim matching B.
    for key in state:
        assert state[key].shape[0] == B
        assert S[key].shape[0] == B


def test_many_segments_stay_finite():
    """Threading memory across many segments must not blow up to NaN/Inf.

    Regression guard: the online memory update accumulates momentum across
    segments; without the RMS-normalised memory output ``M(q)`` and the
    associative loss grow ~geometrically per segment and overflow to NaN once the
    sequence spans enough segments (surfaced first on 8-segment video latents).
    A single-precision forward over 16 segments must stay finite.
    """
    torch.manual_seed(0)
    B, C, d, num_seg = 2, 8, 32, 16
    block = make_block(dim=d, num_heads=4, segment_size=C,
                       num_persistent_tokens=2, memory_hidden_dim=d,
                       memory_num_layers=2, seed=0, double=False)
    x = torch.randn(B, C * num_seg, d)  # float32, like real training
    out, state, S = block(x)
    assert torch.isfinite(out).all(), "block output is not finite over many segments"
    for key in state:
        assert torch.isfinite(state[key]).all(), f"memory state {key} not finite"


def test_requires_divisible_length():
    block = make_block(dim=8, segment_size=4)
    x = torch.randn(2, 10, 8, dtype=torch.float64)  # 10 % 4 != 0
    with pytest.raises(ValueError):
        block(x)


# ------------------------------------------------------------- causality test
def test_causality_across_segments():
    """Changing a token at the start of segment ``s`` must leave every output at
    a position < s*C bit-for-bit (allclose) unchanged, including all earlier
    segments. This catches future-info leakage through the persistent/history
    concat, the attention mask, or the causal conv.
    """
    B, N, C, d = 2, 12, 4, 8
    block = make_block(dim=d, num_heads=2, segment_size=C,
                       num_persistent_tokens=3, memory_hidden_dim=d, seed=7)
    block.eval()

    x1 = torch.randn(B, N, d, dtype=torch.float64)
    # Modify from the start of segment s=2 onwards (position j = s*C = 8):
    j = 2 * C
    x2 = x1.clone()
    x2[:, j:, :] = torch.randn(B, N - j, d, dtype=torch.float64)

    out1, _, _ = block(x1)
    out2, _, _ = block(x2)

    assert torch.allclose(out1[:, :j], out2[:, :j], rtol=1e-10, atol=1e-10), (
        "future tokens leaked into earlier outputs: max |diff| = "
        f"{(out1[:, :j] - out2[:, :j]).abs().max().item():.3e}"
    )
    # Sanity: the change *does* affect positions >= j (otherwise the test is void).
    assert not torch.allclose(out1[:, j:], out2[:, j:])


def test_causality_single_token():
    """Same guarantee for a single differing token at a segment boundary."""
    B, N, C, d = 1, 8, 4, 8
    block = make_block(dim=d, num_heads=1, segment_size=C,
                       num_persistent_tokens=2, memory_hidden_dim=d, seed=11)
    block.eval()

    x1 = torch.randn(B, N, d, dtype=torch.float64)
    j = C  # first position of segment 1
    x2 = x1.clone()
    x2[:, j, :] += 1.0

    out1, _, _ = block(x1)
    out2, _, _ = block(x2)
    assert torch.allclose(out1[:, :j], out2[:, :j], rtol=1e-10, atol=1e-10)


# --------------------------------------------------------- gradient-flow test
def test_gradient_flow():
    """Every outer parameter must receive a non-None, non-all-zero gradient."""
    B, N, C, d = 2, 8, 4, 8
    block = make_block(dim=d, num_heads=2, segment_size=C,
                       num_persistent_tokens=3, memory_hidden_dim=12,
                       memory_num_layers=2, seed=3)
    x = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)

    out, _, _ = block(x)
    out.sum().backward()

    missing, zero = [], []
    for name, p in block.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.any(p.grad != 0):
            zero.append(name)
    assert not missing, f"parameters with no gradient: {missing}"
    assert not zero, f"parameters with all-zero gradient: {zero}"
    assert x.grad is not None


# --------------------------------------------------- optional trainability toy
def test_toy_trainability():
    """A few Adam steps on a synthetic copy-ish task should reduce the loss."""
    torch.manual_seed(0)
    B, N, C, d = 4, 8, 4, 8
    block = make_block(dim=d, num_heads=2, segment_size=C,
                       num_persistent_tokens=2, memory_hidden_dim=16,
                       memory_num_layers=2, seed=0, double=False)
    x = torch.randn(B, N, d)
    target = torch.roll(x, shifts=1, dims=1)  # predict the previous token

    opt = torch.optim.Adam(block.parameters(), lr=1e-2)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        out, _, _ = block(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert not math.isnan(losses[-1])
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_memory_update_bias_false_freezes_bias_but_not_weights():
    """``memory_update_bias=False`` keeps M's bias terms fixed at M_0 across the
    whole inner-loop recurrence, while weights still update online, and the
    outer loop can still learn the (fixed-at-test-time) bias via backprop."""
    torch.manual_seed(0)
    block = make_block(dim=16, segment_size=4, memory_hidden_dim=12,
                       memory_num_layers=2, seed=0, double=False)
    block.memory.update_bias = False  # cheap toggle without rebuilding the block

    x = torch.randn(2, 16, 16)
    out, state, _ = block(x)
    assert torch.isfinite(out).all()

    m0 = block.memory.memory.init_state(2)
    for key in state:
        if key.startswith("b"):
            assert torch.allclose(state[key], m0[key]), f"{key} should stay frozen"
        else:
            assert not torch.allclose(state[key], m0[key]), f"{key} should still update"

    # outer loop can still learn M_0's bias via backprop (only frozen online).
    for p in block.parameters():
        p.grad = None
    out.pow(2).mean().backward()
    g = block.memory.memory.params["b1"].grad
    assert g is not None and torch.isfinite(g).all()


def test_decouple_memory_update_forward_backward_and_changes_output():
    """decouple_memory_update=True runs end-to-end with finite gradients, and
    differs numerically from the default (paper-faithful) path -- it computes
    the memory-update signal from [persistent; segment] only, severing the
    retrieve->attend->update feedback loop identified as driving the memory's
    rank-1 collapse, while leaving the output path (y_i-with-h_i, gate) intact."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 16)

    block_off = make_block(dim=16, segment_size=4, seed=1, double=False)
    out_off, _, _ = block_off(x)

    torch.manual_seed(1)
    block_on = TitansMACBlock(dim=16, num_heads=2, segment_size=4,
                              num_persistent_tokens=3, memory_hidden_dim=12,
                              decouple_memory_update=True)
    out_on, state, _ = block_on(x)

    assert torch.isfinite(out_on).all()
    assert not torch.allclose(out_on, out_off), "should differ from the default path"

    out_on.pow(2).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in block_on.parameters())


# ------------------------------------------------- learned attention temperature
def test_learned_qk_scale_off_by_default_adds_no_parameters():
    """Default off must stay bit-identical to the pre-existing block."""
    torch.manual_seed(3)
    off = TitansMACBlock(dim=16, num_heads=2, segment_size=4)
    torch.manual_seed(3)
    on = TitansMACBlock(dim=16, num_heads=2, segment_size=4, learned_qk_scale=True)

    assert not hasattr(off.attn, "log_qk_scale")
    assert on.attn.log_qk_scale.shape == (2,)  # one temperature per head
    assert sum(p.numel() for p in on.parameters()) == \
        sum(p.numel() for p in off.parameters()) + 2


def test_learned_qk_scale_sharpens_attention_and_decorrelates_output():
    """The point of the flag: with cosine Q/K, SDPA's default 1/sqrt(head_dim)
    leaves attention near-uniform, so its output tokens are near-collinear (which
    is what makes the memory's per-token gradients collinear). A learned
    temperature initialised >1 must measurably decorrelate that output."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32)

    def meancos(z):
        zn = torch.nn.functional.normalize(z, dim=-1)
        c = zn @ zn.T
        return c[~torch.eye(c.shape[0], dtype=bool)].mean().item()

    torch.manual_seed(4)
    off = TitansMACBlock(dim=32, num_heads=2, segment_size=8)
    torch.manual_seed(4)
    on = TitansMACBlock(dim=32, num_heads=2, segment_size=8, learned_qk_scale=True)

    with torch.no_grad():
        cos_off = meancos(off.attn(x)[0])
        cos_on = meancos(on.attn(x)[0])

    # Bare attention over 8 tokens; inside a real block the context is
    # [persistent; h_i; segment] (a longer causal prefix), where the same
    # measurement is worse still (~0.90).
    assert cos_off > 0.55, f"expected the default path to be a smoother, got {cos_off}"
    assert cos_on < cos_off - 0.2, f"temperature did not decorrelate: {cos_off} -> {cos_on}"


def test_learned_qk_scale_is_trained_and_clamped():
    """The temperature receives gradient, and the clamp caps the effective scale."""
    block = TitansMACBlock(dim=16, num_heads=2, segment_size=4,
                           learned_qk_scale=True, qk_scale_init=10.0, qk_scale_max=20.0)
    out, _, _ = block(torch.randn(2, 8, 16))
    out.pow(2).mean().backward()
    g = block.attn.log_qk_scale.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0

    with torch.no_grad():
        block.attn.log_qk_scale.fill_(10.0)  # far above the clamp
        eff = block.attn.log_qk_scale.clamp(max=block.attn.log_qk_scale_max).exp()
    assert torch.allclose(eff, torch.full_like(eff, 20.0))


def test_learned_qk_scale_rejects_bad_init():
    with pytest.raises(ValueError):
        TitansMACBlock(dim=16, segment_size=4, learned_qk_scale=True, qk_scale_init=0.0)
