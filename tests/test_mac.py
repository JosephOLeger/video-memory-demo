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
