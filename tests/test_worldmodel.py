"""Tests for the Phase-4 WorldModel: memory carry threading and ablation."""

import torch

from titans.worldmodel.model import WorldModel


def _latents(B=2, T=32, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d, generator=g)


def test_carry_threading_matches_single_forward():
    """Processing a sequence in chunks with the carry must equal one forward.

    This is the core Phase-4 property: the memory recurrence M_0 -> M_1 -> ...
    continues across calls, so truncated-BPTT-style chunking does not reset or
    diverge from processing the whole sequence at once.
    """
    torch.manual_seed(0)
    z = _latents(B=2, T=32, d=16)
    model = WorldModel(latent_dim=16, dim=16, depth=2, segment_size=8)

    full = model(z)
    p1, carry = model(z[:, :16], return_carry=True)
    p2, _ = model(z[:, 16:], carry=carry, return_carry=True)
    chunked = torch.cat([p1, p2], dim=1)

    assert torch.allclose(full, chunked, atol=1e-4), (
        f"chunked-with-carry differs from single forward: "
        f"max {(full - chunked).abs().max().item():.2e}"
    )


def test_memory_ablation_changes_output():
    """Disabling the long-term memory branch must run and change the output."""
    z = _latents(B=2, T=32, d=16)
    torch.manual_seed(0)
    full = WorldModel(latent_dim=16, dim=16, depth=2, segment_size=8, use_memory=True)
    torch.manual_seed(0)
    ablated = WorldModel(latent_dim=16, dim=16, depth=2, segment_size=8, use_memory=False)

    of, oa = full(z), ablated(z)
    assert torch.isfinite(of).all() and torch.isfinite(oa).all()
    assert not torch.allclose(of, oa), "ablation had no effect on the output"
