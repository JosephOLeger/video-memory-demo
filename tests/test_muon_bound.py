"""The magnitude conventions of Newton-Schulz, and the bound they do or do not give.

The point of these tests is one number: how far can a single inner-loop step move
``M``? Under the default (``rescale``) the answer is "as far as the gradient says",
which is why the memory can run away. Under ``unit`` -- ATLAS Eq 32 taken literally --
the answer is ``theta_t``, whatever the gradient says.
"""

from __future__ import annotations

import pytest
import torch

from titans.memory import NeuralMemory, newton_schulz5


def _G(scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 12, 16, generator=g) * scale


def test_rescale_is_the_old_behaviour_exactly():
    G = _G()
    assert torch.equal(newton_schulz5(G), newton_schulz5(G, rescale=True, clip=None))
    assert torch.allclose(
        newton_schulz5(G).norm(dim=(-2, -1)), G.norm(dim=(-2, -1)), rtol=1e-5
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_unit_bounds_the_spectral_norm_regardless_of_the_gradient(scale):
    """The property the whole change exists for: the output's size does not depend
    on the input's. A gradient 10^6 times bigger must not give a bigger step."""
    out = newton_schulz5(_G(scale), rescale=False)
    sv = torch.linalg.svdvals(out)
    assert sv.max() < 1.3, sv.max()
    # ...and it is not merely small: the map is near-semi-orthogonal, not a shrink.
    assert sv.min() > 0.5, sv.min()


def test_clip_caps_the_tail_and_leaves_the_body_untouched():
    small, big = _G(0.01, seed=1), _G(100.0, seed=1)
    # Below the cap, clipping is a no-op: theta's learned scale still reaches through.
    assert torch.allclose(newton_schulz5(small, clip=1.0), newton_schulz5(small))
    # Above it, the magnitude is exactly the cap and the direction is preserved.
    capped = newton_schulz5(big, clip=1.0)
    assert torch.allclose(capped.norm(dim=(-2, -1)), torch.ones(2), rtol=1e-4)
    ref = newton_schulz5(big)
    cos = torch.nn.functional.cosine_similarity(
        capped.flatten(1), ref.flatten(1), dim=-1
    )
    assert (cos > 0.999).all()


def test_bias_still_passes_through_in_every_mode():
    b = torch.randn(3, 8)
    for kw in ({}, dict(rescale=False), dict(clip=1.0)):
        assert torch.equal(newton_schulz5(b, **kw), b)


def test_mode_aliases_and_validation():
    assert NeuralMemory(dim=8, orthogonalize_momentum=True)._orth_mode == "rescale"
    # The public attribute keeps what the caller passed, so frozen configs that
    # assert `is True` still hold.
    assert NeuralMemory(dim=8, orthogonalize_momentum=True).orthogonalize_momentum is True
    assert NeuralMemory(dim=8)._orth_mode is False
    with pytest.raises(ValueError, match="orthogonalize_momentum"):
        NeuralMemory(dim=8, orthogonalize_momentum="muon")


def _step_move(mode: str, grad_scale: float, **kw) -> float:
    """Largest singular value of ``M_t - (1-alpha) M_{t-1}`` for one step."""
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03,
                       orthogonalize_momentum=mode, update_bias=False, **kw)
    state, S = mem.init_memory(1)
    g = {k: (torch.randn_like(v) * grad_scale if v.dim() >= 3 else torch.zeros_like(v))
         for k, v in state.items()}
    one = torch.ones(1)
    new_state, _ = mem._step(state, S, g, alpha_t=one * 0.002, eta_t=one * 0.9,
                             theta_t=one * 0.03)
    moved = max(
        float(torch.linalg.svdvals((new_state[k] - 0.998 * state[k]).detach()).max())
        for k, v in state.items() if v.dim() >= 3
    )
    return moved


def test_unit_step_is_bounded_by_theta_and_rescale_is_not():
    """End to end through ``_step``: the failure mode, and the fix, in one assert."""
    small = _step_move("unit", 1e-2)
    huge = _step_move("unit", 1e4)
    # theta_t = 0.03; NS-5's singular values are ~1, so the move is ~theta either way.
    assert small == pytest.approx(huge, rel=0.15)
    assert huge < 0.05, huge
    # The default mode carries the gradient straight through into the step.
    assert _step_move(True, 1e4) > 1e3


def test_unit_runs_end_to_end_with_finite_gradients():
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03,
                       orthogonalize_momentum="unit", update_bias=False)
    x = torch.randn(2, 12, 8, requires_grad=True)
    state, S = mem.init_memory(2)
    state, S = mem.update(x, state, S, 4)
    y = mem.retrieve(state, mem.q_proj(x))
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in mem.parameters() if p.requires_grad)


def test_unit_works_on_the_cascade_path_too():
    """The cascade orthogonalises the readout, so it needs its own theta placement."""
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03,
                       orthogonalize_momentum="unit", update_bias=False,
                       use_cascade_momentum=True, cascade_reservoirs=4)
    x = torch.randn(2, 12, 8)
    state, S = mem.init_memory(2)
    state, S = mem.update(x, state, S, 4)
    y = mem.retrieve(state, mem.q_proj(x))
    assert torch.isfinite(y).all()
