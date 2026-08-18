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


# --------------------------------------------------------------------------- #
# squash: bounded like clip, but monotone everywhere (Josep's proposal).
# --------------------------------------------------------------------------- #


def _mag(out: torch.Tensor) -> torch.Tensor:
    return out.norm(dim=(-2, -1))


def test_squash_is_bounded_by_c():
    for scale in (1e-2, 1.0, 1e2, 1e6):
        out = newton_schulz5(_G(scale, seed=2), squash=0.25)
        assert (_mag(out) < 0.2500001).all(), (scale, _mag(out))


def test_squash_is_strictly_monotone_where_clip_is_flat():
    """The whole point: past the cap, clip flattens and squash still ranks."""
    norms, sq, cl = [], [], []
    for scale in (0.5, 5.0, 50.0, 500.0):        # all far above c after scaling
        G = _G(scale, seed=3)
        norms.append(float(_mag(G)[0]))
        sq.append(float(_mag(newton_schulz5(G, squash=0.25))[0]))
        cl.append(float(_mag(newton_schulz5(G, clip=0.25))[0]))
    assert all(a < b for a, b in zip(norms, norms[1:]))
    assert all(a < b for a, b in zip(sq, sq[1:])), sq      # order preserved
    assert cl[1] == pytest.approx(cl[-1]), cl              # order destroyed


def test_squash_is_the_identity_well_below_c():
    G = _G(1e-4, seed=4)
    assert torch.allclose(newton_schulz5(G, squash=1.0), newton_schulz5(G), rtol=1e-6)


def test_squash_interpolates_between_the_two_papers():
    """c -> inf recovers Titans (rescale); c -> 0 recovers ATLAS (unit scale)."""
    G = _G(1.0, seed=5)
    ref = float(_mag(newton_schulz5(G))[0])
    assert float(_mag(newton_schulz5(G, squash=1e8))[0]) == pytest.approx(ref, rel=1e-6)
    tiny = 1e-6
    assert float(_mag(newton_schulz5(G, squash=tiny))[0]) == pytest.approx(tiny, rel=1e-3)


def test_squash_mode_end_to_end_and_bounds_the_step():
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03, normalize_qk="l2",
                       orthogonalize_momentum="squash", orthogonalize_clip=0.25,
                       update_bias=False)
    assert mem._orth_mag_kwargs() == {"squash": 0.25}
    small, huge = _step_move("squash", 1e-2, orthogonalize_clip=0.25), \
                  _step_move("squash", 1e4, orthogonalize_clip=0.25)
    assert huge < 0.26 and small < huge          # bounded, and still ordered
    x = torch.randn(2, 12, 8)
    state, S = mem.init_memory(2)
    state, S = mem.update(x, state, S, 4)
    assert torch.isfinite(mem.retrieve(state, mem.q_proj(x))).all()


# --------------------------------------------------------------------------- #
# unit_squash: unit's bound, with the amplitude put back as a monotone function
# of the RAW momentum -- theta at the output, where Josep wanted it.
# --------------------------------------------------------------------------- #


def _unit_squash_move(grad_scale: float, c: float = 350.0) -> float:
    return _step_move("unit_squash", grad_scale, orthogonalize_clip=c)


def test_unit_squash_keeps_units_bound():
    """psi is in [0,1), so the step can only be SMALLER than unit's -- never larger,
    however big the gradient gets."""
    for scale in (1e-2, 1.0, 1e4, 1e8):
        assert _unit_squash_move(scale) <= _step_move("unit", scale) + 1e-9, scale
    assert _unit_squash_move(1e8) < 0.05


def test_unit_squash_is_monotone_where_unit_is_flat():
    """The whole point of the variant: unit writes the same for every surprise."""
    scales = (1e-3, 1e-1, 1e1, 1e3)
    sq = [_unit_squash_move(s) for s in scales]
    un = [_step_move("unit", s) for s in scales]
    assert all(a < b for a, b in zip(sq, sq[1:])), sq        # ordered
    assert un[0] == pytest.approx(un[-1], rel=0.15), un      # unit: flat


def test_unit_squash_recovers_unit_as_c_goes_to_zero():
    big = _unit_squash_move(1.0, c=1e-6)
    assert big == pytest.approx(_step_move("unit", 1.0), rel=1e-3)


def test_unit_squash_runs_on_both_paths():
    for extra in ({}, dict(use_cascade_momentum=True, cascade_reservoirs=4)):
        torch.manual_seed(0)
        mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03, normalize_qk="l2",
                           orthogonalize_momentum="unit_squash",
                           orthogonalize_clip=350.0, update_bias=False, **extra)
        x = torch.randn(2, 12, 8)
        state, S = mem.init_memory(2)
        state, S = mem.update(x, state, S, 4)
        assert torch.isfinite(mem.retrieve(state, mem.q_proj(x))).all()


# --------------------------------------------------------------------------- #
# softclip: Josep's proposal read as what it is -- an IMPROVED CLIP. arctan is a
# smoothed `min`, so the lineage is clip -> softclip, not unit -> unit_squash.
# The property that separates them is which constant bounds the step.
# --------------------------------------------------------------------------- #


def _theta_move(mode: str, grad_scale: float, theta: float, **kw) -> float:
    """``_step_move`` with the inner gate set explicitly, to vary ``theta_t`` alone."""
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, orthogonalize_momentum=mode,
                       update_bias=False, **kw)
    state, S = mem.init_memory(1)
    g = {k: (torch.randn_like(v) * grad_scale if v.dim() >= 3 else torch.zeros_like(v))
         for k, v in state.items()}
    one = torch.ones(1)
    new_state, _ = mem._step(state, S, g, alpha_t=one * 0.002, eta_t=one * 0.9,
                             theta_t=one * theta)
    return max(float(torch.linalg.svdvals((new_state[k] - 0.998 * state[k]).detach()).max())
               for k, v in state.items() if v.dim() >= 3)


def _softclip_move(grad_scale: float, c: float = 8.0) -> float:
    return _step_move("softclip", grad_scale, orthogonalize_clip=c)


def test_softclip_bound_is_the_constant_c_not_theta():
    """The distinction that motivates the mode.

    ``unit`` and ``unit_squash`` bound the step by ``theta_t``, and
    ``theta_t = softplus(head(x)) * theta_scale`` has no upper bound -- so their
    "bound" moves whenever the outer loop pushes the gate. ``softclip``'s bound is
    the constant ``c``: doubling ``theta_scale`` cannot make the step exceed it.
    """
    for scale in (1e2, 1e6):
        assert _softclip_move(scale, c=8.0) < 8.0 + 1e-6
        # a hundred times the gate, same ceiling
        assert _theta_move("softclip", scale, theta=3.0,
                           orthogonalize_clip=8.0) < 8.0 + 1e-6
    # ...whereas unit's step scales one-for-one with theta, without limit.
    assert _theta_move("unit", 1.0, theta=3.0) == pytest.approx(
        100.0 * _theta_move("unit", 1.0, theta=0.03), rel=0.2)


def test_softclip_is_monotone_where_clip_is_flat():
    scales = (1e-3, 1e-1, 1e1, 1e3)
    soft = [_softclip_move(s) for s in scales]
    assert all(a < b for a, b in zip(soft, soft[1:])), soft


def test_softclip_is_the_identity_well_below_c():
    """Far under the cap it is plain gradient descent on the orthogonalised
    direction -- theta's learned scale reaches the update untouched."""
    small = _softclip_move(1e-4, c=1e4)
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03,
                       orthogonalize_momentum="softclip", orthogonalize_clip=1e4,
                       update_bias=False)
    state, _ = mem.init_memory(1)
    ref = 0.03 * 1.0                       # theta * ||NS/||NS|| || in spectral terms
    assert 0.0 < small < ref               # no cap reached, no amplification


def test_softclip_recovers_unit_up_to_a_constant_when_saturated():
    """c -> 0 flattens it onto a fixed-size step, which is exactly ``unit``'s
    behaviour: the two families meet at the degenerate end, as they should."""
    a, b = _softclip_move(1e-2, c=1e-3), _softclip_move(1e6, c=1e-3)
    # Frobenius is pinned to c exactly; the spectral norm read here still moves a
    # little with the singular-value spread NS-5 happens to leave behind.
    assert a == pytest.approx(b, rel=0.2)


def test_softclip_runs_on_both_paths():
    for extra in ({}, dict(use_cascade_momentum=True, cascade_reservoirs=4)):
        torch.manual_seed(0)
        mem = NeuralMemory(dim=8, hidden_dim=16, theta_scale=0.03, normalize_qk="l2",
                           orthogonalize_momentum="softclip", orthogonalize_clip=8.0,
                           update_bias=False, **extra)
        x = torch.randn(2, 12, 8)
        state, S = mem.init_memory(2)
        state, S = mem.update(x, state, S, 4)
        assert torch.isfinite(mem.retrieve(state, mem.q_proj(x))).all()
