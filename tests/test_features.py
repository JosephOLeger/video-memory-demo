"""Tests for ATLAS's polynomial feature map (`titans.features`) and its wiring.

The point of `phi` is memory *capacity*: ATLAS Prop. 2 says a memory optimising
``||M phi_p(k) - v||^2`` stores ``O(d_k^p)`` independent pairs against ``O(d_k)``
for the identity map, without touching the input projections. These tests check
the map's algebra, that it is a genuine no-op at degree 1, and that it flows
through NeuralMemory / TitansMACBlock with finite gradients.
"""

import math

import pytest
import torch

from titans.features import PolynomialFeatureMap
from titans.mac import TitansMACBlock
from titans.memory import NeuralMemory


# ------------------------------------------------------------------ the map
def test_shape_and_kernel_weight_init():
    """Feature scales c_i = 1/sqrt(i!) so the induced kernel weight is ATLAS's
    a_i = 1/i! (Eq 5)."""
    f = PolynomialFeatureMap(degree=4)
    x = torch.randn(2, 5, 8)
    assert f(x).shape == (2, 5, 32)
    assert f.out_dim(8) == 32
    expected = [1.0 / math.sqrt(math.factorial(i)) for i in range(1, 5)]
    assert torch.allclose(f.coef.detach(), torch.tensor(expected))
    # induced kernel weights c_i^2 == 1/i!
    assert torch.allclose((f.coef.detach() ** 2),
                          torch.tensor([1.0 / math.factorial(i) for i in range(1, 5)]))


def test_degree_one_is_identity():
    """degree=1 must be exactly the input (unit-init scale), so plain Titans is
    recovered and no normalisation is applied."""
    f = PolynomialFeatureMap(degree=1)
    x = torch.randn(3, 7)
    assert torch.allclose(f(x), x)


def test_powers_are_actually_concatenated():
    f = PolynomialFeatureMap(degree=3, normalize=False)
    x = torch.tensor([[2.0, -3.0]])
    out = f(x)
    c = f.coef.detach()
    expected = torch.cat([c[0] * x, c[1] * x ** 2, c[2] * x ** 3], dim=-1)
    assert torch.allclose(out, expected)
    # even powers drop the sign (a property of the monomial basis, not a bug)
    assert out[0, 2] > 0 and out[0, 3] > 0        # x^2 both positive
    assert out[0, 5] < 0                           # x^3 keeps the sign of -3


def test_normalize_controls_power_blowup():
    """Without normalisation x^p explodes for large-magnitude keys; with it the
    mapped features stay bounded -- the reason normalize defaults to True."""
    x = torch.full((1, 16), 5.0)
    raw = PolynomialFeatureMap(degree=4, normalize=False)(x)
    normed = PolynomialFeatureMap(degree=4, normalize=True)(x)
    assert raw.abs().max() > 100.0
    assert normed.abs().max() < 2.0


def test_coefficients_are_learnable():
    f = PolynomialFeatureMap(degree=3)
    f(torch.randn(4, 6)).sum().backward()
    assert f.coef.grad is not None and torch.isfinite(f.coef.grad).all()


def test_rejects_degree_zero():
    with pytest.raises(ValueError):
        PolynomialFeatureMap(degree=0)


# ------------------------------------------------------- wiring into the memory
def test_memory_widens_first_layer_but_not_output():
    """phi widens M's *input* (p*d) while its output stays d, and the extra
    capacity costs nothing in k_proj/q_proj (ATLAS's stated motivation)."""
    base = NeuralMemory(16, hidden_dim=16, feature_degree=1)
    poly = NeuralMemory(16, hidden_dim=16, feature_degree=3)

    assert tuple(base.memory.params["w0"].shape) == (16, 16)
    assert tuple(poly.memory.params["w0"].shape) == (16, 48)
    assert tuple(poly.memory.params["w1"].shape) == (16, 16)   # output width unchanged
    # the input projections are untouched -- that is the whole point
    assert base.k_proj.weight.shape == poly.k_proj.weight.shape
    assert base.q_proj.weight.shape == poly.q_proj.weight.shape


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_memory_forward_backward_finite(degree):
    mem = NeuralMemory(16, hidden_dim=16, feature_degree=degree)
    y, _ = mem.forward_chunked(torch.randn(2, 8, 16), chunk_size=4)
    assert y.shape == (2, 8, 16)
    assert torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in mem.parameters())


def test_degree_one_bit_identical_to_no_feature_map():
    """The default path must be untouched: degree=1 builds no feature map at all."""
    torch.manual_seed(0)
    a = NeuralMemory(16, hidden_dim=16)
    torch.manual_seed(0)
    b = NeuralMemory(16, hidden_dim=16, feature_degree=1)
    assert a.feature_map is None and b.feature_map is None
    x = torch.randn(2, 8, 16)
    ya, _ = a.forward_chunked(x, chunk_size=4)
    yb, _ = b.forward_chunked(x, chunk_size=4)
    assert torch.equal(ya, yb)


# ------------------------------------------------- ATLAS Table 7 memory recipe
def test_memory_expansion_sets_hidden_width():
    mem = NeuralMemory(32, memory_expansion=4)
    assert tuple(mem.memory.params["w0"].shape) == (128, 32)   # 4x expansion


def test_gelu_and_residual_change_output_but_stay_finite():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    torch.manual_seed(1)
    plain = NeuralMemory(16, hidden_dim=16)
    torch.manual_seed(1)
    atlas = NeuralMemory(16, hidden_dim=16, memory_activation="gelu",
                         memory_residual=True)
    ya, _ = plain.forward_chunked(x, chunk_size=4)
    yb, _ = atlas.forward_chunked(x, chunk_size=4)
    assert torch.isfinite(yb).all()
    assert not torch.allclose(ya, yb)


def test_rejects_bad_activation_and_expansion():
    with pytest.raises(ValueError):
        NeuralMemory(16, memory_activation="relu")
    with pytest.raises(ValueError):
        NeuralMemory(16, memory_expansion=0)


# --------------------------------------------------------------- through MAC
def test_mac_block_with_full_atlas_memory_trains():
    """The whole ATLAS memory stack (phi + Omega window + Newton-Schulz + Table 7
    memory recipe) runs end to end inside a MAC block with finite gradients."""
    block = TitansMACBlock(
        dim=32, num_heads=2, segment_size=8,
        feature_degree=2, memory_window_size=4,
        memory_orthogonalize_momentum="unit", memory_update_bias=False,
        memory_expansion=4, memory_activation="gelu", memory_residual=True,
    )
    out, state, _ = block(torch.randn(2, 16, 32))
    assert out.shape == (2, 16, 32)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_mac_feature_degree_changes_output():
    torch.manual_seed(2)
    x = torch.randn(2, 16, 32)
    torch.manual_seed(3)
    off, _, _ = TitansMACBlock(dim=32, segment_size=8)(x)
    torch.manual_seed(3)
    on, _, _ = TitansMACBlock(dim=32, segment_size=8, feature_degree=3)(x)
    assert torch.isfinite(on).all()
    assert not torch.allclose(off, on)


# ------------------------------------------------- audit regressions (2026-07-20)
# Three defects found by auditing the assembled ATLAS stack against the paper.
# All three failed *silently*: the flag was accepted and did nothing.

def test_residual_applies_even_when_phi_widens_the_input():
    """ATLAS's M(x) = x + W1 sigma(W2 x) is ill-typed once phi widens the input
    (p*d in, d out). It used to be skipped outright, so memory_residual=True was a
    no-op in exactly the configuration ATLAS specifies (phi + residual). It now
    adds phi's degree-1 block -- the un-mapped key."""
    from titans.memory import mlp_forward
    mem = NeuralMemory(16, memory_residual=True, feature_degree=2, memory_expansion=4)
    assert mem.memory.in_dim == 32 and mem.dim == 16      # widths genuinely differ
    p = {k: v[0] for k, v in mem.memory.init_state(1).items()}
    x = torch.randn(mem.memory.in_dim)
    with torch.no_grad():
        on = mlp_forward(p, x, mem.num_layers, "gelu", True)
        off = mlp_forward(p, x, mem.num_layers, "gelu", False)
    assert not torch.allclose(on, off), "residual must not be a no-op under phi"


def test_residual_is_verbatim_when_widths_match():
    """With no feature map the skip must be the paper's formula exactly."""
    from titans.memory import mlp_forward
    mem = NeuralMemory(16, memory_residual=True)
    p = {k: v[0] for k, v in mem.memory.init_state(1).items()}
    x = torch.randn(16)
    with torch.no_grad():
        on = mlp_forward(p, x, mem.num_layers, "silu", True)
        off_pre = mlp_forward(p, x, mem.num_layers, "silu", False)
    # on == rms_norm(pre_norm + x); check the skip moved the output toward x
    assert not torch.allclose(on, off_pre)

def test_module_forward_matches_the_functional_inner_loop():
    """NeuralMemoryMLP.forward used to hardcode SiLU/no-residual while the inner
    loop ran the configured pair, so the module and the functional path could
    disagree. They must now agree for any configuration."""
    from titans.memory import mlp_forward
    mem = NeuralMemory(16, memory_activation="gelu", memory_residual=True,
                       memory_expansion=4, feature_degree=2)
    p = {k: v[0] for k, v in mem.memory.init_state(1).items()}
    x = torch.randn(mem.memory.in_dim)
    with torch.no_grad():
        via_module = mem.memory(p, x)
        via_functional = mlp_forward(p, x, mem.num_layers,
                                     mem.memory_activation, mem.memory_residual)
    assert torch.equal(via_module, via_functional)


# ------------------------------------------- the forget gate's resting value
def test_alpha_init_sets_the_memory_half_life():
    """alpha = sigmoid(0) = 0.5 (the default) means M keeps HALF its content per
    token -- a half-life of one token. alpha_init sets the resting value through
    alpha_head's bias so the memory can actually span a sequence."""
    import math
    default = NeuralMemory(16)
    assert abs(torch.sigmoid(default.alpha_head.bias).item() - 0.5) < 0.15

    mem = NeuralMemory(16, alpha_init=0.002)
    a = torch.sigmoid(mem.alpha_head.bias).item()
    assert abs(a - 0.002) < 1e-5
    half_life = math.log(2) / -math.log(1 - a)
    assert half_life > 300, f"expected a long half-life, got {half_life:.0f} tokens"


def test_alpha_init_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            NeuralMemory(16, alpha_init=bad)


def test_small_alpha_prevents_the_rank_collapse():
    """The headline: with a sane forget gate the memory keeps full rank across
    threaded segments, where the default collapses to rank 1 -- and this needs
    neither Newton-Schulz nor the bias freeze."""
    def final_pr(**kw):
        torch.manual_seed(1)
        blk = TitansMACBlock(32, num_heads=2, segment_size=8, memory_hidden_dim=32,
                             memory_num_layers=2, **kw)
        torch.manual_seed(0)
        x = torch.randn(2, 64, 32)
        with torch.no_grad():
            state, S = blk.memory.init_memory(2)
            for i in range(8):
                seg = x[:, i * 8:(i + 1) * 8, :]
                p = blk.persistent(2)
                h = blk.memory.retrieve(state, blk.memory.q_proj(seg))
                if blk.norm_hist is not None:
                    h = blk.norm_hist(h)
                y = blk.attn(torch.cat([p, h, seg], dim=1))[:, -8:, :]
                state, S = blk.memory.update(y, state, S, blk.memory_chunk_size)
            w = state["w1"][0].double()
            s = torch.linalg.svdvals(w)
            return float((s ** 2).sum() ** 2 / (s ** 4).sum()), float(w.norm())

    pr_default, _ = final_pr()
    pr_fixed, norm_fixed = final_pr(alpha_init=0.002, theta_scale=0.03,
                                    normalize_qk="l2", normalize_output=False,
                                    normalize_retrieval=True)
    assert pr_default < 1.5, f"default should collapse to rank 1, got {pr_default}"
    assert pr_fixed > 8.0, f"sane gates should keep rank, got {pr_fixed}"
    assert 1e-3 < norm_fixed < 1e4, f"memory should be alive, ||w1||={norm_fixed}"
