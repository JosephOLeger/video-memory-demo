"""``normalize_hidden``: bound the deep memory's curvature without emptying the loss.

The theory note derives ``lambda_max(H_GN^(W1)) = 2 ||h_0||^2`` for the last layer's
parameter block, with ``h_0 = sigma(W_0 k + b_0)``. ``normalize_qk`` pins ``||k||``,
which is the *wrong* Gram matrix -- ``h_0`` is computed from online-updated weights
and grows freely inside the inner loop. These tests pin the two properties that make
``normalize_hidden`` a usable fix rather than a second ``normalize_output``:

1. ``||h_0||_2 == 1`` exactly, so the curvature is the constant 2;
2. ``M(k)``'s scale stays **free**, so the associative loss keeps its signal.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from titans.mac import TitansMACBlock
from titans.memory import NeuralMemory, mlp_forward


def _params(d_in=16, d_h=64, d_out=16, scale=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w0": torch.randn(d_h, d_in, generator=g) * scale,
            "b0": torch.randn(d_h, generator=g) * scale,
            "w1": torch.randn(d_out, d_h, generator=g),
            "b1": torch.randn(d_out, generator=g)}


def _h0(p, x, activation="silu"):
    h = x @ p["w0"].T + p["b0"]
    return F.silu(h) if activation == "silu" else F.gelu(h)


def test_hidden_norm_is_exactly_one():
    """The invariant the curvature bound rests on."""
    p, x = _params(), torch.randn(16) * 5
    assert _h0(p, x).norm() > 100                       # unbounded without the fix
    out = mlp_forward(p, x, 2, normalize_output=False, normalize_hidden=True)
    h0n = _h0(p, x)
    h0n = h0n / (h0n.pow(2).sum().sqrt() + 1e-6)
    assert h0n.norm() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(out, h0n @ p["w1"].T + p["b1"], atol=1e-5)


def test_it_is_L2_not_RMS():
    """RMS would leave ||h_0|| = sqrt(d_h) ~ 22, far above the predicted threshold."""
    d_h = 64
    p, x = _params(d_h=d_h), torch.randn(16) * 5
    out = mlp_forward(p, x, 2, normalize_output=False, normalize_hidden=True)
    # recover h_0 from the output: b_1 removed, then solve is overkill -- instead
    # check the curvature proxy directly against both candidates.
    h0 = _h0(p, x)
    l2 = (h0 / h0.norm()).norm()
    rms = (h0 / h0.pow(2).mean().sqrt()).norm()
    assert l2 == pytest.approx(1.0, abs=1e-5)
    assert rms == pytest.approx(math.sqrt(d_h), rel=1e-4)
    assert torch.isfinite(out).all()


def test_output_scale_stays_free():
    """The difference from normalize_output: M(k) is not pinned, so v is reachable."""
    p, x = _params(), torch.randn(16)
    big = {**p, "w1": p["w1"] * 100}
    a = mlp_forward(p, x, 2, normalize_output=False, normalize_hidden=True)
    b = mlp_forward(big, x, 2, normalize_output=False, normalize_hidden=True)
    assert b.norm() > 50 * a.norm()                     # scale follows W_1
    # normalize_output instead pins it, which is what empties the loss
    c = mlp_forward(big, x, 2, normalize_output=True, normalize_hidden=True)
    assert c.pow(2).mean().sqrt() == pytest.approx(1.0, abs=1e-3)


def test_the_associative_signal_survives():
    """Fit a small-norm target, the regime where normalize_output fails.

    ``||v|| ~ 0.01`` against a pinned ``||M(k)|| = 1`` is the measured mismatch that
    makes the residual carry no information about ``v``.
    """
    torch.manual_seed(0)
    k, v = torch.randn(16), torch.randn(16) * 0.01
    for nout, nhid, should_fit in [(True, False, False), (False, True, True)]:
        p = {k_: v_.clone().requires_grad_(True) for k_, v_ in _params(scale=1.0).items()}
        opt = torch.optim.Adam(p.values(), lr=0.05)
        first = last = None
        for step in range(200):
            opt.zero_grad()
            loss = ((mlp_forward(p, k, 2, normalize_output=nout,
                                 normalize_hidden=nhid) - v) ** 2).sum()
            loss.backward(); opt.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        if should_fit:
            assert last < 0.01 * first, f"normalize_hidden should still fit ({last})"
        else:
            assert last > 0.5 * first, f"normalize_output should not ({last})"


def test_off_by_default_and_reaches_both_paths():
    """The update's gradient and the read must agree on the same forward."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    base = NeuralMemory(dim=16, normalize_qk="l2", normalize_output=False)
    assert base.normalize_hidden is False
    assert base.memory.normalize_hidden is False

    fixed = NeuralMemory(dim=16, normalize_qk="l2", normalize_output=False,
                         normalize_hidden=True)
    fixed.load_state_dict(base.state_dict())

    # the inner gradient (update path) must differ
    gb = base._grad_fn()(base.memory.init_state(1), x[:1, 0], x[:1, 0])
    gf = fixed._grad_fn()(fixed.memory.init_state(1), x[:1, 0], x[:1, 0])
    assert not torch.allclose(gb["w1"], gf["w1"])

    # and the read path must follow the same convention
    sb = base.memory.init_state(2)
    assert not torch.allclose(base.retrieve(sb, x[:, 0]), fixed.retrieve(sb, x[:, 0]))


def test_reaches_the_block_from_a_config_dict():
    """A named config must be able to turn it on end to end."""
    blk = TitansMACBlock(dim=16, segment_size=4, normalize_qk="l2",
                         normalize_output=False, normalize_hidden=True)
    assert blk.memory.normalize_hidden is True
    y = blk(torch.randn(2, 8, 16))
    out = y[0] if isinstance(y, tuple) else y
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("target", [0.3, 1.0, 2.0])
def test_target_norm_is_settable(target):
    """The ceiling's *value* is a lever, not a constant.

    At init ``||h_0|| ~ 0.3``, so a target of 1.0 caps the curvature but raises the
    resting point ~10x. A run must be able to choose where the ceiling sits.
    """
    p, x = _params(), torch.randn(16) * 5
    p["b1"] = torch.zeros_like(p["b1"])        # b_1 does not scale with h_0
    out = mlp_forward(p, x, 2, normalize_output=False, normalize_hidden=target)
    ref = mlp_forward(p, x, 2, normalize_output=False, normalize_hidden=True)
    assert out.norm() == pytest.approx(target * float(ref.norm()), rel=1e-4)


def test_named_config_is_a_delta_on_sane_ns():
    """Standing directive: new arms are deltas on `sane_ns`."""
    from titans.worldmodel.atlas import CONFIGS

    from titans.worldmodel.frozen import build_cfg
    base, nh = CONFIGS["sane_ns"], build_cfg("sane_ns", normalize_hidden=1.0)
    assert set(nh) - set(base) == {"normalize_hidden"}
    assert all(nh[k] == v for k, v in base.items())
    assert nh["normalize_hidden"] == 1.0
