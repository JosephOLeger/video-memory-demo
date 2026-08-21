"""``alpha_override`` must change the computation when set and be inert when None.

The recurring failure mode of this codebase is a flag that is accepted and
silently inert (README §8). ``alpha_override`` is set post-construction on a
loaded model (it is deliberately not a constructor argument), so the tests
exercise exactly that path.
"""
import torch

from titans.memory import NeuralMemory


def _mem(seed: int = 0) -> NeuralMemory:
    torch.manual_seed(seed)
    return NeuralMemory(dim=16, hidden_dim=32, alpha_init=0.002, eta_init=0.9)


def test_none_is_inert():
    x = torch.randn(2, 12, 16)
    m1, m2 = _mem(), _mem()
    m2.alpha_override = None
    y1, _ = m1.forward_chunked(x, chunk_size=4)
    y2, _ = m2.forward_chunked(x, chunk_size=4)
    assert torch.equal(y1, y2)


def test_override_pins_the_gate():
    m = _mem()
    m.alpha_override = 0.5
    x = torch.randn(2, 12, 16)
    alpha = m._project(x)[3]
    assert torch.all(alpha == 0.5)


def test_override_changes_the_computation():
    x = torch.randn(2, 12, 16)
    m1, m2 = _mem(), _mem()
    m2.alpha_override = 0.5
    y1, _ = m1.forward_chunked(x, chunk_size=4)
    y2, _ = m2.forward_chunked(x, chunk_size=4)
    assert not torch.allclose(y1, y2)
