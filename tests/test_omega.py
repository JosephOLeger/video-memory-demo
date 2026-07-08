"""Tests for the ATLAS Omega rule (sliding-window memory update, c > 1)."""

import torch
from torch.func import grad

from titans.memory import NeuralMemory, mlp_forward
from titans.worldmodel.model import WorldModel


def test_c1_has_no_gamma_head():
    """c=1 must stay the plain Titans memory: no extra params, no gamma head."""
    m1 = NeuralMemory(dim=8, num_layers=2, window_size=1)
    m3 = NeuralMemory(dim=8, num_layers=2, window_size=3)
    assert not hasattr(m1, "gamma_head")
    assert hasattr(m3, "gamma_head")


def test_window_changes_memory_and_stays_finite():
    """A c>1 update runs, stays finite, and differs from the c=1 update."""
    torch.manual_seed(0)
    x = torch.randn(2, 12, 8)

    def run(c):
        torch.manual_seed(0)
        mem = NeuralMemory(dim=8, num_layers=2, window_size=c)
        state, S = mem.init_memory(x.shape[0])
        return mem.update(x, state, S, chunk_size=12)

    s1, _ = run(1)
    s3, _ = run(3)
    for key in s1:
        assert torch.isfinite(s3[key]).all()
    assert any(not torch.allclose(s1[k], s3[k]) for k in s1), \
        "window c=3 produced the same memory as c=1"


def test_omega_gradient_equals_windowed_loss_gradient():
    """Core Omega identity: the gated sliding-window *sum of per-token gradients*
    equals the gradient of the gated windowed *loss* (grad of a sum = sum of
    grads). This is what makes Omega free given the per-token grads we compute.
    """
    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, num_layers=2, window_size=3)
    x = torch.randn(1, 5, 8)
    k, v, *_ = mem._project(x)
    gamma = torch.sigmoid(mem.gamma_head(x)).squeeze(-1)  # [1, 5]
    params = {key: w[0] for key, w in mem.memory.init_state(1).items()}
    window = [2, 3, 4]  # window of size 3 ending at t=4

    def windowed_loss(p):
        tot = x.new_zeros(())
        for i in window:
            pred = mlp_forward(p, k[0, i], mem.num_layers)
            tot = tot + gamma[0, i] * ((pred - v[0, i]) ** 2).sum()
        return tot

    g_of_sum = grad(windowed_loss)(params)

    def token_loss(p, ki, vi):
        pred = mlp_forward(p, ki, mem.num_layers)
        return ((pred - vi) ** 2).sum()

    sum_of_g = {key: torch.zeros_like(w) for key, w in params.items()}
    for i in window:
        gi = grad(token_loss)(params, k[0, i], v[0, i])
        for key in sum_of_g:
            sum_of_g[key] = sum_of_g[key] + gamma[0, i] * gi[key]

    for key in g_of_sum:
        assert torch.allclose(g_of_sum[key], sum_of_g[key], atol=1e-6)


def test_worldmodel_with_omega_window_trains():
    """WorldModel with an Omega window runs a forward+backward and stays finite."""
    torch.manual_seed(0)
    z = torch.randn(2, 32, 16)
    model = WorldModel(latent_dim=16, dim=16, depth=2, segment_size=8,
                       memory_window_size=4)
    out = model(z)
    assert torch.isfinite(out).all()
    loss = (out[:, :-1] - z[:, 1:]).abs().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
