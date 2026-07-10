"""Tests for the metaplastic K-reservoir momentum cascade (standalone module).

Covers: the user's exact hand-worked K=2 regression, eigenvalue/stability of the
fixed geometric B, mass conservation, the step/scan public seam, the selective
(data-dependent) mode producing non-commuting B_t, and the impulse-response
power-law diagnostic (K=6) whose numbers are reproduced in the README/report.
"""

import numpy as np
import torch

from titans.metaplastic import (
    MetaplasticCascade,
    build_tridiagonal_B,
    cascade_scan,
    cascade_step,
    geometric_betas,
    read,
    spectral_radius,
)


def test_hand_worked_k2_regression():
    """Exact reproduction of the user's hand derivation: K=2, single beta=0.3,
    unit gradient at t=1 then zero. u^(1): 1, 0.70, 0.58, 0.532, 0.5128 -> ~0.5.

    Also contrasts with a plain exponential momentum (theta=0.7) that keeps
    decaying (1, 0.7, 0.49, 0.343, 0.24) -- the cascade's whole point.
    """
    betas = torch.tensor([0.3], dtype=torch.float64)  # single edge coupling b_2
    u = torch.zeros(2, dtype=torch.float64)
    got = []
    for t in range(1, 6):
        grad = torch.tensor(1.0 if t == 1 else 0.0, dtype=torch.float64)
        u = cascade_step(u, grad, betas)
        got.append(read(u).item())

    expected = [1.0, 0.70, 0.58, 0.532, 0.5128]
    assert np.allclose(got, expected, atol=1e-6), f"got {got}, expected {expected}"

    # Plain exponential momentum keeps decaying, well below the cascade tail.
    exp = [0.7 ** k for k in range(5)]  # 1, 0.7, 0.49, 0.343, 0.24
    assert got[-1] > 2 * exp[-1]


def test_scan_matches_step_loop():
    """cascade_scan is exactly the sequential cascade_step loop (the API seam)."""
    torch.manual_seed(0)
    K = 4
    betas = geometric_betas(K, beta1=0.5, g=2.0)
    u0 = torch.zeros(3, 5, K, dtype=torch.float64)  # [d_out, d_in, K]
    grads = torch.randn(7, 3, 5, dtype=torch.float64)  # [T, d_out, d_in]

    us = cascade_scan(u0, grads, betas)
    assert us.shape == (7, 3, 5, K)

    u = u0
    for t in range(7):
        u = cascade_step(u, grads[t], betas)
        assert torch.allclose(u, us[t])


def test_B_is_doubly_stochastic_and_stable():
    """B (fixed geometric) is symmetric, row-stochastic, eigenvalues in (-1, 1],
    spectral radius exactly 1 -- the stability finding in the module docstring."""
    K = 6
    betas = geometric_betas(K, beta1=0.5, g=2.0)
    B = build_tridiagonal_B(betas)

    assert torch.allclose(B, B.T)  # symmetric
    assert torch.allclose(B.sum(dim=1), torch.ones(K, dtype=B.dtype))  # row sums = 1
    assert (B >= 0).all()  # non-negative -> doubly stochastic

    eig = torch.linalg.eigvalsh(B)  # symmetric -> real
    assert eig.max().item() <= 1 + 1e-9
    assert eig.min().item() > -1 + 1e-6  # comfortably away from -1
    assert abs(spectral_radius(betas).item() - 1.0) < 1e-9


def test_mass_conservation_and_equilibrium():
    """A closed cascade conserves total mass; a unit impulse relaxes u^(1) to 1/K
    (not 0). Confirms the marginally-stable, mass-conserving character of B."""
    K = 5
    cas = MetaplasticCascade(K, beta1=0.5, g=2.0)
    u = cas.init_state(torch.zeros(()))  # scalar entry -> shape [K]
    u = cascade_step(u, torch.tensor(1.0), cas.betas())  # unit impulse at t=0
    total0 = u.sum().item()
    for _ in range(2000):
        u = cascade_step(u, torch.tensor(0.0), cas.betas())
    assert abs(u.sum().item() - total0) < 1e-5  # mass conserved (no leak; fp32)
    assert abs(read(u).item() - 1.0 / K) < 1e-4  # equilibrium redistribution


def test_selective_B_varies_and_is_noncommuting():
    """Selective mode: B_t depends on the token, and two different B_t generally
    do NOT commute -- the property that makes it more than a fixed filter bank."""
    torch.manual_seed(1)
    dim, K = 8, 4
    cas = MetaplasticCascade(K, beta1=0.5, g=2.0, dim=dim, selective=True)

    x_a = torch.randn(dim)
    x_b = torch.randn(dim)
    B_a = build_tridiagonal_B(cas.selective_betas(x_a).detach())
    B_b = build_tridiagonal_B(cas.selective_betas(x_b).detach())
    assert not torch.allclose(B_a, B_b, atol=1e-4)  # data-dependent
    comm = B_a @ B_b - B_b @ B_a
    assert comm.abs().max().item() > 1e-6  # non-commuting

    # Selective gate never exceeds the geometric base (stability preserved).
    assert (cas.selective_betas(x_a) <= cas.base_betas + 1e-9).all()


def test_selective_scan_shapes_and_grad():
    """Selective scan runs over [T, ...] and stays differentiable to the head."""
    torch.manual_seed(2)
    dim, K, T = 6, 3, 5
    cas = MetaplasticCascade(K, beta1=0.5, g=2.0, dim=dim, selective=True)
    u0 = torch.zeros(2, K)  # [entries, K]
    grads = torch.randn(T, 2)
    x = torch.randn(T, 2, dim)
    us = cas.scan(u0, grads, x)
    assert us.shape == (T, 2, K)
    read(us).sum().backward()
    assert cas.beta_head.weight.grad is not None


def test_impulse_response_power_law_diagnostic(capsys):
    """K=6 impulse response vs a matched single-exponential momentum.

    Baseline lambda = 1 - b_2 matches the cascade's *initial* one-step decay.
    Reports retention ratios at t=10/100/500 and a log-log fit of the cascade's
    excess-above-equilibrium (power-law-like) vs the exponential. Numbers are
    echoed (pytest -s) and mirrored in the README.
    """
    K = 6
    betas = geometric_betas(K, beta1=0.5, g=2.0)
    b2 = betas[0].item()
    lam = 1.0 - b2  # matched initial decay
    plateau = 1.0 / K

    u = torch.zeros(K, dtype=torch.float64)
    u[0] = 1.0  # unit impulse at t=0
    cas = [u[0].item()]
    for t in range(1, 601):
        u = cascade_step(u, torch.tensor(0.0, dtype=torch.float64), betas)
        cas.append(u[0].item())
    cas = np.array(cas)
    exp = lam ** np.arange(0, 601)

    ratios = {t: cas[t] / exp[t] for t in (10, 100, 500)}
    # Cascade retains dramatically more than the matched exponential.
    assert ratios[10] > 3.0
    assert ratios[100] > 1e6
    assert ratios[500] > 1e30

    # Cascade relaxes toward 1/K, never to 0; exponential vanishes.
    assert abs(cas[500] - plateau) < 1e-2
    assert exp[500] < 1e-30

    # Power-law check over the intermediate window: excess above equilibrium is
    # far straighter on a log-log axis than the exponential is.
    def loglog_r2(y, lo, hi):
        t = np.arange(lo, hi)
        yy = y[lo:hi]
        m = yy > 0
        lt, ly = np.log(t[m]), np.log(yy[m])
        a = np.polyfit(lt, ly, 1)
        pred = np.polyval(a, lt)
        r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        return a[0], r2

    slope_c, r2_cas = loglog_r2(cas - plateau, 2, 30)
    _, r2_exp = loglog_r2(exp, 2, 30)
    assert r2_cas > 0.98  # power-law-like over the intermediate window
    assert r2_cas > r2_exp  # straighter on log-log than the exponential

    print(
        "\n[impulse-response diagnostic] K=6 beta1=0.5 g=2 (b_2=%.3f, lambda=%.3f)"
        % (b2, lam)
    )
    for t in (10, 100, 500):
        print(
            "  t=%3d  cascade u1=%.5f  exp=%.3e  ratio=%.3e  excess=%.5f"
            % (t, cas[t], exp[t], ratios[t], cas[t] - plateau)
        )
    print(
        "  loglog[2,30): cascade excess slope=%.3f R2=%.4f | exp R2=%.4f"
        % (slope_c, r2_cas, r2_exp)
    )
