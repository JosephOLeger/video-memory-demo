"""Noyau du mode direct, mesure sur le vrai NeuralMemory._step (20/10)."""
import json, math, numpy as np, torch
from titans.memory import NeuralMemory
torch.set_default_dtype(torch.float64)

def ideal(K, n, beta1=0.5, c=2.0):
    C = c ** np.arange(K); gE = beta1 * c ** -np.arange(1, K + 1)   # g_{k,k+1}, last = terminal
    L = np.zeros((K, K))
    for k in range(K):
        if k > 0: L[k, k] += gE[k-1]; L[k, k-1] -= gE[k-1]
        L[k, k] += gE[k]
        if k < K - 1: L[k, k+1] -= gE[k]
    A = np.eye(K) - np.diag(1 / C) @ L
    u = np.zeros(K); u[0] = 1.0; out = []
    for _ in range(n): out.append(u.copy()); u = A @ u
    return np.array(out)                   # [n, K]

def mem(K, readout, muon=False, tau=None):
    return NeuralMemory(dim=8, use_cascade_momentum=True, cascade_reservoirs=K, cascade_mode="direct",
                        cascade_leak="none", cascade_areas=2.0, cascade_g=2.0, cascade_beta1=0.5,
                        cascade_terminal_leak=True, cascade_terminal_tau=tau, cascade_readout=readout,
                        orthogonalize_momentum=muon)

def run_scalar(m, K, n):
    w0 = torch.zeros(1, 1, 1); w = w0.clone(); U = torch.zeros(1, 1, 1, K)
    a = torch.zeros(1); e = torch.zeros(1); th = torch.ones(1); out = []
    for t in range(n):
        g = torch.full((1, 1, 1), -1.0 if t == 0 else 0.0)
        st, S = m._step({"w": w}, {"w": U}, {"w": g}, a, e, th)
        w, U = st["w"], S["w"]; out.append(float(w.reshape(()) - w0.reshape(())))
    return np.array(out)

def slope(phi, lo, hi):
    return (math.log(phi[hi]) - math.log(phi[lo])) / (math.log(hi) - math.log(lo))

res = {}
N = 40000
inf = ideal(16, N)[:, 0]
for K in (5, 6, 7):
    m = mem(K, "shallow")
    phi = run_scalar(m, K, N)
    ref = ideal(K, N)[:, 0]
    tau = 4 ** K
    dev = np.abs(phi / inf - 1); n10 = int(np.argmax(dev[10:] > 0.1)) + 10
    res[f"K{K}_shallow"] = {"max_rel_err_vs_closed_form": float(np.max(np.abs(phi[:20000] - ref[:20000]) / ref[:20000])),
        "slope_10_100": slope(phi, 10, 100), "slope_100_1000": slope(phi, 100, 1000),
        "slope_10_tau": slope(phi, 10, tau), "n_10pct_dev_vs_infinite": n10, "n10_over_tau": n10 / tau,
        "phi_1000": phi[1000], "kappa_chunk8": float(phi[:8].sum()), "sum_phi_40000": float(phi.sum())}
    for ro in ("mean", "geom4"):
        p2 = run_scalar(mem(K, ro), K, 5000)
        res[f"K{K}_{ro}"] = {"phi0": p2[0], "slope_10_100": slope(p2, 10, 100), "slope_100_1000": slope(p2, 100, 1000),
                             "phi_1000": p2[1000], "phi_4000": p2[4000]}

# NS (unit) : trajectoires appariees, gradients aleatoires fixes (independants de M), evenement en t=0 ou non
K = 6; m = mem(K, "shallow", muon="unit"); torch.manual_seed(0)
T = 3000; gs = [torch.randn(1, 16, 8) for _ in range(T)]
def traj(with_event):
    w = torch.zeros(1, 16, 8); U = torch.zeros(1, 16, 8, K); ws = []
    for t in range(T):
        g = gs[t] if (t > 0 or with_event) else torch.zeros(1, 16, 8)
        st, S = m._step({"w": w}, {"w": U}, {"w": g}, torch.zeros(1), torch.zeros(1), torch.full((1,), 0.03))
        w, U = st["w"], S["w"]; ws.append(w)
    return ws
A_, B_ = traj(True), traj(False)
ev = (A_[0] - B_[0]); evn = ev / ev.norm()
proj = np.array([float(((A_[t] - B_[t]) * evn).sum() / ev.norm()) for t in range(T)])
ref6 = ideal(6, T)[:, 0]
res["K6_unit_paired"] = {"event_norm_spectral": float(torch.linalg.matrix_norm(ev[0], ord=2)),
    "max_abs_err_vs_phi": float(np.max(np.abs(proj - ref6))), "proj_1000": proj[1000], "phi_1000": ref6[1000]}
# derive de M-M0 sous NS, gradients decorrelees : ||M-M0||_F au fil du flux
res["K6_unit_drift_F"] = {str(t): float(A_[t].norm()) for t in (10, 100, 1000, 2999)}
print(json.dumps(res, indent=1, default=float))
json.dump(res, open("t/kernel_direct.json", "w"), indent=1, default=float)
