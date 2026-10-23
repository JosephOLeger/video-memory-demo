"""K=4 avec tau terminal = 16000 (~4T, T=4000) vs tau natif, readouts shallow/geo/mean (20/10)."""
import json, math, numpy as np, torch, sys
sys.path.insert(0, "t")
torch.set_default_dtype(torch.float64)
from titans.memory import NeuralMemory

def mem(K, ro, tau):
    return NeuralMemory(dim=8, use_cascade_momentum=True, cascade_reservoirs=K, cascade_mode="direct",
                        cascade_leak="none", cascade_areas=2.0, cascade_g=2.0, cascade_beta1=0.5,
                        cascade_terminal_leak=True, cascade_terminal_tau=tau, cascade_readout=ro)
def run(m, K, n):
    w = torch.zeros(1, 1, 1); U = torch.zeros(1, 1, 1, K); out = []
    a = torch.zeros(1); e = torch.zeros(1); th = torch.ones(1)
    for t in range(n):
        g = torch.full((1, 1, 1), -1.0 if t == 0 else 0.0)
        st, S = m._step({"w": w}, {"w": U}, {"w": g}, a, e, th)
        w, U = st["w"], S["w"]; out.append(float(w.reshape(())))
    return np.array(out)
pts = [1, 10, 100, 256, 1000, 2000, 4000, 8000, 16000, 32000, 60000]
N = 60001; res = {}
for K, tau in ((4, None), (4, 16000), (6, None), (6, 16000), (8, None)):
    for ro in ("shallow", "geo", "mean"):
        phi = run(mem(K, ro, tau), K, N)
        sl = lambda a, b: (math.log(phi[b]) - math.log(phi[a])) / (math.log(b) - math.log(a))
        r = {f"phi_{p}": phi[p] for p in pts}
        r.update(slope_10_100=sl(10, 100), slope_100_1000=sl(100, 1000), slope_1000_4000=sl(1000, 4000),
                 slope_4000_16000=sl(4000, 16000), plateau_ratio_phi4000_over_phi1000=phi[4000] / phi[1000],
                 sum_phi_4000=float(phi[:4000].sum()), sum_phi_60000=float(phi.sum()))
        res[f"K{K}_tau{tau or 'nat'}_{ro}"] = r
        print(f"K{K} tau={tau or 'nat':>6} {ro:8s} " + " ".join(f"{p}:{phi[p]:.4f}" for p in pts)
              + f" | pentes {r['slope_10_100']:.2f} {r['slope_100_1000']:.2f} {r['slope_1000_4000']:.2f} {r['slope_4000_16000']:.2f}"
              + f" | Σφ(4000) {r['sum_phi_4000']:.0f}", flush=True)
json.dump(res, open("t/kernel_k4tau.json", "w"), indent=1, default=float)
