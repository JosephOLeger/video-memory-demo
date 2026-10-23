import torch
from titans.memory import NeuralMemory
cfgs = {"unit scalaire (eta, alpha appris)": dict(orthogonalize_momentum="unit"),
        "unit eta=0 alpha=2.4e-4": dict(orthogonalize_momentum="unit"),
        "direct K6 shallow": dict(orthogonalize_momentum="unit", use_cascade_momentum=True, cascade_reservoirs=6,
                                  cascade_mode="direct", cascade_leak="none", cascade_areas=2.0, cascade_terminal_leak=True)}
for name, cfg in cfgs.items():
    torch.manual_seed(0); m = NeuralMemory(dim=32, **cfg)
    if "eta=0" in name: m.eta_override, m.alpha_override = 0.0, 2.44e-4
    torch.manual_seed(1); x = torch.randn(2, 64, 32)
    st, S = m.init_memory(2); w0 = {k: v.detach().clone() for k, v in st.items()}
    st, S = m.update(x, st, S, 8)
    # signal de memoire : lecture des cles ecrites, difference avec M0
    k = m.k_proj(x[:, :8]); r = m.retrieve(st, k); r0 = m.retrieve(w0, k)
    loss = ((r - r0) ** 2).mean(); loss.backward()
    th = m.theta_head.weight.grad
    print(f"{name:34s} |dM| = {max(float((st[k]-w0[k]).norm()) for k in st):.4f}  loss {float(loss):.3e}  |grad theta_w| = {float(th.norm()) if th is not None else None:.3e}")
