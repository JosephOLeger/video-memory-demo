import torch
from titans.memory import NeuralMemory
torch.manual_seed(0)
for ro in ("shallow", "mean", "geom4", "geo"):
    m = NeuralMemory(dim=32, use_cascade_momentum=True, cascade_reservoirs=6, cascade_mode="direct",
                     cascade_leak="none", cascade_areas=2.0, cascade_terminal_leak=True,
                     cascade_readout=ro, orthogonalize_momentum="unit")
    x = torch.randn(2, 64, 32)
    st, S = m.init_memory(2)
    w0 = {k: v.detach().clone() for k, v in st.items()}
    st, S = m.update(x, st, S, 8)
    r = m.retrieve(st, m.k_proj(x[:, -8:]))
    (r ** 2).mean().backward()
    gn = {n: (None if p.grad is None else float(p.grad.norm())) for n, p in m.named_parameters()
          if any(h in n for h in ("theta", "alpha", "eta"))}
    # M - M0 = c^T U exactly?
    err = max(float((st[k] - w0[k] - m.cascade.readout(S[k])).abs().max()) for k in st if S[k].dim() == st[k].dim() + 1)
    print(ro, "grads", gn, "| max|M-M0-c^T U| =", err, "| nan:", any(torch.isnan(v).any().item() for v in st.values()))
