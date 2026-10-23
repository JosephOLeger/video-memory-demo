"""Dump outputs/state/grads of NeuralMemory for a config (run under each PYTHONPATH)."""
import json, sys, torch
from titans.memory import NeuralMemory
cfg = json.loads(sys.argv[1]); out = sys.argv[2]
torch.manual_seed(0)
mem = NeuralMemory(dim=32, **cfg)
torch.manual_seed(1)
x = torch.randn(2, 48, 32)
st, S = mem.init_memory(2)
st, S = mem.update(x, st, S, 8)
k = mem.k_proj(x[:, -8:])
r = mem.retrieve(st, k)
loss = (r ** 2).mean()
loss.backward()
torch.save({"r": r.detach(), "st": {k: v.detach() for k, v in st.items()},
            "S": {k: v.detach() for k, v in S.items()},
            "g": {n: (p.grad.clone() if p.grad is not None else None) for n, p in mem.named_parameters()}}, out)
print("dumped", out)
