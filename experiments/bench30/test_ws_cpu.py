"""03/11 CPU: ws_install (copied from train_b30_sel.py source) on a tiny NeuralMemory; no model, no GPU."""
import os, re, sys, textwrap, types, inspect
import torch
from titans.memory import NeuralMemory
from titans.graft.train_load import write_stream
src = open(os.path.expanduser("~/work/iclr/bench30/train_b30_sel.py")).read()
m = re.search(r"    def ws_install\(mem, log\):.*?return lambda: delattr\(mem, \"_step\"\)\n", src, re.S)
ns = {"torch": torch}; exec(textwrap.dedent(m.group(0)), ns)
print("NeuralMemory params:", list(inspect.signature(NeuralMemory.__init__).parameters)[:8])
torch.manual_seed(0)
mem = NeuralMemory(16, hidden_dim=8, orthogonalize_momentum="unit")
br = types.SimpleNamespace(memory=mem, init_carry=mem.init_memory)
z = torch.randn(1, 40, 16)
with torch.no_grad():
    ref, _, _, gates = write_stream(br, z, 16)
    log = []; undo = ns["ws_install"](mem, log)
    st, _, _, _ = write_stream(br, z, 16)
    undo()
W = torch.stack(log)
print("writes logged:", W.shape, "| state identical with hook:", all(torch.equal(ref[k], st[k]) for k in ref),
      "| hook removed:", "_step" not in mem.__dict__)
print("theta matches gates:", torch.allclose(W[:, 0], gates[2][0]), "eta:", torch.allclose(W[:, 2], gates[1][0]),
      "alpha:", torch.allclose(W[:, 3], gates[0][0]))
print("first rows [theta, |g|, eta, alpha, |dW|]:\n", W[:3])
