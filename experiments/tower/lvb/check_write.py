import torch
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.train_load import write_stream
b = torch.load("ckpt/dist_unit_s0_at4000.pt", map_location="cpu", weights_only=False)
a = b["args"]; pca = torch.load(a["basis"] if a["basis"].startswith("/") else "/home/josephleger/work/titans-mac/" + a["basis"], map_location="cpu", weights_only=False)["pca"]
br = MemoryBranch(2048, a.get("d_mem", 128), chunk_size=8, down_pca=pca, **memory_kwargs_from_args(a))
br.load_state_dict(b["branch"]); br.eval()
torch.manual_seed(0); z = torch.randn(2, 720, 128) * 3
with torch.no_grad():
    s1, S1, _, _ = write_stream(br, z, 8)
    s2, S2 = br.init_carry(2); s2, S2 = br.memory.update(z, s2, S2, 8)
for k in s1:
    if torch.is_tensor(s1[k]):
        print(k, float((s1[k] - s2[k]).abs().max()), float(s1[k].abs().max()))
