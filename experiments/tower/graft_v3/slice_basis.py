"""Slice the k=1024 basis to its first k components (PCA comps are ordered): d_mem sweep without refit."""
import sys, torch
src, k, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
b = torch.load(src, weights_only=False)
p = dict(b["pca"])
S2 = None
p["components"], p["sv"], p["k"] = p["components"][:k].clone(), p["sv"][:k].clone(), k
b["pca"], b["sliced_from"] = p, src
torch.save(b, dst)
print("saved", dst, "k", k)
