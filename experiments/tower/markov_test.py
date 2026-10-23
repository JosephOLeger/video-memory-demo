"""Test de markovianite des flux de latents : ACF par retard + R2 de prediction lineaire
z_{t+D} ~ [z_t..z_{t-p+1}] (+ moyenne du passe), ridge, fit sur une moitie des flux, eval sur l'autre."""
import sys, json, torch
torch.manual_seed(0)
name, Z = sys.argv[1], None
src = sys.argv[2]
b = torch.load(src, map_location="cpu", weights_only=False)
if name == "graft":
    cl = [c for c in b["train"]["fillers"]] + [c for c in b["eval"]["fillers"]]
    cl = [c if torch.is_tensor(c) else c[0] for c in cl]
    L = max(set(c.shape[-2] for c in cl), key=[c.shape[-2] for c in cl].count)
    Z = torch.stack([c.reshape(-1, c.shape[-1])[:L] for c in cl if c.reshape(-1, c.shape[-1]).shape[0] >= L])
else:
    Z = b["latents"]
Z = Z.float()
N, T, d = Z.shape
mu = Z.reshape(-1, d).mean(0)
Zc = Z - mu
out = {"name": name, "shape": [N, T, d]}
def cos(a, b): return (torch.nn.functional.cosine_similarity(a, b, dim=-1)).mean().item()
lags = [l for l in [1, 2, 4, 8, 16, 32, 48, 64, 128, 192] if l < T - 4]
out["acf_raw"] = {l: cos(Z[:, :-l], Z[:, l:]) for l in lags}
out["acf_centred"] = {l: cos(Zc[:, :-l], Zc[:, l:]) for l in lags}
r1 = out["acf_centred"][1]
out["ar1_pred"] = {l: r1 ** l for l in lags}
perm = torch.randperm(N); tr, te = perm[: N // 2], perm[N // 2:]
def design(X, t0, p, D, with_mean):
    feats = [X[:, t0 - j] for j in range(p)]
    if with_mean: feats.append(X[:, : t0 + 1].mean(1))
    return torch.cat(feats, -1), X[:, t0 + D]
def r2(p, D, with_mean=False, lam=1e-1):
    ts = list(range(max(p - 1, 8), T - D, max(1, (T - D) // 24)))
    def stack(idx):
        A, Y = zip(*[design(Zc[idx], t, p, D, with_mean) for t in ts]); return torch.cat(A), torch.cat(Y)
    A, Y = stack(tr); At, Yt = stack(te)
    W = torch.linalg.solve(A.T @ A + lam * len(A) * torch.eye(A.shape[1]), A.T @ Y)
    return 1 - ((At @ W - Yt) ** 2).sum().item() / ((Yt - Yt.mean(0)) ** 2).sum().item()
res = {}
for D in [l for l in [1, 2, 4, 8, 16, 32, 64, 128] if l < T // 2]:
    res[D] = {f"p{p}": round(r2(p, D), 4) for p in [1, 2, 4, 8, 16] if p < T // 2}
    res[D]["p1+mean"] = round(r2(1, D, True), 4)
    res[D]["mean_only"] = round(r2(0, D, True), 4) if False else None
out["r2"] = res
print(json.dumps(out, indent=None))
