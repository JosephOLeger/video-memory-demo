"""Loss dense normalisee (MSE/var, comme train_load) de predicteurs TRIVIAUX, sur les latents
de la greffe (banque, clips de remplissage) : la memoire effondree fait-elle mieux qu'un tampon ?"""
import torch
b = torch.load("/home/josephleger/work/iclr/load/bank.pt", map_location="cpu", weights_only=False)
def pairs(split, delta, feats):
    X, Y = [], []
    for f in b[split]["fillers"]:
        z = f.float()
        n = z.shape[0]
        for t in range(4, n):
            u = t + delta
            if 0 <= u < n:
                X.append(feats(z, t)); Y.append(z[u])
    return torch.stack(X), torch.stack(Y)
def ema(z, t, a=0.5):
    m = z[0]
    for i in range(1, t + 1): m = a * m + (1 - a) * z[i]
    return m
FE = {"moyenne seule": lambda z, t: torch.zeros(1),
      "dernier latent z_t": lambda z, t: z[t],
      "z_t, z_{t-1}": lambda z, t: torch.cat([z[t], z[t-1]]),
      "4 derniers": lambda z, t: torch.cat([z[t-i] for i in range(4)]),
      "EMA 0,5": lambda z, t: ema(z, t)}
for delta in (1, -1):
    print(f"cible z_(t{delta:+d})")
    for name, fe in FE.items():
        X, Y = pairs("train", delta, fe); Xe, Ye = pairs("eval", delta, fe)
        X1 = torch.cat([X, torch.ones(len(X), 1)], 1); Xe1 = torch.cat([Xe, torch.ones(len(Xe), 1)], 1)
        lam = 1e-1 * torch.eye(X1.shape[1])
        W = torch.linalg.solve(X1.T @ X1 + lam, X1.T @ Y)
        P = Xe1 @ W
        print(f"  {name:20s} loss dense {float(((P - Ye) ** 2).mean() / Ye.var()):.3f}")
