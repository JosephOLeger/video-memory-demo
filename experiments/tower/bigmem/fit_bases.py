"""21/10 : bases PCA de la couche 24 de Qwen pour des memoires plus larges (k = 128 a 2048).
Une seule SVD sur un echantillon des clips des videos d entrainement du flux (tgc, eval=False) ;
les bases de toutes les tailles sont des troncatures de la meme SVD (axe taille sans confusion de base)."""
import glob, random, torch
from titans.worldmodel.bottleneck import fit_pca
random.seed(0); rows = []
for f in sorted(glob.glob("/home/josephleger/work/iclr/stream/tgc/*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    if d["eval"]:
        continue
    for c in random.sample(d["clips"], min(4, len(d["clips"]))):
        rows.append(c.float())
X = torch.cat(rows); X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:200000]]
print("echantillon", tuple(X.shape), flush=True)
full = fit_pca(X, 2048)
for k in (128, 256, 512, 1024, 2048):
    p = dict(full, k=k, components=full["components"][:k].clone(), sv=full["sv"][:k].clone())
    S2 = full["sv"] ** 2; p["explained"] = float(S2[:k].sum() / S2.sum())
    torch.save({"pca": p}, f"/home/josephleger/work/titans-mac/cache/qwen_l24_stream_pca{k}.pt")
    print(k, "variance expliquee %.3f" % p["explained"], flush=True)
print("BASES-FINIES")
