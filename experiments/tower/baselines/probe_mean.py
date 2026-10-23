"""Sonde CPU : la couleur de l aiguille est-elle lineairement decodable depuis une memoire triviale ?
Entrees = passe PCA (meme load que train_stream). Caracteristiques : moyenne de tout le passe (128 fl.),
FIFO des 128 dernieres ecritures aplaties (pas faite : trop large), moyenne par 3 (pool) moyennee.
Classifieur logistique multiclasse, videos d entrainement -> videos d eval (memes que C1)."""
import torch, random, sys
sys.path.insert(0, "src")
torch.set_num_threads(4)
from titans.graft.train_stream import load, past
vids, qs = load("/home/josephleger/work/iclr/stream/tgc/*.pt", "cache/qwen_l24_pca128.pt", 8)
qs = [q for q in qs if q["kind"] == "color"]
cols = sorted({q["meta"]["color"] for q in qs}); ci = {c: i for i, c in enumerate(cols)}
def feats(q):
    z = past(vids, q, 8, "cpu")[0]
    return torch.cat([z.mean(0), z[-128:].mean(0)])
tr = [q for q in qs if not q["eval"] and q["teacher_ok"]]
ev = [q for q in qs if q["eval"]]; random.Random(1).shuffle(ev); ev = ev[:300]
for name, sl in (("moyenne globale", slice(0, 128)), ("moyenne 128 dernieres ecritures", slice(128, 256))):
    X = torch.stack([feats(q)[sl] for q in tr]).float(); Y = torch.tensor([ci[q["meta"]["color"]] for q in tr])
    Xe = torch.stack([feats(q)[sl] for q in ev]).float(); Ye = torch.tensor([ci[q["meta"]["color"]] for q in ev])
    mu, sd = X.mean(0), X.std(0) + 1e-6; X, Xe = (X - mu) / sd, (Xe - mu) / sd
    W = torch.zeros(128, len(cols), requires_grad=True); b = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(X @ W + b, Y) + 1e-3 * (W ** 2).sum(); l.backward(); return l
    opt.step(cl)
    acc = ((Xe @ W + b).argmax(1) == Ye).float().mean().item()
    acc_tr = ((X @ W + b).argmax(1) == Y).float().mean().item()
    print(f"{name}: couleur ({len(cols)} classes, hasard {1/len(cols):.3f}) train {acc_tr:.3f} eval {acc:.3f} (n {len(ev)})", flush=True)
    # par distance
    for lo, hi in ((2, 4), (4, 10)):
        m = torch.tensor([lo <= q["c"] - q["p"] < hi for q in ev])
        print(f"   dist {lo}-{hi}: {((Xe @ W + b).argmax(1) == Ye)[m].float().mean().item():.3f} (n {int(m.sum())})", flush=True)
