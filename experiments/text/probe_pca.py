"""Input bottleneck probe: can a linear probe decode the colour (8 classes) / the number's tens digit from the
layer-24 state, full 2048 dims vs the text PCA-128 basis used by `down`? States at (a) the value token itself,
(b) the sentence's final '.' token (what a later write would carry). Train stories 3e6+, 75/25 split."""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pylib"))
import gen_story2 as G  # noqa: E402
import train_text_ab as T  # noqa: E402
from titans.worldmodel.bottleneck import apply_pca  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
host = T.TextHost("Qwen/Qwen2.5-VL-3B-Instruct", "cuda")
tok = host.tok
pca = torch.load(os.path.join(HERE, "basis_l24_text_pca128.pt"), weights_only=False)["pca"]
feats = {(f, w): [] for f in ("colour", "number") for w in ("val", "dot", "mean8", "stride8")}
labels = {"colour": [], "number": []}
for i in range(N):
    st = G.make_story(3_000_000 + i, "train")
    L = G.layout(st, tok, P=2048, W=256, D=0)
    v = i % 4
    h = host.layer_states(torch.tensor([L[v]["past"]], device="cuda"))[0]
    for fam in ("colour", "number"):
        it = st["items"][fam][v]
        who = it["question"].split("'s")[0].split()[-1]
        for s_, e_ in G.facts_of(L[v], fam):
            txt = tok.decode(L[v]["past"][s_:e_])
            if txt.strip().startswith(who + " ") or txt.strip().startswith(who + "'s"):
                ids = L[v]["past"][s_:e_]
                ans = it["answer_text"]
                pv = [j for j in range(len(ids)) if ans.startswith(tok.decode([ids[j]]).strip()) and tok.decode([ids[j]]).strip()]
                pv = pv[-1] if pv else len(ids) - 2
                feats[(fam, "val")].append(h[s_ + pv].cpu())
                feats[(fam, "dot")].append(h[e_ - 1].cpu())
                pos = s_ + pv
                g8 = pos // 8 * 8
                feats[(fam, "mean8")].append(h[g8:g8 + 8].mean(0).cpu())
                feats[(fam, "stride8")].append(h[g8].cpu())          # the token a stride-8 write keeps
                lab = G.COLOURS.index(ans) if fam == "colour" else int(ans) // 10
                labels[fam].append(lab)
                break


def probe(X, y, ncls):
    n = len(y)
    ntr = int(0.75 * n)
    X = (X - X[:ntr].mean(0)) / (X[:ntr].std(0) + 1e-6)
    W = torch.zeros(X.shape[1], ncls, requires_grad=True)
    b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(500):
        loss = F.cross_entropy(X[:ntr] @ W + b, y[:ntr]) + 1e-3 * W.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(((X[ntr:] @ W + b).argmax(1) == y[ntr:]).float().mean())


for fam, ncls in (("colour", len(G.COLOURS)), ("number", 10)):
    y = torch.tensor(labels[fam])
    for where in ("val", "dot", "mean8", "stride8"):
        X = torch.stack(feats[(fam, where)]).float()
        Z = apply_pca(X, pca)
        print(f"{fam:7s} token={where:3s} n={len(y)}  full2048 {100 * probe(X, y, ncls):5.1f}%   "
              f"pca128 {100 * probe(Z, y, ncls):5.1f}%   (chance {100 / ncls:.0f}%)", flush=True)
print("PROBE-FINI")
