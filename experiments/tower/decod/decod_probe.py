"""Sonde de decodabilite, etape 2 : un LECTEUR ORACLE peut-il repondre aux QCM depuis les latents ?

Pour chaque QCM : Z = tokens des clips du passe (clips[:end-1]) dans une representation
donnee, E = embeddings des 4 options (texte seul, couche 24). Lecteur entraine :
    a_tj = <P_z z_t, P_e e_j> / sqrt(r)          (r = 64 pour toutes les representations)
    s_j  = logsumexp_t(beta a_tj) / beta + u . e_j  (le second terme = a priori du texte)
Validation croisee a 5 plis par video. Mesures sur les plis tenus a l'ecart :
    bonne  : Z de la bonne video
    autre  : Z d'une autre video (meme fin) -> contenu = bonne - autre
    texte  : un lecteur entraine SANS video (a priori seul)
Representations : brut 2048 (sub / pool), PCA 128 / 256 / 512 (sub et pool), PCA deployee 128.
C'est un plafond optimiste (lecteur dedie, acces a tous les tokens) : si meme lui ne lit
rien en PCA 128, la memoire ne peut pas faire mieux.

    python decod_probe.py --feat DIR --basis cache/qwen_l24_pca128.pt --out res.json
"""
import argparse
import glob
import hashlib
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_of(vid, k=5):
    return int(hashlib.md5(vid.encode()).hexdigest()[:8], 16) % k


class Reader(nn.Module):
    def __init__(self, d, de=2048, r=64, use_video=True):
        super().__init__()
        self.use_video = use_video
        self.pz, self.pe = nn.Linear(d, r), nn.Linear(de, r)
        self.prior = nn.Linear(de, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.r = r

    def forward(self, Z, M, E):                   # Z [B,T,d], M [B,T] bool, E [B,4,de]
        s = self.prior(E).squeeze(-1)
        if self.use_video:
            a = torch.einsum("btr,bjr->bjt", self.pz(Z), self.pe(E)) / math.sqrt(self.r)
            b = F.softplus(self.beta) + 1e-3
            a = a.masked_fill(~M[:, None, :], -1e4)
            s = s + torch.logsumexp(b * a, -1) / b
        return s


def fit_pca(X, k):
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mean) / std
    U, S, V = torch.linalg.svd(Xs, full_matrices=False)
    comps, sv = V[:k], S[:k] / math.sqrt(X.shape[0])
    expl = float((S[:k] ** 2).sum() / (S ** 2).sum())
    return lambda h: ((h - mean) / std) @ comps.T / sv, expl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    files = sorted(glob.glob(a.feat + "/*.pt"))
    V, items, opts = {}, [], {}
    for f in files:
        d = torch.load(f, weights_only=False)
        vid = d["mcq"][0]["vid"]
        V[vid] = {"sub": d["sub"], "pool": d["pool"]}          # 11 clips [n_tok, 2048], n_tok variable
        opts.update(d["opts"])
        items += [m for m in d["mcq"]]
    print(f"{len(V)} videos, {len(items)} QCM", flush=True)
    okeys = sorted(opts)
    oidx = {o: i for i, o in enumerate(okeys)}
    Eall = torch.stack([opts[o] for o in okeys]).float()
    Eall = ((Eall - Eall.mean(0)) / (Eall.std(0) + 1e-6)).to(dev)

    vids = sorted(V)
    g = torch.Generator().manual_seed(0)
    def allt(v): return torch.cat(V[v]["sub"])
    sample = torch.cat([allt(v)[torch.randperm(allt(v).shape[0], generator=g)[:64]]
                        for v in vids]).float()
    dep = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    dW = (dep["components"] / dep["std"][None] / dep["sv"][:, None]).float()
    db = (-((dep["mean"] / dep["std"]) @ dep["components"].T) / dep["sv"]).float()
    raw_mean, raw_std = sample.mean(0), sample.std(0) + 1e-6
    reps = {"brut2048": lambda h: (h - raw_mean) / raw_std,
            "pca128_deploye": lambda h: h @ dW.T + db}
    expl = {}
    for k in (128, 256, 512):
        reps[f"pca{k}"], expl[k] = fit_pca(sample, k)
    print("variance expliquee (sur cet echantillon) :", {k: round(v, 3) for k, v in expl.items()},
          "| deployee (ajustee ailleurs) :", round(float(dep["explained"]), 3), flush=True)
    plan = [("brut2048", "sub"), ("brut2048", "pool"), ("pca128_deploye", "sub"),
            ("pca128", "sub"), ("pca256", "sub"), ("pca512", "sub"),
            ("pca128", "pool"), ("pca256", "pool"), ("pca512", "pool")]

    ends = sorted({m["end"] for m in items})
    Tmax = max(sum(c.shape[0] for c in V[m["vid"]]["sub"][:m["end"] - 1]) for m in items)
    res = {"n_items": len(items), "expl": expl, "rows": []}
    for rep, kind in [("texte", None)] + plan:
        if rep != "texte":
            f = reps[rep]
            Zv = {v: [f(c.float()).half() for c in V[v][kind]] for v in vids}
            d = Zv[vids[0]][0].shape[-1]
        else:
            d = 1
        accs = {"bonne": [], "autre": []}
        bykind = {"activity": [], "prop": []}
        for seed in range(a.seeds):
            rng = random.Random(seed)
            hits, hits_w, kinds = [], [], []
            for fold in range(5):
                tr = [m for m in items if fold_of(m["vid"]) != fold]
                te = [m for m in items if fold_of(m["vid"]) == fold]
                torch.manual_seed(seed * 10 + fold)
                net = Reader(d, use_video=rep != "texte").to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

                def batch(ms, wrong=False):
                    B = len(ms)
                    Z = torch.zeros(B, Tmax, d, device=dev)
                    M = torch.zeros(B, Tmax, dtype=torch.bool, device=dev)
                    if rep != "texte":
                        for i, m in enumerate(ms):
                            src = m
                            if wrong:
                                pool = sorted({x["vid"] for x in te if x["vid"] != m["vid"]}) or [v for v in vids if v != m["vid"]]
                                h = int(hashlib.md5((m["vid"] + m["question"] + "".join(m["options"])).encode()).hexdigest()[:8], 16)
                                src = {"vid": pool[h % len(pool)]}
                            z = torch.cat(Zv[src["vid"]][:m["end"] - 1]).float()
                            Z[i, :z.shape[0]], M[i, :z.shape[0]] = z.to(dev), True
                    E = Eall[torch.tensor([[oidx[o] for o in m["options"]] for m in ms], device=dev)]
                    y = torch.tensor([m["answer"] for m in ms], device=dev)
                    return Z, M, E, y

                for ep in range(a.epochs):
                    rng.shuffle(tr)
                    for b0 in range(0, len(tr), 64):
                        Z, M, E, y = batch(tr[b0:b0 + 64])
                        if rep != "texte":
                            Z = F.dropout(Z, 0.1)
                        loss = F.cross_entropy(net(Z, M, E), y)
                        opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    for b0 in range(0, len(te), 64):
                        ms = te[b0:b0 + 64]
                        Z, M, E, y = batch(ms)
                        hits += (net(Z, M, E).argmax(1) == y).tolist()
                        Z, M, E, y = batch(ms, wrong=True)
                        hits_w += (net(Z, M, E).argmax(1) == y).tolist()
                        kinds += [m["kind"] for m in ms]
            accs["bonne"].append(sum(hits) / len(hits))
            accs["autre"].append(sum(hits_w) / len(hits_w))
            for k_ in bykind:
                sel = [i for i, x in enumerate(kinds) if x == k_]
                bykind[k_].append((sum(hits[i] for i in sel) / len(sel), sum(hits_w[i] for i in sel) / len(sel)))
        row = {"rep": rep, "tokens": kind, "d": d,
               "bonne": sum(accs["bonne"]) / a.seeds, "autre": sum(accs["autre"]) / a.seeds,
               "par_type": {k_: [sum(x[0] for x in v) / a.seeds, sum(x[1] for x in v) / a.seeds]
                            for k_, v in bykind.items()}}
        res["rows"].append(row)
        pt = "  ".join(f"{k_} {100 * v[0]:.1f}/{100 * v[1]:.1f}" for k_, v in row["par_type"].items())
        print(f"{rep:15s} {str(kind):5s} bonne {100 * row['bonne']:.1f} | autre {100 * row['autre']:.1f} "
              f"| contenu {100 * (row['bonne'] - row['autre']):+.1f}   ({pt})", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("DECOD-PROBE-FINI", flush=True)


if __name__ == "__main__":
    main()
