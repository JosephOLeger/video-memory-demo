"""Contre-verification du collapse intra-sequence lucidrains (31/08).

Le resultat a verifier : pr(etat) 63.7 -> 1-3 en ~1150 ecritures,
12/12 flux (R_lad_*). Trois controles par un chemin independant de la
sonde principale :
1. pr et norme des poids effectifs aux points debut/milieu/fin, sur LES
   4 TETES (la sonde ne lit que la tete 0), SVD numpy float64.
2. ||W_fin|| vs ||W_init|| : discrimine "rampe rang-1 dominante"
   (norme >> init) de "fenetre d'oubli courte" (norme ~ ecritures
   recentes) — deux mecanismes que pr seul ne distingue pas.
3. Signature FONCTIONNELLE sans passer par les poids : les lectures
   (retrieved) de la memoire pour des tokens differents doivent devenir
   quasi colineaires en fin de sequence si l'etat est rang-1.
"""
import argparse
import gzip
import itertools

import numpy as np
import torch

from probe_r_prlaw import build_model, pr_of_np
from titans_pytorch.neural_memory import NeuralMemory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_prlaw_final.pt")
    ap.add_argument("--enwik", default="./data/enwik8.gz")
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=4096, dest="seq_len")
    a = ap.parse_args()

    model = build_model().cuda()
    blob = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    with gzip.open(a.enwik) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    val = torch.from_numpy(data[int(90e6):])

    mems = [m for m in model.modules() if isinstance(m, NeuralMemory)]
    weights_series = [[] for _ in mems]   # tenseurs [bh, n, d1, d2]
    retrieved_series = [[] for _ in mems]  # tenseurs [b, n, d]

    def make_wrap(orig, idx):
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            state = out[1]
            upd = state.updates
            if upd is not None and len(list(upd.keys())) > 0:
                ks = [k for k in upd.keys()
                      if hasattr(upd[k], "numel") and upd[k].ndim >= 3]
                if ks:
                    key = max(ks, key=lambda k: upd[k][0].numel())
                    weights_series[idx].append(upd[key].detach().float().cpu())
            r = out[0]
            if torch.is_tensor(r):
                retrieved_series[idx].append(r.detach().float().cpu())
            return out
        return wrapped

    for i, m in enumerate(mems):
        m.forward = make_wrap(m.forward, i)

    def cos_stats(X):
        # X [n, d] -> |cos| moyen hors diagonale sur 64 tokens echantillonnes
        idx = torch.linspace(0, X.shape[0] - 1, 64).long()
        V = X[idx]
        V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
        C = (V @ V.T).abs()
        n = C.shape[0]
        return float((C.sum() - n) / (n * (n - 1)))

    g = torch.Generator().manual_seed(0)
    for s in range(a.seqs):
        for ws, rs in zip(weights_series, retrieved_series):
            ws.clear(); rs.clear()
        start = int(torch.randint(0, len(val) - a.seq_len - 1, (1,),
                                  generator=g))
        seq = val[start:start + a.seq_len].long().cuda()[None]
        with torch.no_grad():
            model(seq, return_loss=False)
        print(f"=== sequence {s} (debut {start}) ===", flush=True)
        for mi, (ws, rs) in enumerate(zip(weights_series, retrieved_series)):
            if not ws:
                continue
            W = torch.cat(ws, dim=1)          # [bh, N, d1, d2]
            bh, N = W.shape[0], W.shape[1]
            for h in range(bh):
                Wf = W[h].numpy().astype(np.float64)
                i0, i1, i2 = 0, N // 2, N - 1
                line = (f"mem{mi} tete{h}: N={N} "
                        f"pr {pr_of_np(Wf[i0]):.1f}/{pr_of_np(Wf[i1]):.1f}/"
                        f"{pr_of_np(Wf[i2]):.1f} "
                        f"norme {np.linalg.norm(Wf[i0]):.2f}/"
                        f"{np.linalg.norm(Wf[i1]):.2f}/"
                        f"{np.linalg.norm(Wf[i2]):.2f}")
                print(line, flush=True)
            R = torch.cat(rs, dim=1)[0]       # [n_tokens, d]
            nt = R.shape[0]
            early = R[nt // 16: nt // 16 + 512]
            late = R[-512:]
            print(f"mem{mi} lectures: |cos| tot debut {cos_stats(early):.3f} "
                  f"fin {cos_stats(late):.3f}", flush=True)
    print("VERIF-R-FINIE", flush=True)


if __name__ == "__main__":
    main()
