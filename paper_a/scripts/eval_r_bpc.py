"""Ancrage familier : bpc de validation enwik8 sur la pile lucidrains.

Metrique standard (bits par caractere) + profil par position dans la
sequence (8 tranches de 512 sur 4096 tokens) — le profil relie le bpc a
l'histoire de la fenetre memoire. --psic active la regle de magnitude
(obligatoire pour evaluer ckpt_psic2_*, entraine avec la regle).
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_r_prlaw import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--psic", action="store_true")
    ap.add_argument("--enwik", default="./data/enwik8.gz")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=4096, dest="seq_len")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.psic:
        import patch_psic
        patch_psic.install()

    model = build_model().cuda()
    blob = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    with gzip.open(a.enwik) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    val = torch.from_numpy(data[int(90e6):])

    g = torch.Generator().manual_seed(1234)
    B = 8
    tot = torch.zeros(B)
    cnt = torch.zeros(B)
    losses = []
    with torch.no_grad():
        for _ in range(a.seqs):
            start = int(torch.randint(0, len(val) - a.seq_len - 2, (1,),
                                      generator=g))
            seq = val[start:start + a.seq_len + 1].long().cuda()[None]
            logits = model(seq[:, :-1], return_loss=False)
            ce = F.cross_entropy(logits.transpose(1, 2), seq[:, 1:],
                                 reduction="none")[0].cpu()
            losses.append(float(ce.mean()))
            L = ce.shape[0]
            bs = L // B
            for b in range(B):
                tot[b] += ce[b * bs:(b + 1) * bs].sum()
                cnt[b] += bs
    ln2 = float(np.log(2))
    res = {"ckpt": a.ckpt, "psic": a.psic, "seqs": a.seqs,
           "seq_len": a.seq_len,
           "bpc_mean": float(np.mean(losses)) / ln2,
           "bpc_par_position": [float(t / c) / ln2
                                for t, c in zip(tot, cnt)]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
