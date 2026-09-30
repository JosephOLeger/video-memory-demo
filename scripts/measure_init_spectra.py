"""Spectres des INITS APPRISES de la memoire (W1_0, W2_0, ...).

Le suspect designe par la forme de U : delta = (W2^T e) . sigma'(W1 k).
L'init de W2 est un poids LENT (mem.memory.params) — si la boucle
externe d'unit apprend une W2_0 degeneree, l'entonnoir est ferme avant
la premiere ecriture. Indice prealable : au gap 0 de la retention, le pr
d'etat des graines unit varie 20-72 et co-varie avec la severite du
collapse.

Pour chaque artefact et chaque tenseur 2D de l'init : pr (participation
des valeurs singulieres au carre), part du 1er axe, norme.
"""
import argparse
import glob
import json

import torch

from titans.worldmodel.frozen import load_frozen


def spec(t):
    s2 = torch.linalg.svdvals(t.float()) ** 2
    tot = float(s2.sum())
    pr = float(tot ** 2 / float((s2 ** 2).sum())) if tot > 0 else 0.0
    return pr, float(s2[0] / tot) if tot > 0 else 0.0, float(t.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(a.frozen)):
        model, _ = load_frozen(path, a.device)
        for bi, blk in enumerate(model.blocks):
            mem = getattr(blk, "memory", None)
            if mem is None:
                continue
            row = {"frozen": path, "block": bi}
            parts = []
            for key, t in mem.memory.params.items():
                if t.dim() != 2:
                    continue
                pr, top1, n = spec(t.detach())
                row[f"{key}_pr"] = pr
                row[f"{key}_top1"] = top1
                row[f"{key}_norm"] = n
                parts.append(f"{key}: pr {pr:5.1f} top1 {top1:.2f} n {n:5.1f}")
            rows.append(row)
            print(f"{path.split('/')[-1]:26s} b{bi} " + " | ".join(parts),
                  flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
