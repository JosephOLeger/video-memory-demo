"""Cache latent TEXTE : enwik8 -> embedding octet fixe 128-d.

Fabrique un cache au format exact des caches PCA egolife (dict avec
``latents`` [N, 64, 128] et les cles de metadonnees que echo.py lit :
``pca``, ``k``, ``whiten``, ``explained``, ``latent_dim_original``),
pour entrainer NOTRE banc sur un flux aux statistiques du texte.

Embedding : matrice fixe 256x128 ~ N(0,1) (graine 0) — par-dim std ~1 et
norme par frame ~ sqrt(128), la MEME echelle que le cache egolife
(mesure : std 1.01, norme 11.33). Aucun apprentissage dans l'embedding :
la geometrie des cles vue par la memoire est celle de la distribution
des octets (frequences skewees -> participation < 256), et la
correlation temporelle est celle du texte, pas celle des frames.

Train : sequences tirees des 90 premiers Mo ; val : des 10 derniers
(zones disjointes).

Usage (sur le box) :
  python scripts/build_text_cache.py \
    --enwik ~/repro/titans-pytorch/data/enwik8.gz \
    --out-train cache/enwik8_emb128_train.pt \
    --out-val cache/enwik8_emb128_val.pt
"""
import argparse
import gzip
import os

import torch


def make(raw: bytes, n: int, T: int, E: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(raw) - T - 1, (n,), generator=g)
    seqs = torch.stack([
        torch.frombuffer(raw[s:s + T], dtype=torch.uint8).long()
        for s in starts.tolist()
    ])
    return E[seqs]  # [n, T, 128]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enwik", required=True)
    ap.add_argument("--out-train", required=True, dest="out_train")
    ap.add_argument("--out-val", required=True, dest="out_val")
    ap.add_argument("--n-train", type=int, default=4000, dest="n_train")
    ap.add_argument("--n-val", type=int, default=1000, dest="n_val")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args()

    raw = gzip.open(os.path.expanduser(a.enwik), "rb").read()
    cut = int(len(raw) * 0.9)
    E = torch.randn(256, a.dim, generator=torch.Generator().manual_seed(0))

    meta = {"pca": {"kind": "byte-embedding-fixe", "note":
                    "cache texte fabrique par build_text_cache.py ; les cles "
                    "pca/k/whiten/explained ne sont la que pour le contrat "
                    "d'echo.py"},
            "k": a.dim, "whiten": True, "explained": 1.0,
            "latent_dim_original": 256}
    tr = make(raw[:cut], a.n_train, a.T, E, seed=1)
    va = make(raw[cut:], a.n_val, a.T, E, seed=2)
    torch.save({"latents": tr, **meta}, a.out_train)
    torch.save({"latents": va, **meta}, a.out_val)
    print("train", tuple(tr.shape), "std", float(tr.std()),
          "-> ", a.out_train)
    print("val  ", tuple(va.shape), "std", float(va.std()),
          "-> ", a.out_val)


if __name__ == "__main__":
    main()
