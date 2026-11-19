"""Paire causale decorrelation : caches stride-4 et controle depuis long256.

  enwik-like ? Non : video egolife, clips de 256 frames [1448, 256, 128].
  - stride4 : frames 0,4,8,...,252  -> [N, 64, 128], correlation successive
    reduite (les frames sont 4x plus espacees dans le temps reel), meme
    statistique marginale, meme tache (predire la frame suivante DU FLUX
    sous-echantillonne).
  - ctrl64  : frames 0..63          -> [N, 64, 128], le controle apparie
    (meme source, meme forme, correlation native).

Les metadonnees pca/k/whiten sont recopiees du cache source (contrat
echo.py).
"""
import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-train", required=True, dest="src_train")
    ap.add_argument("--src-val", required=True, dest="src_val")
    ap.add_argument("--out-prefix", required=True, dest="out_prefix")
    a = ap.parse_args()

    for split, src in [("train", a.src_train), ("val", a.src_val)]:
        o = torch.load(src, weights_only=False)
        z = o["latents"]
        meta = {k: v for k, v in o.items() if k != "latents"}
        meta["provenance"] = f"build_stride_cache.py depuis {src}"
        s4 = z[:, ::4, :].contiguous()
        c64 = z[:, :64, :].contiguous()
        torch.save({"latents": s4, **meta}, f"{a.out_prefix}_str4_{split}.pt")
        torch.save({"latents": c64, **meta}, f"{a.out_prefix}_ctrl64_{split}.pt")
        print(split, "str4", tuple(s4.shape), "ctrl64", tuple(c64.shape))


if __name__ == "__main__":
    main()
