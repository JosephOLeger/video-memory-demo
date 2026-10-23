"""21/10 : la memoire couleur donne-t-elle une importance particuliere a l'aplat colore ?

Sans Qwen (CPU) : l'ecriture ne lit que les latents PCA des clips. Pour chaque question
d'evaluation couleur (videos jamais vues), on ecrit le passe clip par clip dans une memoire
fraiche et on mesure la norme de l'ecriture de chaque clip, ||W_i - W_{i-1}||_F (etat complet
des poids rapides). On compare le clip de l'aplat aux autres clips de la meme question :
  ratio = norme(aplat) / mediane(normes des autres clips) ; top = l'aplat est-il l'ecriture max ?
Temoin : la meme question SANS aplat (clip d'origine a la meme position), meme mesure.

    PYTHONPATH=src python diag_write.py --ckpts a.pt b.pt --targets 'tgc/*.pt' --basis ... --out x.jsonl
"""
import argparse, json, statistics, types
import torch
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.train_load import write_stream
from titans.graft.train_stream import load, past

HIDDEN = 2048                                   # Qwen2.5-VL-3B, la largeur ne sert qu'a la branche de lecture


def flat(p):
    if isinstance(p, dict):
        p = [p[k] for k in sorted(p)]
    return torch.cat([t.reshape(-1).float() for t in p])


@torch.no_grad()
def norms(branch, vids, q, max_past, chunk):
    lo = max(0, q["c"] - max_past)
    z = past(vids, q, max_past, "cpu")
    v = vids[q["vid"]]
    sizes = [v["clips"][i].shape[0] for i in range(lo, q["c"])]
    prev, out, pos = flat(branch.init_carry(1)[0]), [], 0
    for n in sizes:
        pos += n
        st, _, _, _ = write_stream(branch, z[:, :pos], chunk)
        cur = flat(st); out.append((cur - prev).norm().item()); prev = cur
    return out, q["p"] - lo


def summ(rows):
    r = [x["ratio"] for x in rows]
    return {"ratio_med": round(statistics.median(r), 3), "ratio_moy": round(sum(r) / len(r), 3),
            "top": round(sum(x["top"] for x in rows) / len(rows), 3), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    a0 = ap.parse_args()
    torch.set_num_threads(8)
    vids, qs = load(a0.targets, a0.basis, 8)
    qs = [q for q in qs if q["kind"] == "color" and q["eval"] and q["patch"] is not None][:a0.n]
    print(f"{len(qs)} questions d'evaluation couleur", flush=True)
    pca = torch.load(a0.basis, map_location="cpu", weights_only=False)["pca"]
    with open(a0.out, "a") as fo:
        for path in a0.ckpts:
            rnd = path.startswith("RANDOM:")          # memes reglages, poids initiaux (temoin non entraine)
            path = path.removeprefix("RANDOM:")
            blob = torch.load(path, map_location="cpu", weights_only=False)
            torch.manual_seed(0)
            a = types.SimpleNamespace(**blob["args"])
            branch = MemoryBranch(HIDDEN, a.d_mem, chunk_size=a.chunk, write_stride=a.write_stride,
                                  down_pca=pca, **memory_kwargs_from_args(a))
            if not rnd:
                branch.load_state_dict(blob["branch"])
            branch.eval()
            res = {"aplat": [], "temoin": []}
            for q in qs:
                for tag, qq in (("aplat", q), ("temoin", dict(q, patch=None))):
                    w, j = norms(branch, vids, qq, 8, a.chunk)
                    if not 0 <= j < len(w) or len(w) < 3:
                        continue
                    others = [x for i, x in enumerate(w) if i != j]
                    res[tag].append({"ratio": w[j] / max(statistics.median(others), 1e-12),
                                     "top": int(w[j] >= max(others))})
            row = {"ckpt": ("INIT:" if rnd else "") + path.split("/")[-1], **{k: summ(v) for k, v in res.items() if v}}
            fo.write(json.dumps(row) + "\n"); fo.flush()
            print(json.dumps(row), flush=True)
    print("DIAG-WRITE-FINI", flush=True)


if __name__ == "__main__":
    main()
