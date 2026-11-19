"""Cache latent TEXTE NORMAL : WikiText-103, tokens BPE de Qwen2.5, embeddings de Qwen.

Pourquoi : le world model video fait collapser ``unit`` (eta -> 0, theta x10-25,
pr 3-10, collapse-is-horizon-retreat) et la greffe reproduit ce "cache" des que
l'objectif recompense la prediction locale. Hypothese :
c'est la LOCALITE de la donnee video (deux latents voisins se ressemblent) qui fait du
cache du dernier pas la meilleure solution. Le texte enwik8 (octets) est lui aussi tres
local ; ici, du texte normal, tokenise en sous-mots, avec la geometrie semantique des
embeddings d'entree de Qwen2.5 (PCA 128 blanchie, meme echelle que egolife_pca128 :
std 1 par dimension).

Format : celui d'``echo.py`` (``latents`` [N, T, 128] + pca/k/whiten/explained/
latent_dim_original). Train = fenetres tirees du split train, val = du split
validation (articles disjoints).

    PYTHONPATH=src python paper_a/scripts/build_bpe_cache.py \\
        --parquet-dir data/wikitext103 \\
        --out-train cache/wt103_qwen128_train.pt --out-val cache/wt103_qwen128_val.pt
"""
import argparse
import glob
import json
import os

import torch


def read_parquet_text(paths):
    import pyarrow.parquet as pq
    return "".join(t for p in paths for t in pq.read_table(p).column("text").to_pylist())


def qwen_embeddings(model_id: str) -> torch.Tensor:
    from safetensors import safe_open
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    root = sorted(glob.glob(os.path.join(hub, "models--" + model_id.replace("/", "--"),
                                         "snapshots", "*")))[-1]
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    key = next(k for k in idx if k.endswith("embed_tokens.weight"))
    with safe_open(os.path.join(root, idx[key]), "pt") as f:
        return f.get_tensor(key).float()                      # [V, D]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, dest="parquet_dir")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out-train", required=True, dest="out_train")
    ap.add_argument("--out-val", required=True, dest="out_val")
    ap.add_argument("--n-train", type=int, default=5999, dest="n_train")
    ap.add_argument("--n-val", type=int, default=1000, dest="n_val")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--fit-tokens", type=int, default=400000, dest="fit_tokens")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tr_txt = read_parquet_text(sorted(glob.glob(os.path.join(a.parquet_dir, "train-*.parquet"))))
    va_txt = read_parquet_text(sorted(glob.glob(os.path.join(a.parquet_dir, "validation-*.parquet"))))
    # le train complet fait ~540 Mo de texte : 60 Mo suffisent pour 6000 x 64 tokens
    tr_ids = torch.tensor(tok(tr_txt[:60_000_000])["input_ids"])
    va_ids = torch.tensor(tok(va_txt)["input_ids"])
    print("tokens train", len(tr_ids), "val", len(va_ids), flush=True)

    E = qwen_embeddings(a.model)                              # [V, 2048]
    g = torch.Generator().manual_seed(0)
    fit = E[tr_ids[torch.randint(0, len(tr_ids), (a.fit_tokens,), generator=g)]]
    mean = fit.mean(0)
    U, S, Vh = torch.linalg.svd(fit - mean, full_matrices=False)
    comps = Vh[:a.dim]                                        # [k, D]
    sd = (S[:a.dim] / (len(fit) - 1) ** 0.5)                  # ecart-type par composante
    explained = float((S[:a.dim] ** 2).sum() / (S ** 2).sum())

    def lat(ids, n, seed):
        g = torch.Generator().manual_seed(seed)
        st = torch.randint(0, len(ids) - a.T - 1, (n,), generator=g)
        seqs = torch.stack([ids[s:s + a.T] for s in st.tolist()])
        return ((E[seqs] - mean) @ comps.T) / sd              # [n, T, k], std ~1

    meta = {"pca": {"kind": "qwen2.5-embed-pca", "components": comps, "mean": mean,
                    "sd": sd, "model": a.model},
            "k": a.dim, "whiten": True, "explained": explained,
            "latent_dim_original": int(E.shape[1]),
            "source": "wikitext-103-raw-v1 (Salesforce/wikitext), Qwen2.5 BPE"}
    tr = lat(tr_ids, a.n_train, 1)
    va = lat(va_ids, a.n_val, 2)
    torch.save({"latents": tr, **meta}, a.out_train)
    torch.save({"latents": va, **meta}, a.out_val)
    # localite : cos moyen entre latents voisins (video egolife : a comparer)
    cos = torch.nn.functional.cosine_similarity(tr[:, 1:], tr[:, :-1], dim=-1).mean()
    print("train", tuple(tr.shape), "std", float(tr.std()), "explained", round(explained, 3),
          "cos(voisins)", float(cos), flush=True)
    print("val  ", tuple(va.shape), "std", float(va.std()), flush=True)


if __name__ == "__main__":
    main()
