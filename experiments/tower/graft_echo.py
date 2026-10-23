"""Rappel associatif isole sur les memoires de la GREFFE (protocole de la Fig. 3 du banc).

Par checkpoint train_load et par charge N : tirer N latents reels de la greffe (clips
d'eval de la banque, couche 24 apres PCA), les ecrire dans une memoire FRAICHE avec les
projections et les portes apprises (mem.update, sans l'hote ni la lecture co-adaptee),
interroger chaque item par sa CLE k_proj(x_i), et compter si la lecture est la plus
proche de sa propre valeur (top-1 parmi les N, cosinus). Plancher : memoire sans ecriture.

    PYTHONPATH=src python graft_echo.py --bank ~/work/iclr/load/bank.pt \
        --ckpts ~/work/iclr/echo_graft/ckpt/*.pt --out ~/work/iclr/echo_graft/echo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from titans.graft.branch import memory_kwargs_from_args, sane_memory_kwargs
from titans.memory import NeuralMemory


def pr_of_w(w):
    s2 = torch.linalg.svdvals(w.double()) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def score(r, v):
    rn = r / (r.norm(dim=-1, keepdim=True) + 1e-9)
    vn = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
    C = rn @ vn.T
    return float((C.argmax(1) == torch.arange(C.shape[0], device=C.device)).float().mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--loads", default="2,4,8,16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ev = torch.load(a.bank, map_location="cpu", weights_only=False)["eval"]["fillers"]
    pool = torch.cat([f.float() for f in ev], 0)          # [n_lat, 128], latents d'eval
    print(f"pool : {pool.shape[0]} latents d'eval", flush=True)
    loads = [int(x) for x in a.loads.split(",")]
    with open(a.out, "a") as fo:
        for path in a.ckpts:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            args = blob["args"]
            mem = NeuralMemory(dim=args.get("d_mem", 128),
                               **sane_memory_kwargs(**memory_kwargs_from_args(args)))
            sd = {k[len("memory."):]: v for k, v in blob["branch"].items()
                  if k.startswith("memory.")}
            missing, unexpected = mem.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected
            mem.to(a.device).eval()
            chunk = args.get("chunk", 8)
            name = os.path.basename(path)[:-3]
            for N in loads:
                t1, t0, prs, fin = [], [], [], 0
                for dr in range(a.draws):
                    g = torch.Generator().manual_seed(1000 * dr + N)
                    X = pool[torch.randperm(pool.shape[0], generator=g)[:N]].unsqueeze(0).to(a.device)
                    v = mem.v_proj(X)[0]
                    st0, _ = mem.init_memory(1)
                    t0.append(score(mem.retrieve(st0, mem.k_proj(X))[0], v))
                    st, S = mem.init_memory(1)
                    st, S = mem.update(X, st, S, chunk)
                    r = mem.retrieve(st, mem.k_proj(X))[0]
                    fin += int(torch.isfinite(r).all())
                    t1.append(score(r, v))
                    prs.append(pr_of_w(st["w1"][0]))
                row = {"ckpt": name, "N": N, "top1_k": sum(t1) / len(t1),
                       "top1_k_sd": float(torch.tensor(t1).std()),
                       "nomem": sum(t0) / len(t0), "pr_final": sum(prs) / len(prs),
                       "finite": fin / a.draws, "missing": len(missing)}
                fo.write(json.dumps(row) + "\n"); fo.flush()
                print(f"{name:28s} N={N:5d} top1_k={row['top1_k']:.3f} nomem={row['nomem']:.3f} "
                      f"pr={row['pr_final']:.1f}", flush=True)
    print("ECHO-FINI", flush=True)


if __name__ == "__main__":
    main()
