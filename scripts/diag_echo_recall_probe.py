"""Rappel associatif isole-memoire sur artefacts geles (echo on-manifold).

But : convertir la separation GEOMETRIQUE mesuree par le probe pr
(usq reste diverse, unit collapse) en une separation FONCTIONNELLE.
On teste la memoire SEULE (aucune attention, aucun readout) : c'est ce
qui evite le confound de co-adaptation du reseau. Tout est on-manifold
(vrais latents val) et sans parametre libre (rappel top-1 par argmax).

Protocole, par (artefact, bloc, charge N) :
  1. tirer N tokens latents DISTINCTS (un par sequence val) -> X [1,N,d].
  2. ecrire ce flux dans une memoire FRAICHE : mem.update(X, ...).
     La memoire accumule N associations cle->valeur (comme MQAR), via
     sa propre voie k_proj/v_proj -- rien d'autre.
  3. requeter chaque item i avec q_proj(x_i) et lire r_i = M_W(q_i)
     (exactement comme memory_health), au poids final apres les N ecritures.
  4. scorer, sans seuil :
       - top1 : argmax_j cos(r_i, v_j) == i  (rappel associatif)
       - echo_gap : cos(r_i, v_i) - moyenne_{j!=i} cos(r_i, v_j)
         (specificite-contenu, la metrique [V;W])
     Chance top1 = 1/N.

On balaye N (la charge). Une memoire DIVERSE tient plus d'items separes
(top1 haut a grand N) qu'une memoire COLLAPSEE (top1 -> 1/N).

Sortie JSONL : un en-tete par artefact, puis une ligne par (bloc, N).
"""
from __future__ import annotations

import argparse
import glob
import json

import torch

from titans.worldmodel.frozen import load_frozen
from titans.worldmodel.train import load_latent_cache


def pr_of_w(w: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(w.double())
    sv2 = sv ** 2
    return float(sv2.sum() ** 2 / (sv2 ** 2).sum())


@torch.no_grad()
def sample_items(va: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """N tokens latents DISTINCTS : un token aleatoire dans N sequences
    distinctes (items maximalement separes, toujours on-manifold).
    Repli sur un tirage a plat si N > nombre de sequences."""
    S, T, d = va.shape
    if n <= S:
        seqs = torch.randperm(S, generator=gen)[:n]
        toks = torch.randint(0, T, (n,), generator=gen)
        X = va[seqs, toks]                       # [n, d]
    else:
        flat = va.reshape(S * T, d)
        idx = torch.randperm(S * T, generator=gen)[:n]
        X = flat[idx]
    return X.unsqueeze(0)                          # [1, n, d]


def _score(r: torch.Tensor, v: torch.Tensor) -> tuple[float, float]:
    """top1 (argmax_j cos(r_i, v_j) == i) et echo_gap (diag - moyenne off-diag)."""
    rn = r / (r.norm(dim=-1, keepdim=True) + 1e-9)
    vn = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
    C = (rn @ vn.T).double()                       # [N, N] cos(r_i, v_j)
    N = C.shape[0]
    top1 = float((C.argmax(dim=1) == torch.arange(N, device=C.device)).double().mean())
    eye = torch.eye(N, dtype=torch.bool, device=C.device)
    echo_gap = float((C[eye] - C[~eye].reshape(N, N - 1).mean(dim=1)).mean())
    return top1, echo_gap


@torch.no_grad()
def recall_at_load(mem, chunk, X: torch.Tensor, device: str) -> dict:
    """Ecrit X (N items) dans une memoire fraiche, lit, score top1 + echo_gap.

    Deux requetes :
      - q : mem.q_proj(X)  -> voie d'inference reelle (comme memory_health).
      - k : mem.k_proj(X)  -> objectif de reconstruction propre a la memoire
            (M_W(k_i) approx v_i). retrieve applique norm+feature_map, donc
            passer k_proj(X) brut reproduit exactement la cle ecrite.
    """
    X = X.to(device)
    v = mem.v_proj(X)[0]                            # [N, d]  banque de valeurs
    # --- baseline SANS memoire : lire a l'init APPRIS, aucune ecriture.
    # C'est le pendant de l'ablation memory-free : la memoire ne contient
    # aucun des N items, donc le rappel = plancher "pas de contenu".
    state0, _ = mem.init_memory(1)
    r0 = mem.retrieve(state0, mem.k_proj(X))[0]
    t1_0, eg_0 = _score(r0, v)
    # --- memoire ecrite : N ecritures, memoire seule (aucune attention).
    state, S = mem.init_memory(1)
    state, S = mem.update(X, state, S, chunk)
    rq = mem.retrieve(state, mem.q_proj(X))[0]      # requete = q_proj
    rk = mem.retrieve(state, mem.k_proj(X))[0]      # requete = k_proj (reconstruction)
    finite = bool(torch.isfinite(rq).all()) and bool(torch.isfinite(state["w1"]).all())
    t1q, egq = _score(rq, v)
    t1k, egk = _score(rk, v)
    return {"top1": t1q, "echo_gap": egq, "top1_k": t1k, "echo_gap_k": egk,
            "top1_nomem": t1_0, "echo_gap_nomem": eg_0,
            "pr_final": pr_of_w(state["w1"][0]), "finite": finite}


@torch.no_grad()
def probe_one(path: str, va: torch.Tensor, a) -> list[dict]:
    model, info = load_frozen(path, a.device)
    model.eval()
    rows = [{"frozen": path, "loads": a.loads, "draws": a.draws,
             "n_blocks": len(model.blocks)}]
    for bi, blk in enumerate(model.blocks):
        mem = blk.memory
        chunk = blk.memory_chunk_size
        if a.eta_override is not None:
            mem.eta_override = a.eta_override
        if a.alpha_override is not None:
            mem.alpha_override = a.alpha_override
        if a.theta_override is not None:
            mem.theta_override = a.theta_override
        for N in a.loads:
            acc = {k: [] for k in ("top1", "echo_gap", "top1_k", "echo_gap_k",
                                   "top1_nomem", "echo_gap_nomem", "pr_final")}
            fins = 0
            for dr in range(a.draws):
                gen = torch.Generator().manual_seed(a.seed + 1000 * bi + N + dr)
                X = sample_items(va, N, gen)
                res = recall_at_load(mem, chunk, X, a.device)
                for k in acc:
                    acc[k].append(res[k])
                fins += int(res["finite"])
            k = a.draws
            mean = lambda z: sum(z) / k
            sd = lambda z, m: (sum((x - m) ** 2 for x in z) / k) ** 0.5
            row = {"frozen": path, "block": bi, "N": N, "draws": k,
                   "eta_override": a.eta_override, "alpha_override": a.alpha_override,
                   "theta_override": a.theta_override,
                   "chance": round(1.0 / N, 4), "finite_frac": round(fins / k, 3)}
            for key in acc:
                m = mean(acc[key])
                row[key + "_mean"] = round(m, 4)
                row[key + "_std"] = round(sd(acc[key], m), 4)
            rows.append(row)
            print(f"  {path.split('/')[-1]} b{bi} N={N:>4} "
                  f"top1_k={row['top1_k_mean']:.3f} nomem={row['top1_nomem_mean']:.3f} "
                  f"(chance {1.0/N:.3f}) egap_k={row['echo_gap_k_mean']:+.3f} "
                  f"pr={row['pr_final_mean']:.1f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--loads", default="2,4,8,16,32,64,128")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--eta-override", type=float, default=None, dest="eta_override")
    ap.add_argument("--alpha-override", type=float, default=None, dest="alpha_override")
    ap.add_argument("--theta-override", type=float, default=None, dest="theta_override")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.loads = [int(x) for x in a.loads.split(",") if x]

    va, _ = load_latent_cache(a.val)
    files = sorted(glob.glob(a.frozen))
    if not files:
        raise SystemExit(f"aucun artefact ne matche {a.frozen}")
    with open(a.out, "w") as f:
        for path in files:
            print(f"--- {path}", flush=True)
            for row in probe_one(path, va, a):
                f.write(json.dumps(row) + "\n")
            f.flush()
    print("ECHO-RECALL-PROBE-FINIE", flush=True)


if __name__ == "__main__":
    main()
