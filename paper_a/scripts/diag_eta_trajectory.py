"""Quel mode d'entrainement eteint eta, et quand ? Portes au rejeu, le long de l'entrainement.

Pour chaque artefact gele (typiquement une trajectoire ``<arm>_s<k>_at<step>.pt``),
on rejoue un flux de validation (``--n-seq`` sequences enchainees, carry conserve)
et on enregistre, par bloc, les portes que ``memory._project`` produit sur les VRAIES
activations du tronc (pas sur les latents bruts : cf. la correction de methode de
measure_alpha). On rapporte eta median, la fraction de tokens a eta < 0.5 (momentum
eteint), alpha et theta medians, et le pr de W1 en fin de flux.

    PYTHONPATH=src python scripts/diag_eta_trajectory.py \\
        --frozen 'frozen_muon/unit30k_s0_at*.pt' --val cache/egolife_val_8000.pt \\
        --out results/eta_traj/unit30k_s0.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import torch

from titans.worldmodel.frozen import load_frozen
from titans.worldmodel.train import load_latent_cache


def pr(w: torch.Tensor) -> float:
    s2 = torch.linalg.svdvals(w.double()) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def step_of(path: str) -> int:
    m = re.search(r"_at(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def probe(path, va, a):
    model, info = load_frozen(path, a.device)
    rec = {}
    for bi, blk in enumerate(model.blocks):
        mem = getattr(blk, "memory", None)
        if mem is None:
            continue
        rec[bi] = {"eta": [], "alpha": [], "theta": []}
        orig = mem._project

        def wrapped(x, _orig=orig, _r=rec[bi]):
            out = _orig(x)
            _, _, _, al, et, th = out
            _r["alpha"].append(al.detach().float().flatten().cpu())
            _r["eta"].append(et.detach().float().flatten().cpu())
            _r["theta"].append(th.detach().float().flatten().cpu())
            return out

        mem._project = wrapped
    gen = torch.Generator().manual_seed(a.seed)
    seqs = va[torch.randperm(va.shape[0], generator=gen)[:a.n_seq]].to(a.device)
    carry = None
    for i in range(seqs.shape[0]):
        _, carry = model(seqs[i:i + 1], carry=carry, return_carry=True)
    rows = []
    for bi, r in rec.items():
        et, al, th = (torch.cat(r[k]) for k in ("eta", "alpha", "theta"))
        row = {"frozen": path, "step": step_of(path), "block": bi,
               "eta_med": float(et.median()), "eta_frac_off": float((et < 0.5).float().mean()),
               "eta_p10": float(et.quantile(.1)), "eta_p90": float(et.quantile(.9)),
               "alpha_med": float(al.median()), "theta_med": float(th.median())}
        try:
            st = carry[bi][0] if isinstance(carry, (list, tuple)) else None
            if isinstance(st, (list, tuple)):
                st = st[0]
            if isinstance(st, dict) and "w1" in st:
                row["pr_w1"] = pr(st["w1"][0])
        except Exception:
            pass
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True, help="glob")
    ap.add_argument("--val", required=True)
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    va, _ = load_latent_cache(a.val)
    files = sorted(glob.glob(a.frozen), key=step_of)
    with open(a.out, "w") as f:
        for p in files:
            for row in probe(p, va, a):
                f.write(json.dumps(row) + "\n")
                print(f"{p.split('/')[-1]:32s} b{row['block']} eta_med {row['eta_med']:.3g} "
                      f"off {row['eta_frac_off']:.2f} alpha {row['alpha_med']:.2g} "
                      f"theta {row['theta_med']:.2g} pr {row.get('pr_w1', float('nan')):.1f}",
                      flush=True)
            f.flush()
    print("ETA-TRAJ-FINIE", flush=True)


if __name__ == "__main__":
    main()
