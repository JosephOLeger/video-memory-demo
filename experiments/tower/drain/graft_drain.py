"""Graft isolated recall: momentum drain + projections x gates factorial (rebuttal A, 15/10).

Two reviewer controls on the graft memories (train_load checkpoints, no host needed):
  * DRAIN: after writing N latents, continue with zero gradient, the items' median
    eta/theta and alpha = 0 for D = min(ceil(5/(1-eta)), dmax) steps, then query.
  * FACTORIAL (same seed): memory with the projections of arm P and the gate heads
    (alpha, eta, theta) of arm G, for P, G in {A, B}. Separates "the learned gates cut
    writing" from "the projections lost the content".
Queries by key k_proj(x_i), top-1 among N by cosine, as graft_echo.py.

    PYTHONPATH=src python graft_drain.py --bank ~/work/iclr/load/bank.pt --u2 ~/work/iclr/u2 \
        --out ~/work/iclr/drain/graft_drain.jsonl --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import math

import torch

from titans.graft.branch import memory_kwargs_from_args, sane_memory_kwargs
from titans.memory import NeuralMemory

GATES = ("alpha_head", "eta_head", "theta_head")


def pr_of_w(w):
    s2 = torch.linalg.svdvals(w.double()) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def hits(r, v):
    rn = r / (r.norm(dim=-1, keepdim=True) + 1e-9)
    vn = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
    C = rn @ vn.T
    return (C.argmax(1) == torch.arange(C.shape[0], device=C.device)).float()


def load_mem(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    args = blob["args"]
    mem = NeuralMemory(dim=args.get("d_mem", 128),
                       **sane_memory_kwargs(**memory_kwargs_from_args(args)))
    sd = {k[len("memory."):]: v for k, v in blob["branch"].items() if k.startswith("memory.")}
    _, unexpected = mem.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return mem.to(device).eval(), args.get("chunk", 8)


def graft_gates(proj_mem, gate_mem):
    m = copy.deepcopy(proj_mem)
    for g in GATES:
        getattr(m, g).load_state_dict(getattr(gate_mem, g).state_dict())
    return m


@torch.no_grad()
def run(mem, chunk, X, dmax):
    v = mem.v_proj(X)[0]
    kq = mem.k_proj(X)
    st, S = mem.init_memory(1)
    st, S = mem.update(X, st, S, chunk)
    h0 = hits(mem.retrieve(st, kq)[0], v)
    pr0 = pr_of_w(st["w1"][0])
    _, _, _, alpha, eta, theta = mem._project(X)
    eta_m, th_m = float(eta.median()), theta.median().reshape(1)
    D = min(int(math.ceil(5.0 / max(1.0 - eta_m, 1e-6))), dmax)
    za, et = torch.zeros(1, device=X.device), torch.full((1,), eta_m, device=X.device)
    for _ in range(D):
        st, S = mem._step(st, S, {k: torch.zeros_like(w) for k, w in st.items()}, za, et, th_m)
    h1 = hits(mem.retrieve(st, kq)[0], v)
    q = lambda h: [float(h[(i * len(h)) // 4:((i + 1) * len(h)) // 4].mean()) for i in range(4)]  # noqa: E731
    return {"none": float(h0.mean()), "drain": float(h1.mean()), "none_quart": q(h0),
            "drain_quart": q(h1), "eta_med": eta_m, "theta_med": float(th_m),
            "prod": float(th_m) / max(1 - eta_m, 1e-9), "D": D, "pr": pr0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--u2", required=True)
    ap.add_argument("--arms", default="A,B,C,E3_d1_l10,FAR_D32_l10,H_d32_l10,PN_D32_l10,P1_l10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--loads", default="1024,4096")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--dmax", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(16)
    ev = torch.load(a.bank, map_location="cpu", weights_only=False)["eval"]["fillers"]
    pool = torch.cat([f.float() for f in ev], 0)
    loads = [int(x) for x in a.loads.split(",")]
    arms = a.arms.split(",")
    with open(a.out, "a") as fo:
        for s in a.seeds.split(","):
            mems = {arm: load_mem(f"{a.u2}/{arm}_s{s}_at2500.pt", a.device) for arm in arms}
            confs = [(arm, arm, arm) for arm in arms]
            confs += [("projA_gatesB", "A", "B"), ("projB_gatesA", "B", "A")]
            for name, P, G in confs:
                m = mems[P][0] if P == G else graft_gates(mems[P][0], mems[G][0])
                chunk = mems[P][1]
                for N in loads:
                    rs = []
                    for dr in range(a.draws):
                        g = torch.Generator().manual_seed(1000 * dr + N)
                        X = pool[torch.randperm(pool.shape[0], generator=g)[:N]].unsqueeze(0).to(a.device)
                        rs.append(run(m, chunk, X, a.dmax))
                    row = {"conf": name, "seed": int(s), "N": N}
                    for k in rs[0]:
                        vals = [r[k] for r in rs]
                        row[k] = ([round(sum(c) / len(c), 4) for c in zip(*vals)]
                                  if isinstance(vals[0], list) else round(sum(vals) / len(vals), 4))
                    fo.write(json.dumps(row) + "\n"); fo.flush()
                    print(f"s{s} {name:18s} N={N:5d} none={row['none']:.3f} drain={row['drain']:.3f} "
                          f"D={row['D']:.0f} prod={row['prod']:.4f} pr={row['pr']:.1f} "
                          f"q_none={row['none_quart']} q_drain={row['drain_quart']}", flush=True)
    print("GRAFT-DRAIN-FINI", flush=True)


if __name__ == "__main__":
    main()
