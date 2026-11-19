"""Temoins a budget d'octets egal pour la greffe en flux couleur (22/10).

On ne modifie RIEN dans ~/work/titans-mac/src : ce module remplace, dans l'espace de noms de
``titans.graft.train_stream``, la classe ``MemoryBranch``, ``write_stream`` et ``pr``, puis appelle
``train_stream.main()`` tel quel. Recette, donnees, ordre des lots (rng = Random(seed)), eval :
le code de C1_color_s0, a l'identique. Seule la memoire change.

Memoire profonde de reference (C1_color_s0) : d_mem 128, memory_hidden 64, update_bias=False
-> contenu ecrit = w0 [64x128] + w1 [128x64] = 16 384 flottants (64 Kio fp32) ; + l'elan S
de meme taille (non compte ici, ce qui desavantage les temoins).

Temoins (meme down PCA gele, meme up a zero, meme norm_r, meme q_proj(z) sur chaque token hote) :
  linear : M [128x128] = 16 384 flottants, regle delta (DeltaNet) :
           M <- (1-a_t) M + b_t (v_t - M k_t) k_t^T, ||k||=1 ; lecture M q/||q||.
  pool   : 128 cases de 128 = 16 384 flottants ; chaque case = moyenne de P ecritures
           consecutives (P=3 : 360 ecritures -> 120 cases, tout le passe tient) ; lecture
           softmax(tau q.k) sur les cases, k = k_proj(case), v = v_proj(case).
  fifo   : les 128 dernieres ecritures telles quelles (16 384 flottants), meme lecture.
  mean   : une seule moyenne de toutes les ecritures (128 flottants, 128x MOINS), lecture
           v_proj(moyenne), independante de la requete. Borne basse.

    PYTHONPATH=src:paper_b/text python -m slot_memory --mem-type linear \
        <arguments de train_stream>
"""
from __future__ import annotations

import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import titans.graft.train_stream as ts
from titans.graft.branch import MemoryBranch as _DeepBranch

CFG = {"mem_type": "deep", "slots": 128, "pool": 3, "tau_init": 10.0, "beta_init": 0.5}


def _l2(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


class LinearDeltaMemory(nn.Module):
    """Memoire lineaire a regle delta, meme interface que NeuralMemory pour la greffe."""

    def __init__(self, dim, alpha_init, beta_init):
        super().__init__()
        self.dim = dim
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_head = nn.Linear(dim, 1)          # oubli -> sigmoid (meme init que la profonde)
        self.beta_head = nn.Linear(dim, 1)           # pas d'ecriture -> sigmoid, borne (0,1)
        with torch.no_grad():
            self.alpha_head.weight.zero_(); self.beta_head.weight.zero_()
            self.alpha_head.bias.fill_(math.log(alpha_init / (1 - alpha_init)))
            self.beta_head.bias.fill_(math.log(beta_init / (1 - beta_init)))

    def init_memory(self, B):
        dev = self.k_proj.weight.device
        return {"M": torch.zeros(B, self.dim, self.dim, device=dev)}, {}

    def update(self, *a, **k):
        raise RuntimeError("ecriture hors write() interdite (la greffe lit sans ecrire)")

    def retrieve(self, state, q):
        return torch.einsum("bij,bnj->bni", state["M"], _l2(q))

    def write(self, z):
        k, v = _l2(self.k_proj(z)), self.v_proj(z)
        a = torch.sigmoid(self.alpha_head(z)).squeeze(-1)
        b = torch.sigmoid(self.beta_head(z)).squeeze(-1)
        M = self.init_memory(z.shape[0])[0]["M"]
        for t in range(z.shape[1]):
            kt, vt = k[:, t], v[:, t]
            err = vt - torch.einsum("bij,bj->bi", M, kt)
            M = (1 - a[:, t, None, None]) * M + b[:, t, None, None] * err[:, :, None] * kt[:, None, :]
        return {"M": M}, {}, (a.detach(), torch.zeros_like(a), b.detach())


class SlotMemory(nn.Module):
    """Tampon de cases (FIFO brut, moyennes par P, ou moyenne globale), lu par attention."""

    def __init__(self, dim, slots, pool, tau_init):
        super().__init__()
        self.dim, self.slots, self.pool = dim, slots, pool
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    def init_memory(self, B):
        dev = self.k_proj.weight.device
        return {"slots": torch.zeros(B, 0, self.dim, device=dev)}, {}

    def update(self, *a, **k):
        raise RuntimeError("ecriture hors write() interdite (la greffe lit sans ecrire)")

    def write(self, z):
        B, N, d = z.shape
        if self.pool == 0:                                   # moyenne globale
            s = z.mean(1, keepdim=True)
        else:
            P = self.pool
            G = math.ceil(N / P)
            s = torch.stack([z[:, g * P:(g + 1) * P].mean(1) for g in range(G)], 1) if P > 1 else z
            s = s[:, -self.slots:]                           # FIFO sur les cases
        dummy = torch.zeros(B, N, device=z.device)
        return {"slots": s}, {}, (dummy, dummy, dummy)

    def retrieve(self, state, q):
        s = state["slots"]
        if s.shape[1] == 0:
            return torch.zeros(*q.shape[:-1], self.dim, device=q.device, dtype=q.dtype)
        vals = self.v_proj(s)                                # [B, K, d]
        if s.shape[1] == 1:                                  # une case : lecture constante
            return vals.expand(q.shape[0], q.shape[1], -1)
        keys = _l2(self.k_proj(s))
        att = torch.softmax(self.log_tau.exp() * torch.einsum("bnd,bkd->bnk", _l2(q), keys), -1)
        return torch.einsum("bnk,bkd->bnd", att, vals)


class BaselineBranch(_DeepBranch):
    """MemoryBranch dont la memoire est remplacee ; down / up / norm_r / delta inchanges."""

    def __init__(self, d_model, d_mem, **kw):
        super().__init__(d_model, d_mem, **kw)
        t = CFG["mem_type"]
        if t == "linear":
            self.memory = LinearDeltaMemory(d_mem, kw.get("alpha_init", 1.27e-4), CFG["beta_init"])
        elif t == "pool":
            self.memory = SlotMemory(d_mem, CFG["slots"], CFG["pool"], CFG["tau_init"])
        elif t == "fifo":
            self.memory = SlotMemory(d_mem, CFG["slots"], 1, CFG["tau_init"])
        elif t == "mean":
            self.memory = SlotMemory(d_mem, 1, 0, CFG["tau_init"])
        else:
            raise ValueError(t)

    def stored_floats(self, st):
        return sum(v[0].numel() for v in st.values())


_orig_write = ts.write_stream


def write_stream(branch, z, chunk, *a, **k):
    if isinstance(branch, BaselineBranch):
        st, S, g = branch.memory.write(z)
        return st, S, None, g
    return _orig_write(branch, z, chunk, *a, **k)


def main():
    argv = sys.argv[1:]
    pre = {"--mem-type": "mem_type", "--slots": "slots", "--pool": "pool", "--tau-init": "tau_init",
           "--beta-init": "beta_init"}
    rest, i = [], 0
    while i < len(argv):
        if argv[i] in pre:
            key = pre[argv[i]]
            CFG[key] = argv[i + 1] if key == "mem_type" else type(CFG[key])(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i]); i += 1
    sys.argv = [sys.argv[0]] + rest
    out = rest[rest.index("--out") + 1]
    with open(out + ".bl.json", "w") as f:
        json.dump(CFG, f)
    print("TEMOIN", CFG, flush=True)
    if CFG["mem_type"] != "deep":
        ts.MemoryBranch = BaselineBranch
        ts.write_stream = write_stream
        # train_stream appelle pr(st["w1"][0]) : on lui donne la matrice stockee du temoin
        _orig_main_pr = ts.pr
        ts.pr = lambda w: _orig_main_pr(w) if w.shape[0] > 1 else 0.0

        class _St(dict):
            def __getitem__(self, k):
                if k == "w1" and k not in self:
                    w = self.get("M", self.get("slots"))
                    if w.shape[1] < 2:                       # moyenne : rien a mesurer
                        return torch.zeros(1, 1, 1, device=w.device)
                    return w
                return dict.__getitem__(self, k)

        def ws(branch, z, chunk, *a, **k):
            st, S, ld, g = write_stream(branch, z, chunk, *a, **k)
            return _St(st), S, ld, g

        ts.write_stream = ws
    ts.main()


if __name__ == "__main__":
    main()
