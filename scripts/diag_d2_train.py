"""Audit D2 interne : ou naissent NOS NaN d'entrainement ?

Rerun instrumente du config famille (base sane_ns, mirror de mu_rescale),
dose --theta-scale. Chaque pas loggue, DANS L'ORDRE, ce qu'il faut pour
classer une mort :

  p_bad     elements de params non finis AVANT le forward
  loss      perte forward (tbptt_step, meme calcul que l'entrainement reel)
  w1_max    max ||w1|| sur tous les memory.update du pas (espion passif)
  s_max     max ||S|| idem ; n_upd = nombre d'appels update (= maillons NS)
  g_bad/g_n elements de grads non finis APRES backward, AVANT opt.step
            (le train_loop du repo n'a pas d'ecretage)
  pr        ratio de participation du w1 final de fenetre, tous les 25 pas

Signatures (predictions_d2interne.md, enregistre AVANT) :
  D1 : p_bad=0 et w1_max/loss non finis PENDANT le forward.
  D2 : p_bad=0, loss finie, w1_max fini, g_bad>0 ; pas suivant p_bad>0.

Arret : premiere non-finitude + 10 pas (propagation), ou --steps.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from titans.worldmodel.echo import build_parser, build_cfg, prepare_data, double
from titans.worldmodel.model import WorldModel
from titans.worldmodel.training import tbptt_step

EXTRA_AFTER_DEATH = 10
PR_EVERY = 25


def nonfinite_count(tensors):
    n = 0
    for t in tensors:
        if t is not None:
            n += int((~torch.isfinite(t.detach())).sum())
    return n


def pr_of(w) -> float:
    try:
        sv = torch.linalg.svdvals(w.detach().float())
        num = float((sv ** 2).sum()) ** 2
        den = float((sv ** 4).sum())
        return num / den if den > 0 else float("nan")
    except Exception:
        return float("nan")


class UpdateSpy:
    """Enrobe memory.update de chaque bloc : max de normes, sans rien changer."""

    def __init__(self, model):
        self.reset()
        self.handles = []
        for blk in model.blocks:
            if not getattr(blk, "use_memory", False):
                continue
            mem = blk.memory
            orig = mem.update

            def wrapped(*args, _orig=orig, **kw):
                out = _orig(*args, **kw)
                state, S = out
                w1 = state.get("w1")
                if w1 is not None:
                    w = w1.detach()
                    n = float(w.float().norm())
                    if not (n == n):          # nan
                        self.w1_max = float("nan")
                        if self.first_bad_upd is None:
                            self.first_bad_upd = self.n_upd
                    elif not (self.w1_max == self.w1_max):
                        pass                  # deja nan, garder
                    else:
                        self.w1_max = max(self.w1_max, n)
                    self.last_w1 = w[0] if w.dim() > 2 else w
                for v in S.values():
                    if torch.is_tensor(v):
                        n = float(v.detach().float().norm())
                        if not (n == n):
                            self.s_max = float("nan")
                        elif self.s_max == self.s_max:
                            self.s_max = max(self.s_max, n)
                self.n_upd += 1
                return out

            mem.update = wrapped
            self.handles.append((mem, orig))

    def reset(self):
        self.w1_max = 0.0
        self.s_max = 0.0
        self.n_upd = 0
        self.last_w1 = None
        self.first_bad_upd = None


def main():
    ap = build_parser()
    ap.add_argument("--diag-out", dest="diag_out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, space = prepare_data(a.train, a.val)
    T = tr.shape[1]
    K = a.horizon
    tr = tr.to(dev)
    torch.manual_seed(a.seed)
    cfg = build_cfg(a)
    model = WorldModel(latent_dim=tr.shape[-1], dim=space["k"], depth=2,
                       segment_size=a.seg, **cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    spy = UpdateSpy(model)

    B = tr.shape[0]
    bs = a.batch or B
    params = [p for p in model.parameters()]
    death = None
    model.train()
    t0 = time.time()
    with open(a.diag_out, "w") as out:
        out.write(json.dumps({"header": True, "label": a.label, "seed": a.seed,
                              "cfg": {k: v for k, v in cfg.items()
                                      if isinstance(v, (int, float, str, bool, type(None)))},
                              "T": T, "K": K, "batch": bs,
                              "window": getattr(a, "bptt_window", None)}) + "\n")
        for step in range(a.steps):
            p_bad = nonfinite_count(params)
            idx = torch.randperm(B, device=tr.device)[:bs] if bs < B else slice(None)
            batch = double(tr[idx])
            opt.zero_grad()
            spy.reset()
            loss = float(tbptt_step(model, batch, horizon=K,
                                    window=getattr(a, "bptt_window", None)))
            g_bad = nonfinite_count(p.grad for p in params)
            g_n = sum(p.grad.numel() for p in params if p.grad is not None)
            rec = {"step": step, "p_bad": p_bad, "loss": loss,
                   "w1_max": spy.w1_max, "s_max": spy.s_max,
                   "n_upd": spy.n_upd, "first_bad_upd": spy.first_bad_upd,
                   "g_bad": g_bad, "g_n": g_n}
            if step % PR_EVERY == 0 and spy.last_w1 is not None:
                rec["pr"] = pr_of(spy.last_w1)
            out.write(json.dumps(rec) + "\n")
            bad = p_bad > 0 or loss != loss or spy.w1_max != spy.w1_max or g_bad > 0
            if bad and death is None:
                death = step
                out.write(json.dumps({"death": step, "p_bad": p_bad,
                                      "loss_finite": loss == loss,
                                      "w1_finite": spy.w1_max == spy.w1_max,
                                      "g_bad": g_bad}) + "\n")
            opt.step()
            if step % 50 == 0 or bad:
                out.flush()
                print(f"{a.label} step {step} loss {loss:.4f} w1max "
                      f"{spy.w1_max:.3e} gbad {g_bad} pbad {p_bad}", flush=True)
            if death is not None and step >= death + EXTRA_AFTER_DEATH:
                break
        summary = {"summary": True, "label": a.label, "seed": a.seed,
                   "theta_scale": getattr(a, "theta_scale", None),
                   "muon": getattr(a, "muon", None),
                   "death_step": death, "steps_done": step + 1,
                   "n_upd_per_step": spy.n_upd,
                   "elapsed_s": round(time.time() - t0, 1)}
        out.write(json.dumps(summary) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
