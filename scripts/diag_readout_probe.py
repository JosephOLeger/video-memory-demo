"""Sonde A : ou nait la premiere non-finitude du chemin de LECTURE au rejeu ?

Verdict chasse 26/08 : sur le protocole echo, l'ETAT de mu_rescale reste fini
partout, mais les scores sont NaN aux gaps >= 4 blocs (256 latents) avec un pr
qui s'effondre. Le mur du rejeu est donc dans le chemin de prediction, pas
dans l'etat. Cette sonde rejoue EXACTEMENT le flux [V ; D... ; V] de
retention.retention_curve (memes distracteurs torch.roll, meme margin, meme
L1) en guettant, etage par etage, la premiere non-finitude :

  retrieve_hist  sortie de memory.retrieve sur l'etat courant (h_i)
  norm_hist_in / norm_hist_out   la RMSNorm de la retrieval (la division)
  attn           sortie du bloc d'attention
  update         w1 apres memory.update (l'etat lui-meme)
  retrieve_out   la lecture post-update (r_i)
  output         la prediction finale du modele

Chaque evenement enregistre l'etat au moment du premier non-fini : absmax et
finitude de chaque tenseur d'etat, ||w1||, absmax de la requete, rms minimale
d'entree de la norme. Predictions P-A1..P-A5 dans
results/predictions_sondes.md, enregistrees AVANT.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json

import torch

from titans.worldmodel.frozen import load_frozen
from titans.worldmodel.harness import memory_health
from titans.worldmodel.retention import arm_of, spaced
from titans.worldmodel.train import load_latent_cache

MAX_EVENTS = 12          # par (bras, gap) : assez pour localiser, borne la sortie
MAX_ROWS = 200           # trajectoire par segment, batch 0 / bloc 0 seulement


def absmax(t) -> float:
    try:
        return float(t.detach().abs().max())
    except Exception:
        return float("nan")


def finite(t) -> bool:
    return bool(torch.isfinite(t.detach()).all())


class PathSpy:
    def __init__(self, model):
        self.model = model
        self.events = []
        self.rows = []
        self.batch = 0
        self.handles = []
        self.wrapped = []
        self._counters = {}
        for bi, blk in enumerate(model.blocks):
            if not getattr(blk, "use_memory", False):
                continue
            self._wrap_retrieve(bi, blk.memory)
            self._wrap_update(bi, blk.memory)
            if blk.norm_hist is not None:
                self.handles.append(blk.norm_hist.register_forward_pre_hook(
                    self._pre_norm(bi)))
                self.handles.append(blk.norm_hist.register_forward_hook(
                    self._post_norm(bi)))
            self.handles.append(blk.attn.register_forward_hook(
                self._post_attn(bi)))

    # ------------------------------------------------------------------ events
    def _emit(self, stage, bi, seg, **extra):
        if len(self.events) >= MAX_EVENTS:
            return
        self.events.append({"stage": stage, "block": bi, "seg": seg,
                            "batch": self.batch, **extra})

    def _state_stats(self, state):
        st = {}
        for k, w in state.items():
            if torch.is_tensor(w):
                st[k] = {"absmax": absmax(w), "finite": finite(w)}
        return st

    # ----------------------------------------------------------------- wrappers
    def _wrap_retrieve(self, bi, mem):
        orig = mem.retrieve
        key = ("ret", bi)
        self._counters[key] = 0

        def wrapped(state, q, _orig=orig, _bi=bi, _key=key):
            idx = self._counters[_key]
            self._counters[_key] += 1
            seg = idx // 2
            kind = "retrieve_hist" if idx % 2 == 0 else "retrieve_out"
            out = _orig(state, q)
            ok = finite(out)
            if not ok:
                w1 = state.get("w1")
                self._emit(kind, _bi, seg,
                           q_absmax=absmax(q), q_finite=finite(q),
                           out_bad=int((~torch.isfinite(out.detach())).sum()),
                           w1_fro=(float(w1.detach().float().norm())
                                   if w1 is not None else None),
                           state=self._state_stats(state))
            if self.batch == 0 and _bi == 0 and len(self.rows) < MAX_ROWS:
                if kind == "retrieve_hist":
                    self.rows.append({"seg": seg, "h_absmax": absmax(out)})
                elif self.rows and self.rows[-1].get("seg") == seg:
                    self.rows[-1]["r_absmax"] = absmax(out)
            return out

        mem.retrieve = wrapped
        self.wrapped.append((mem, "retrieve", orig))

    def _wrap_update(self, bi, mem):
        orig = mem.update

        def wrapped(*args, _orig=orig, _bi=bi, **kw):
            out = _orig(*args, **kw)
            state, S = out
            w1 = state.get("w1")
            seg = max(0, self._counters.get(("ret", _bi), 1) - 1) // 2
            if w1 is not None:
                ok = finite(w1)
                if not ok:
                    self._emit("update", _bi, seg,
                               state=self._state_stats(state))
                if self.batch == 0 and _bi == 0 and self.rows and \
                        self.rows[-1].get("seg") == seg:
                    self.rows[-1]["w1_absmax"] = absmax(w1)
            return out

        mem.update = wrapped
        self.wrapped.append((mem, "update", orig))

    def _pre_norm(self, bi):
        def hook(mod, inputs, _bi=bi):
            h = inputs[0]
            seg = max(0, self._counters.get(("ret", _bi), 1) - 1) // 2
            rms = h.detach().float().pow(2).mean(-1).sqrt()
            if not finite(h):
                self._emit("norm_hist_in", _bi, seg, in_absmax=absmax(h))
            if self.batch == 0 and _bi == 0 and self.rows and \
                    self.rows[-1].get("seg") == seg:
                self.rows[-1]["rms_min"] = float(rms.min())
        return hook

    def _post_norm(self, bi):
        def hook(mod, inputs, output, _bi=bi):
            seg = max(0, self._counters.get(("ret", _bi), 1) - 1) // 2
            if not finite(output):
                self._emit("norm_hist_out", _bi, seg,
                           in_finite=finite(inputs[0]),
                           in_absmax=absmax(inputs[0]),
                           in_rms_min=float(inputs[0].detach().float()
                                            .pow(2).mean(-1).sqrt().min()))
        return hook

    def _post_attn(self, bi):
        def hook(mod, inputs, output, _bi=bi):
            seg = max(0, self._counters.get(("ret", _bi), 1) - 1) // 2
            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out) and not finite(out):
                self._emit("attn", _bi, seg, in_finite=finite(inputs[0]))
        return hook

    # ---------------------------------------------------------------- lifecycle
    def new_pass(self, batch_idx):
        self.batch = batch_idx
        for k in self._counters:
            self._counters[k] = 0

    def close(self):
        for h in self.handles:
            h.remove()
        for obj, name, orig in self.wrapped:
            setattr(obj, name, orig)


@torch.no_grad()
def probe_arm_gap(model, va, T, K, n, batch, margin):
    """Rejoue [V ; D...D ; V] au gap n blocs, instrumente, score le L1 final."""
    distr = [torch.roll(va, k, 0) for k in range(2, n + 2)]
    rep = spaced(va, va, distr)
    for b in model.blocks:
        b.reset_memory_each_segment = False
    spy = PathSpy(model)
    tot = num = 0.0
    pred_bad = 0
    N = rep.shape[1]
    lo, hi = N - T + margin, N - K
    try:
        for bi_, i in enumerate(range(0, rep.shape[0], batch)):
            z = rep[i:i + batch]
            spy.new_pass(bi_)
            pred = model(z)
            if not finite(pred):
                pred_bad += 1
                spy._emit("output", -1, -1,
                          bad=int((~torch.isfinite(pred.detach())).sum()))
            err = (pred[:, :-K] - z[:, K:]).abs().mean(-1)
            e = err[:, lo:hi]
            tot += e.sum().item()
            num += e.numel()
    finally:
        spy.close()
    try:
        h = memory_health(model, rep[:4])
    except Exception:
        h = {"pr": float("nan"), "norm": float("nan"),
             "cos_ret": float("nan"), "finite": False}
    return {"gap_blocks": int(n), "gap_latents": int(n * T),
            "l1_repeat": tot / num if num else float("nan"),
            "pred_bad_batches": pred_bad,
            "events": spy.events, "rows": spy.rows,
            "pr": h["pr"], "w1_norm": h["norm"], "cos_ret": h["cos_ret"],
            "health_finite": bool(h["finite"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=96, dest="n_eval")
    ap.add_argument("--gaps", default="0,2,4,8")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--diag-out", dest="diag_out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    va, meta = load_latent_cache(a.val)
    T = va.shape[1]
    va = va[:a.n_eval].to(a.device)
    gaps = [int(x) for x in a.gaps.split(",")]

    with open(a.diag_out, "w") as out:
        out.write(json.dumps({"header": True, "frozen": a.frozen,
                              "gaps": gaps, "n_eval": a.n_eval,
                              "batch": a.batch, "T": T,
                              "horizon": a.horizon}) + "\n")
        for p in sorted(globmod.glob(a.frozen)) or [a.frozen]:
            model, info = load_frozen(p, a.device)
            model.eval()
            for n in gaps:
                if n + 2 > va.shape[0]:
                    continue
                rec = probe_arm_gap(model, va, T, a.horizon, n,
                                    a.batch, a.margin)
                rec.update({"arm": arm_of(p), "file": p})
                out.write(json.dumps(rec) + "\n")
                out.flush()
                ev = rec["events"][0]["stage"] if rec["events"] else "-"
                print(f"{rec['arm']} gap {n}: l1 {rec['l1_repeat']:.4f} "
                      f"pr {rec['pr']:.1f} premier_evenement {ev} "
                      f"({len(rec['events'])} evts)", flush=True)
        out.write(json.dumps({"done": True}) + "\n")
    print("SONDE-A-FINIE", flush=True)


if __name__ == "__main__":
    main()
