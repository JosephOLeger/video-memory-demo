"""Echelle Titans-LITTERALE : l'objet publie, sans NS, sans rescale.

Config exacte testee : base sane (memes portes theta/eta/alpha, memes
normalisations qk), memory_orthogonalize_momentum=False -> la mise a jour
interne est M_t = (1-alpha)M_{t-1} + S_t, S_t = eta S_{t-1} - theta grad
(magnitude = gradient BRUT, direction = gradient brut). C'est Titans tel
qu'ecrit (Eq 2/5 du papier), la convention non bornee PUBLIEE -- aucun
hybride maison. Harnais identique a diag_d2b_train.py (espion passif +
trace par chunk pour la figure mecanisme). Predictions AVANT :
results/predictions_sane_reanchor.md.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from titans.stability import trace_forward
from titans.worldmodel.echo import build_parser, build_cfg, prepare_data, double
from titans.worldmodel.margin import memory_input_stream
from titans.worldmodel.model import WorldModel
from titans.worldmodel.training import tbptt_step

EXTRA_AFTER_DEATH = 10
PR_EVERY = 25
TRACE_KEYS = ("chunk", "token", "m", "alpha", "eta", "theta", "kappa",
              "lambda_keys_w", "lambda_keys_u", "lambda_hidden_w",
              "lambda_hidden_u", "h0_norm_max", "h0_norm_mean",
              "theta_lambda_hidden", "rho_hidden", "rho_keys",
              "w0_spec", "w1_spec", "w1_fro", "b0_norm", "S_norm",
              "grad_norm_max", "finite", "note")


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
    """Identique a diag_d2_train.py : max de normes, sans rien changer."""

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
                    if not (n == n):
                        self.w1_max = float("nan")
                        if self.first_bad_upd is None:
                            self.first_bad_upd = self.n_upd
                    elif not (self.w1_max == self.w1_max):
                        pass
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


@torch.no_grad()
def step_trace(model, batch):
    """Trace par chunk du batch courant, a l'etat courant des parametres."""
    was_training = model.training
    model.eval()
    try:
        blk = model.blocks[0]
        x = memory_input_stream(model, batch)
        rows = trace_forward(blk.memory, x, blk.memory_chunk_size, b=0)
        return [{k: r[k] for k in TRACE_KEYS if k in r} for r in rows]
    finally:
        if was_training:
            model.train()


def main():
    ap = build_parser()
    ap.add_argument("--diag-out", dest="diag_out", required=True)
    ap.add_argument("--trace-every", dest="trace_every", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, space = prepare_data(a.train, a.val)
    T = tr.shape[1]
    K = a.horizon
    tr = tr.to(dev)
    torch.manual_seed(a.seed)
    cfg = build_cfg(a)
    cfg["memory_orthogonalize_momentum"] = False   # Titans-litteral : NS coupe
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
                              "trace_every": a.trace_every}) + "\n")
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
            if step % a.trace_every == 0 or bad:
                try:
                    out.write(json.dumps(
                        {"step": step, "trace": step_trace(model, batch)}) + "\n")
                except Exception as e:
                    out.write(json.dumps(
                        {"step": step, "trace_error": repr(e)[:200]}) + "\n")
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
                   "death_step": death, "steps_done": step + 1,
                   "elapsed_s": round(time.time() - t0, 1)}
        out.write(json.dumps(summary) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
