"""Distillation a contexte complet, ponderee : la memoire remplace le contexte retire.

Cibles : ``distill_targets`` (p_complet top-k, p_court, poids w_t, clips du passe).
Eleve : Qwen gele sur la fenetre (dernier clip) + la memoire, ecrite avec TOUT le passe.

    L = sum_t w_t KL(p_complet,t || p_eleve,t) / sum_t w_t        (sur le batch)

Mesures a l'eval (memes echantillons a chaque fois, videos tenues a l'ecart) :
  plancher  = KL ponderee de p_court (l'eleve sans memoire EST p_court : up = 0)
  memoire   = KL ponderee de l'eleve
  mauvaise  = KL ponderee de l'eleve lisant la memoire d'une AUTRE video
  recupere  = (plancher - memoire) / plancher        -> part de l'ecart comblee
  contenu   = (mauvaise - memoire) / plancher        -> part due au BON contenu
              ([[distillation-gain-is-mostly-not-content]] : 7 % la derniere fois)
  + recupere par type de question et, pour les scenes, par distance a la fenetre
    (la portee de la memoire, en secondes).
Portes (eta, alpha, theta) et pr(W1) loguees : on surveille la derive "cache".

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m titans.graft.train_distill \\
        --targets '~/work/iclr/distill/targets_*.pt' --basis cache/qwen_l24_pca128.pt \\
        --muon unit --steps 8000 --out ~/work/iclr/distill/unit_s0
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .distill import forced_inputs
from .distill_targets import gather_lp, kl_buckets
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_load import pr, take, write_stream

DIST_BINS = [(0, 60), (60, 180), (180, 400), (400, 10 ** 6)]


def load_targets(pattern: str):
    vids, samples = {}, []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        vids.update(d["videos"])
        samples += d["samples"]
    # longueur de clip commune (formats d'image differents) : on tronque au mode,
    # on ecarte les videos plus courtes (le batch empile des passes de meme longueur)
    L = Counter(int(c.shape[0]) for v in vids.values() for c in v["clips"]).most_common(1)[0][0]
    keep = {k for k, v in vids.items() if all(c.shape[0] >= L for c in v["clips"])}
    for k in keep:
        vids[k]["clips"] = [c[:L] for c in vids[k]["clips"]]
    samples = [s for s in samples if s["vid"] in keep and float(s["w"].sum()) > 0]
    return vids, samples, L


def past_z(vids, s, device):
    v = vids[s["vid"]]
    return torch.cat(v["clips"][:s["end"] - 1], 0).float().to(device)


def student_kl(model, processor, wrapper, vids, s, state, S, device):
    """KL par token ``[T]`` de l'eleve lisant ``(state, S)``."""
    v = vids[s["vid"]]
    win = v["windows"][s["end"]].float()
    y = s["y"].long()
    inp, n_prompt = forced_inputs(processor, win, s["q"], y, v["fps"], device)
    wrapper.carry, wrapper.token_offset = (state, S), 0
    lg = model(**inp, use_cache=False).logits[0, n_prompt - 1: n_prompt - 1 + y.shape[0]]
    lq = gather_lp(lg, s["idx"].long().to(device))
    return kl_buckets(s["lp_full"].float().to(device), lq)


def batches_by_end(samples, B, rng):
    by = defaultdict(list)
    for s in samples:
        by[s["end"]].append(s)
    ends = list(by)
    weights = [len(by[e]) for e in ends]
    while True:
        e = rng.choices(ends, weights)[0]
        yield rng.sample(by[e], min(B, len(by[e])))


@torch.no_grad()
def evaluate(model, processor, branch, wrapper, vids, ev, a):
    branch.eval()
    by = defaultdict(list)
    for s in ev:
        by[s["end"]].append(s)
    rows = []
    for e, ss in sorted(by.items()):
        for b0 in range(0, len(ss), a.batch):
            bs = ss[b0:b0 + a.batch]
            if len(bs) < 2:
                continue
            z = torch.stack([past_z(vids, s, a.device) for s in bs])
            state, S, _, _ = write_stream(branch, z, a.chunk)
            n = len(bs)
            for j, s in enumerate(bs):
                w = s["w"].to(a.device)
                kl_m = student_kl(model, processor, wrapper, vids, s, take(state, j),
                                  take(S, j), a.device)
                # mauvaise memoire : celle d'un AUTRE echantillon d'une autre video
                jj = next((i for i in range(1, n) if bs[(j + i) % n]["vid"] != s["vid"]), None)
                kl_w = (student_kl(model, processor, wrapper, vids, s,
                                   take(state, (j + jj) % n), take(S, (j + jj) % n), a.device)
                        if jj is not None else None)
                rows.append({"kind": s["kind"], "dist": s["dist_s"], "W": float(w.sum()),
                             "floor": float((w * w).sum()), "mem": float((w * kl_m).sum()),
                             "wrong": float((w * kl_w).sum()) if kl_w is not None else None,
                             "kl_mem_u": float(kl_m.mean())})
    branch.train()

    def agg(rs):
        if not rs:
            return None
        W = sum(r["W"] for r in rs)
        fl, me = sum(r["floor"] for r in rs) / W, sum(r["mem"] for r in rs) / W
        rw = [r for r in rs if r["wrong"] is not None]
        Ww = sum(r["W"] for r in rw)
        wr = sum(r["wrong"] for r in rw) / Ww if Ww else float("nan")
        me_w = sum(r["mem"] for r in rw) / Ww if Ww else float("nan")
        return {"n": len(rs), "floor": fl, "mem": me, "wrong": wr,
                "recovered": (fl - me) / fl if fl else float("nan"),
                "content": (wr - me_w) / fl if fl else float("nan")}

    out = {"all": agg(rows)}
    for k in ("scene", "generic", "qa"):
        out[k] = agg([r for r in rows if r["kind"] == k])
    for lo, hi in DIST_BINS:
        out[f"scene_{lo}-{hi}s"] = agg([r for r in rows if r["kind"] == "scene"
                                        and lo <= r["dist"] < hi])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--targets", required=True, help="glob des shards")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--memory-hidden", type=int, default=64, dest="memory_hidden")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=8, dest="write_stride")
    ap.add_argument("--alpha-init", type=float, default=1.27e-4, dest="alpha_init")
    ap.add_argument("--muon", default="unit")
    ap.add_argument("--muon-scale", type=float, default=None, dest="muon_scale")
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    ap.add_argument("--eval-n", type=int, default=320, dest="eval_n")
    ap.add_argument("--save-every", type=int, default=1000, dest="save_every")
    ap.add_argument("--no-weight", action="store_true", dest="no_weight",
                    help="ablation : KL douce non ponderee (w_t = 1) a l'entrainement ;"
                         " l'eval reste ponderee")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    vids, samples, L = load_targets(a.targets)
    tr = [s for s in samples if not s["eval"]]
    ev_all = [s for s in samples if s["eval"]]
    ev = random.Random(1234).sample(ev_all, min(a.eval_n, len(ev_all)))
    print(f"{len(vids)} videos, {len(tr)} echantillons d'entrainement, {len(ev)} d'eval "
          f"(sur {len(ev_all)}), {L} ecritures/clip", flush=True)
    # biais de recence du professeur ? poids moyen par distance (scenes)
    for lo, hi in DIST_BINS:
        ws = [float(s["w"].sum()) for s in tr if s["kind"] == "scene" and lo <= s["dist_s"] < hi]
        if ws:
            print(f"  poids scene {lo:>4}-{hi}s : n {len(ws):5d}  sum_w moyen {sum(ws) / len(ws):.2f}",
                  flush=True)

    from transformers import AutoProcessor

    from .check_qwen import load_host
    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    mem_kw = memory_kwargs_from_args(a)
    branch = MemoryBranch(int(model.config.text_config.hidden_size), a.d_mem,
                          chunk_size=a.chunk, write_stride=a.write_stride,
                          down_pca=pca, **mem_kw).to(a.device)
    branch.train()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    print(f"greffe couche {site} : {sum(p.numel() for p in params)} params, "
          f"muon={mem_kw['orthogonalize_momentum']}", flush=True)

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    hist, gate_log, t0, run = [], [], time.time(), None
    gen = batches_by_end(tr, a.batch, rng)
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        bs = next(gen)
        try:
            z = torch.stack([past_z(vids, s, a.device) for s in bs])
            state, S, _, (al, et, th) = write_stream(branch, z, a.chunk)
            num, den = 0.0, 0.0
            den_w = 0.0
            for j, s in enumerate(bs):
                w = s["w"].to(a.device)
                kl = student_kl(model, processor, wrapper, vids, s, take(state, j),
                                take(S, j), a.device)
                wt = torch.ones_like(w) if a.no_weight else w
                num = num + (wt * kl).sum()
                den += float(wt.sum())
                den_w += float(w.sum())
            loss = num / den
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
        opt.step()
        floor = (sum(float(s["w"].mean()) for s in bs) / len(bs) if a.no_weight
                 else sum(float((s["w"] ** 2).sum()) for s in bs) / den)
        rec = 1 - float(loss) / floor
        run = rec if run is None else 0.95 * run + 0.05 * rec
        if step % 10 == 0:
            gl = {"step": step, "eta_med": float(et.median()),
                  "eta_frac_off": float((et < 0.5).float().mean()),
                  "alpha_med": float(al.median()), "theta_med": float(th.median()),
                  "pr_w1": pr(state["w1"][0].detach()), "writes": int(z.shape[1]),
                  "loss": float(loss), "floor": floor, "rec_run": run}
            gate_log.append(gl)
            print(f"  {step:5d} loss {float(loss):.4f} plancher {floor:.4f} recupere~ {100 * run:+.1f}% "
                  f"|g| {gn:.2f} eta {gl['eta_med']:.3f} alpha {gl['alpha_med']:.1e} "
                  f"theta {gl['theta_med']:.3f} pr {gl['pr_w1']:.1f} ecr {gl['writes']} "
                  f"{(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)
        last = step + 1 == a.steps
        if a.save_every and ((step + 1) % a.save_every == 0 or last):
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a)},
                       f"{a.out}_at{step + 1}.pt")
        if (step + 1) % a.eval_every == 0 or last or step == 0:
            r = evaluate(model, processor, branch, wrapper, vids, ev, a)
            hist.append({"step": step + 1, **r})
            al_ = r["all"]
            print(f"  == eval {step + 1}: plancher {al_['floor']:.4f} memoire {al_['mem']:.4f} "
                  f"mauvaise {al_['wrong']:.4f} | recupere {100 * al_['recovered']:+.1f}% "
                  f"contenu {100 * al_['content']:+.1f}%  || " + "  ".join(
                      f"{k} {100 * v['recovered']:+.0f}/{100 * v['content']:+.0f}"
                      for k, v in r.items() if k != "all" and v), flush=True)
        if step % 50 == 0 or last:
            with open(a.out + ".json", "w") as f:
                json.dump({"args": vars(a), "history": hist, "gates": gate_log,
                           "elapsed_s": time.time() - t0}, f, indent=1)
    print("TRAIN-DISTILL-FINIE", flush=True)


if __name__ == "__main__":
    main()
