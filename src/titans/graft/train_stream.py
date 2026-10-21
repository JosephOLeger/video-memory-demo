"""Greffe en regime FLUX : la memoire retient ce qui sera demande, Qwen apprend a le lire.

Cibles : ``stream_targets`` (1 image/s, fenetre = un clip de 16 s, memoire = clips evinces).
Pour chaque question, l'eleve est Qwen gele sur la FENETRE seule + la memoire ecrite avec
les clips ``0..c-1`` (le clip ``p`` remplace par sa version avec la forme, pour ``shape``),
reponse forcee. Perte, specifique a la question :

    L = lam_ce  * CE(tokens de la reponse)
      + lam_lat * ||h_eleve - h_ens||^2 / ||h_fenetre - h_ens||^2, etats de derniere couche aux
                  SEULES positions de la reponse (l'enseignant voit le clip de preuve ; 1 = la
                  memoire n'apporte rien, 0 = l'eleve rejoint l'enseignant)
      + lam_con * max(0, marge + L_lat(bonne memoire) - L_lat(memoire d'une autre video))

Le terme latent ne copie pas l'etat local de Qwen sur la fenetre : seulement ce qui porte la
reponse. Le terme contrastif rend non rentable l'a priori sans contenu (le QCM du 14/10 l'a
montre : sans lui la branche apprend l'a priori et ignore le passe).

Eval (videos tenues a l'ecart) : precision de la lettre avec la memoire, avec la memoire d'une
autre video (part du contenu = difference), fenetre seule et enseignant (precalcules) ; par
type, par distance, et sous-ensemble « dur » (la fenetre seule se trompe).

    PYTHONPATH=src python -m titans.graft.train_stream --targets '~/work/iclr/stream/tg/*.pt' \\
        --basis cache/qwen_l24_pca128.pt --out run
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from .branch import MemoryBranch, fold_pca, memory_kwargs_from_args
from .distill import forced_inputs
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_load import pr, write_stream
from .write_ckpt import write_stream_ckpt

LET = "ABCD"


def load(pattern, basis, max_past):
    W, b = fold_pca(torch.load(basis, map_location="cpu", weights_only=False)["pca"])
    W, b = W.float(), b.float()
    proj = lambda h: (h.float() @ W.T + b).half()               # noqa: E731
    vids, qs = {}, []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = d["vid"]
        vids[v] = {"clips": [proj(c) for c in d["clips"]], "windows": d["windows"],
                   "fps": d["fps"], "eval": d["eval"]}
        for i, q in enumerate(d["qs"]):
            if q["c"] - q["p"] > max_past:
                continue
            q = dict(q, vid=v, qid=f"{v}#{i}", eval=d["eval"],
                     patch=None if q["patch"] is None else proj(q["patch"]),
                     teacher_ok=int(q["lt_teacher"].argmax()) == q["answer"],
                     window_ok=int(q["lt_window"].argmax()) == q["answer"],
                     blind_ok=("lt_blind" in q and int(q["lt_blind"].argmax()) == q["answer"]))
            qs.append(q)
    return vids, qs


def load_variants(pattern, basis):
    """Variantes de couleur (stream_color_variants) : qid -> liste de {prompt, y, answer, patch}."""
    W, b = fold_pca(torch.load(basis, map_location="cpu", weights_only=False)["pca"])
    W, b = W.float(), b.float()
    out = {}
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for i, rows in d["variants"].items():
            out[f"{d['vid']}#{i}"] = [{"prompt": r["prompt"], "y": r["y"], "answer": r["answer"],
                                       "patch": (r["patch"].float() @ W.T + b).half(), "variant": r["color"]}
                                      for r in rows]
    return out


def past(vids, q, max_past, device):
    v = vids[q["vid"]]
    lo = max(0, q["c"] - max_past)
    cl = [q["patch"] if (q["patch"] is not None and i == q["p"]) else v["clips"][i]
          for i in range(lo, q["c"])]
    return torch.cat(cl, 0).float().to(device)[None]


def student(model, processor, wrapper, vids, q, carry, lid, device):
    """Une passe eleve, reponse forcee : (logits des 4 lettres, CE, etats de la reponse)."""
    v = vids[q["vid"]]
    y = q["y"].long()
    inp, n0 = forced_inputs(processor, v["windows"][q["c"]].float(), q["prompt"], y,
                            v["fps"], device)
    if carry is None:
        with disabled(wrapper):
            out = model(**inp, use_cache=False, output_hidden_states=True)
    else:
        wrapper.carry, wrapper.token_offset = carry, 0
        out = model(**inp, use_cache=False, output_hidden_states=True)
    T = y.shape[0]
    lg = out.logits[0, n0 - 1:n0 - 1 + T].float()
    ce = F.cross_entropy(lg, y.to(device))
    h = out.hidden_states[-1][0, n0 - 1:n0 - 1 + T].float()
    return lg[0, lid], ce, h


_BASE = {}


@torch.no_grad()
def base_states(model, processor, wrapper, vids, q, lid, device):
    """Etats de reponse de Qwen sur la fenetre SEULE (greffe coupee) ; fixes, donc en cache."""
    if q["qid"] not in _BASE:
        _BASE[q["qid"]] = student(model, processor, wrapper, vids, q, None, lid, device)[2].half().cpu()
    return _BASE[q["qid"]].float().to(device)


def lat_loss(h, q, hb, device):
    """Ecart latent NORMALISE a la reponse : ||h - h_ens||^2 / ||h_fenetre - h_ens||^2.

    1 = la memoire n'apporte rien a la fenetre ; 0 = l'eleve rejoint l'enseignant qui voit la
    preuve. (Le 1 - cos du premier jet valait 0,01 des le depart : les etats de derniere
    couche sont presque colineaires, le cosinus ne separe rien.)
    """
    hT = q["h_teacher"].float().to(device)
    return ((h - hT) ** 2).sum() / (((hb - hT) ** 2).sum() + 1e-6)


@torch.no_grad()
def evaluate(model, processor, branch, wrapper, vids, ev, lid, a):
    branch.eval()
    rows = []
    for i, q in enumerate(ev):
        z = past(vids, q, a.max_past, a.device)
        st, S, _, _ = (write_stream_ckpt(branch, z, a.chunk, a.ckpt_seg) if a.ckpt_seg
                       else write_stream(branch, z, a.chunk))
        lat = q.get("h_teacher") is not None
        hb = base_states(model, processor, wrapper, vids, q, lid, a.device) if lat else None
        lm, _, hm = student(model, processor, wrapper, vids, q, (st, S), lid, a.device)
        # memoire d'une AUTRE video : la question suivante d'une autre video
        o = next((ev[(i + k) % len(ev)] for k in range(1, len(ev))
                  if ev[(i + k) % len(ev)]["vid"] != q["vid"]), q)
        zo = past(vids, o, a.max_past, a.device)
        so, So, _, _ = write_stream(branch, zo, a.chunk)
        lw, _, hw = student(model, processor, wrapper, vids, q, (so, So), lid, a.device)
        rows.append({"kind": q["kind"], "dist": q["c"] - q["p"], "gold": q["answer"],
                     "mem": int(lm.argmax()), "wrong": int(lw.argmax()),
                     "window": int(q["lt_window"].argmax()), "teacher": int(q["lt_teacher"].argmax()),
                     "lat_mem": float(lat_loss(hm, q, hb, a.device)) if lat else 1.0,
                     "lat_wrong": float(lat_loss(hw, q, hb, a.device)) if lat else 1.0})
    branch.train()

    def summ(sel):
        n = len(sel)
        if not n:
            return None
        acc = {f: sum(r[f] == r["gold"] for r in sel) / n for f in ("mem", "wrong", "window", "teacher")}
        return {"n": n, **acc, "lat_mem": sum(r["lat_mem"] for r in sel) / n,
                "lat_wrong": sum(r["lat_wrong"] for r in sel) / n}

    out = {"all": summ(rows)}
    for k in ("shape", "natural", "color", "cgb"):
        out[k] = summ([r for r in rows if r["kind"] == k])
        out[k + "_dur"] = summ([r for r in rows if r["kind"] == k and r["window"] != r["gold"]])
    for lo, hi in ((2, 5), (5, 15), (15, 100)):                  # cgb : distance en clips
        out[f"cgb_d{lo}-{hi}"] = summ([r for r in rows if r["kind"] == "cgb" and lo <= r["dist"] < hi])
    for lo, hi in ((2, 4), (4, 10), (10, 100)):
        out[f"shape_d{lo}-{hi}"] = summ([r for r in rows if r["kind"] in ("shape", "color")
                                         and lo <= r["dist"] < hi])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--kinds", default="shape,natural")
    ap.add_argument("--lam-ce", type=float, default=1.0, dest="lam_ce")
    ap.add_argument("--lam-lat", type=float, default=1.0, dest="lam_lat")
    ap.add_argument("--lam-con", type=float, default=1.0, dest="lam_con")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--teacher-ok-only", type=int, default=1, dest="teacher_ok_only")
    ap.add_argument("--max-past", type=int, default=40, dest="max_past", help="clips de memoire au plus")
    ap.add_argument("--ckpt-seg", type=int, default=0, dest="ckpt_seg",
                    help="recalcul par segments de N tokens (0 = non ; gradient identique)")
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--memory-hidden", type=int, default=64, dest="memory_hidden")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=8, dest="write_stride")
    ap.add_argument("--alpha-init", type=float, default=1.27e-4, dest="alpha_init")
    ap.add_argument("--muon", default="unit")
    ap.add_argument("--omega-window", type=int, default=1, dest="omega_window")
    ap.add_argument("--omega-stride", type=int, default=1, dest="omega_stride")
    ap.add_argument("--feature-degree", type=int, default=1, dest="feature_degree")
    ap.add_argument("--init", default=None)
    ap.add_argument("--variants", default=None,
                    help="variantes de couleur (glob) : une couleur tiree au hasard a chaque passage")
    ap.add_argument("--drop-blind", action="store_true", dest="drop_blind",
                    help="entrainement sans les questions que Qwen resout sans video")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=500, dest="save_every")
    ap.add_argument("--max-eval", type=int, default=None, dest="max_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    kinds = set(a.kinds.split(","))
    vids, qs = load(a.targets, a.basis, a.max_past)
    qs = [q for q in qs if q["kind"] in kinds]
    tr = [q for q in qs if not q["eval"] and (q["teacher_ok"] or not a.teacher_ok_only)
          and not (a.drop_blind and q["blind_ok"])]
    ev = [q for q in qs if q["eval"]]
    random.Random(1).shuffle(ev)
    ev = ev[:a.max_eval]
    V = None
    if a.variants:
        assert a.lam_lat == 0 and a.lam_con == 0, "variantes : CE seule"
        V = load_variants(a.variants, a.basis)
        tr = [q for q in tr if q["qid"] in V]
        print(f"variantes : {len(V)} questions, {len(tr)} questions d'entrainement couvertes", flush=True)
    if not ev:                                    # smoke sur quelques videos sans eval
        print("  (aucune video d'eval : eval sur 4 questions d'entrainement, smoke seulement)", flush=True)
        seen = set()
        ev = [q for q in tr if not (q["vid"] in seen or seen.add(q["vid"]))][:4]
    print(f"{len(vids)} videos ; entrainement {len(tr)} questions "
          f"({sum(q['kind'] == 'shape' for q in tr)} formes, {sum(q['kind'] == 'color' for q in tr)} couleurs) ; eval {len(ev)}", flush=True)

    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    lid = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    if a.init:
        blob = torch.load(a.init, map_location="cpu", weights_only=False)
        for k in ("d_mem", "memory_hidden", "alpha_init", "muon", "write_stride",
                  "omega_window", "omega_stride", "feature_degree"):
            if k in blob["args"]:
                setattr(a, k, blob["args"][k])
    branch = MemoryBranch(int(model.config.text_config.hidden_size), a.d_mem,
                          chunk_size=a.chunk, write_stride=a.write_stride,
                          down_pca=pca, **memory_kwargs_from_args(a)).to(a.device)
    if a.init:
        branch.load_state_dict(blob["branch"])
    branch.train()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    hist, glog, t0 = [], [], time.time()

    def report(step):
        r = evaluate(model, processor, branch, wrapper, vids, ev, lid, a)
        hist.append({"step": step, **r})

        def f(k):
            s = r.get(k)
            return "-" if not s else (f"{100 * s['mem']:.0f}/{100 * s['wrong']:.0f}/{100 * s['window']:.0f}"
                                      f"/{100 * s['teacher']:.0f} (n {s['n']})")
        s = r["all"]
        print(f"  == eval {step}: memoire {100 * s['mem']:.1f} | autre video {100 * s['wrong']:.1f} | "
              f"fenetre {100 * s['window']:.1f} | enseignant {100 * s['teacher']:.1f} | lat {s['lat_mem']:.3f}"
              f"/{s['lat_wrong']:.3f}  || formes {f('shape')}  formes dures {f('shape_dur')}  "
              f"naturel {f('natural')}  naturel dur {f('natural_dur')}  couleur {f('color')}  "
              f"couleur dure {f('color_dur')}  || dist "
              + " ".join(f"{k[7:]} {f(k)}" for k in r if k.startswith("shape_d") and k[7].isdigit()), flush=True)

    report(0)
    run = {}
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        bs = rng.sample(tr, a.batch)
        if V is not None:
            bs = [dict(q, **rng.choice(V[q["qid"]])) for q in bs]
        try:
            opt.zero_grad(set_to_none=True)
            tot = {"ce": 0.0, "lat": 0.0, "con": 0.0, "hit": 0}
            gates = pr_w = None
            for j, q in enumerate(bs):
                use_lat = a.lam_lat > 0 and q.get("h_teacher") is not None
                hb = base_states(model, processor, wrapper, vids, q, lid, a.device) if use_lat else None
                zq = past(vids, q, a.max_past, a.device)
                st, S, _, g3 = (write_stream_ckpt(branch, zq, a.chunk, a.ckpt_seg) if a.ckpt_seg
                                else write_stream(branch, zq, a.chunk))
                if gates is None:
                    gates, pr_w = g3, pr(st["w1"][0].detach())
                lm, ce, h = student(model, processor, wrapper, vids, q, (st, S), lid, a.device)
                lat = (lat_loss(h, q, hb, a.device) if use_lat and not q.get("variant")
                       else h.new_zeros(()))
                loss = a.lam_ce * ce + a.lam_lat * lat
                if a.lam_con > 0:
                    o = next((x for x in bs[j + 1:] + bs[:j] if x["vid"] != q["vid"]), None)
                    if o is not None:
                        with torch.no_grad():
                            so, So, _, _ = write_stream(branch, past(vids, o, a.max_past, a.device), a.chunk)
                        _, _, hw = student(model, processor, wrapper, vids, q, (so, So), lid, a.device)
                        con = F.relu(a.margin + lat - lat_loss(hw, q, hb, a.device))
                        loss = loss + a.lam_con * con
                        tot["con"] += float(con)
                (loss / len(bs)).backward()
                tot["ce"] += float(ce); tot["lat"] += float(lat)
                tot["hit"] += int(lm.argmax()) == q["answer"]
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"  OOM au pas {step}", flush=True)
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
        opt.step()
        for k, v in tot.items():
            x = v / len(bs)
            run[k] = x if k not in run else 0.95 * run[k] + 0.05 * x
        if step % 10 == 0:
            al, et, th = gates
            gl = {"step": step, "eta_med": float(et.median()), "alpha_med": float(al.median()),
                  "theta_med": float(th.median()), "pr_w1": pr_w, **run}
            glog.append(gl)
            print(f"  {step:5d} ce {run['ce']:.3f} lat {run['lat']:.3f} con {run['con']:.3f} "
                  f"acc~ {100 * run['hit']:.0f}% |g| {gn:.2f} eta {gl['eta_med']:.3f} "
                  f"alpha {gl['alpha_med']:.1e} theta {gl['theta_med']:.3f} pr {gl['pr_w1']:.1f} "
                  f"{(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)
        last = step + 1 == a.steps
        if a.save_every and ((step + 1) % a.save_every == 0 or last):
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a)},
                       f"{a.out}_at{step + 1}.pt")
        if (step + 1) % a.eval_every == 0 or last:
            report(step + 1)
        if step % 50 == 0 or last:
            with open(a.out + ".json", "w") as f:
                json.dump({"args": vars(a), "history": hist, "gates": glog,
                           "elapsed_s": time.time() - t0}, f, indent=1)
    print("TRAIN-STREAM-FINIE", flush=True)


if __name__ == "__main__":
    main()
