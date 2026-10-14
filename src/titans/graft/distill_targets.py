"""Cibles de distillation a contexte complet, PONDEREES par ce que le contexte apporte.

Ce que la premiere distillation (``distill.py``) a appris, et ce qu'on corrige
-------------------------------------------------------------------------------
Une question fixe, des cibles dures (la reponse gloutonne du professeur) : le gain
etait a 61 % de la projection (l'eleve imite le STYLE), 32 % "ecrire n'importe
quoi", 7 % le bon contenu ([[distillation-gain-is-mostly-not-content]]). Ici :

  * trois passes de l'hote gele, toutes precalculees :
      complet : Qwen voit la video jusqu'a la fin de la fenetre   -> p_complet
      court   : Qwen voit la fenetre seule                         -> p_court
      (l'eleve, a l'entrainement : la fenetre + la memoire du passe -> p_eleve)
  * cibles DOUCES : top-``k`` du professeur + un seau "reste" (KL exacte sur la
    partition k+1) ;
  * poids par token ``w_t = KL(p_complet,t || p_court,t)`` : un token que la fenetre
    suffit a predire (style, syntaxe, a priori) pese ~0 ; ne compte que ce que le
    contexte retire apporte, i.e. ce que seule la memoire peut fournir ;
  * des questions VARIEES et ANCREES : les scenes annotees de FineVideo (titre +
    horodatage) dont la fin precede la fenetre, tirees UNIFORMEMENT sur le passe
    (pas de prime a la recence -> pas de pression "cache",
    [[collapse-is-horizon-retreat]]) ; + questions generiques + QA FineVideo.
    ``dist_s`` (fin de scene -> debut de fenetre) est garde pour verifier que les
    poids ne se concentrent pas sur le recent (biais de recence du professeur).

Memoire de l'eleve : la video est echantillonnee en ``--frames`` images, coupee en
clips de ``--clip`` images encodes SEULS (comme ``train_load``) ; la fenetre est le
dernier clip, le passe = les clips d'avant. Fins de fenetre : ``--ends`` (en clips).

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m titans.graft.distill_targets \\
        --data data/finevideo --basis cache/qwen_l24_pca128.pt \\
        --shard 0 --nshards 5 --out ~/work/iclr/distill/targets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .branch import MemoryBranch
from .distill import forced_inputs
from .qwen import default_layer, find_decoder_layers
from .train_load import video_z

SCENE_Q = [
    'Earlier in this video there was a part about "{t}". What was shown in that part? '
    'Answer in one or two sentences.',
    'Think back to the earlier segment of this video about "{t}". Briefly describe what '
    'happened and who or what was visible.',
]
GENERIC_Q = [
    "What happened earlier in this video, before the current moment? Answer in two "
    "sentences.",
    "Which people, objects or places appeared earlier in this video? Answer briefly.",
    "How did this video begin? Briefly describe the first scene.",
]


def is_eval(vid: str, frac: float) -> bool:
    h = int(hashlib.md5(vid.encode()).hexdigest()[:8], 16)
    return (h % 10000) / 10000 < frac


def ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def questions(meta: dict, t_win: float, rng: random.Random, n_scene: int, n_gen: int):
    """``[(question, kind, dist_s)]`` ; ``dist_s`` = debut de fenetre - fin de scene."""
    cm = meta.get("content_metadata") or {}
    out = []
    past = []
    for sc in cm.get("scenes") or []:
        try:
            end = ts(sc["timestamps"]["end_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < t_win and sc.get("title"):
            past.append((sc["title"], t_win - end))
    for title, d in rng.sample(past, min(n_scene, len(past))):
        out.append((rng.choice(SCENE_Q).format(t=title), "scene", d))
    for q in rng.sample(GENERIC_Q, n_gen):
        out.append((q, "generic", None))
    qa = [x for x in cm.get("qAndA") or [] if x.get("question")]
    if qa:
        out.append((rng.choice(qa)["question"] + " Answer briefly.", "qa", None))
    return out


def topk_lp(logits: torch.Tensor, k: int):
    """``(idx [T,k], lp [T,k+1])`` : top-k log-probas + le seau reste."""
    lp = F.log_softmax(logits.float(), -1)
    top, idx = lp.topk(k, -1)
    rest = torch.log1p(-top.exp().sum(-1).clamp(max=1 - 1e-6))
    return idx, torch.cat([top, rest.unsqueeze(-1)], -1)


def gather_lp(logits: torch.Tensor, idx: torch.Tensor):
    lp = F.log_softmax(logits.float(), -1)
    g = lp.gather(-1, idx)
    rest = torch.log1p(-g.exp().sum(-1).clamp(max=1 - 1e-6))
    return torch.cat([g, rest.unsqueeze(-1)], -1)


def kl_buckets(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) par token sur la partition top-k + reste. ``[T]``."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1).clamp_min(0)


@torch.no_grad()
def teacher(model, processor, frames, q, fps, a):
    from .fit_basis import video_inputs
    inp = video_inputs(processor, frames, q, fps)
    inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False,
                         repetition_penalty=1.0, output_logits=True,
                         return_dict_in_generate=True, use_cache=True)
    y = out.sequences[0, inp["input_ids"].shape[1]:]
    logits = torch.stack(out.logits, 1)[0]                     # [T, V]
    return y.cpu(), logits


@torch.no_grad()
def forced_logits(model, processor, frames, q, y, fps, device):
    inp, n_prompt = forced_inputs(processor, frames, q, y, fps, device)
    lg = model(**inp, use_cache=False).logits
    return lg[0, n_prompt - 1: n_prompt - 1 + y.shape[0]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--ends", default="8,12,16", help="fins de fenetre, en clips")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--n-scene", type=int, default=3, dest="n_scene")
    ap.add_argument("--n-generic", type=int, default=2, dest="n_generic")
    ap.add_argument("--max-new", type=int, default=48, dest="max_new")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--eval-frac", type=float, default=0.05, dest="eval_frac")
    ap.add_argument("--write-stride", type=int, default=8, dest="write_stride")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, help="prefixe ; ecrit <out>_<shard>.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from ..worldmodel.data import load_video_frames
    from .check_qwen import load_host
    from .fit_basis import resize_frames

    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    down = MemoryBranch(int(model.config.text_config.hidden_size), 128,
                        down_pca=pca).down.to(a.device)
    ends = [int(x) for x in a.ends.split(",")]

    rows = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))]
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    if a.limit:
        rows = rows[:a.limit]
    out_path = f"{a.out}_{a.shard}.pt"
    done = {"videos": {}, "samples": []}
    if os.path.exists(out_path):                     # reprise
        done = torch.load(out_path, weights_only=False)
    t0, n_new = time.time(), 0
    for vi, r in enumerate(rows):
        vid = r["id"]
        if vid in done["videos"]:
            continue
        path = os.path.join(a.data, "videos", vid + ".mp4")
        dur = float(r.get("duration") or 0)
        if not os.path.exists(path) or dur <= 0:
            continue
        try:
            allf = resize_frames(load_video_frames(path, num_frames=a.frames), a.max_side)
        except Exception as e:
            print(f"  {vid} saute : {e}", flush=True)
            continue
        fps = a.frames / dur
        nclip = a.frames // a.clip
        clips = [video_z(model, processor, allf[c * a.clip:(c + 1) * a.clip], fps, site,
                         down, a.device, a.write_stride) for c in range(nclip)]
        rng = random.Random(int(hashlib.md5(vid.encode()).hexdigest()[:8], 16))
        wins = {}
        for e in ends:
            lo, hi = (e - 1) * a.clip, e * a.clip
            win = allf[lo:hi]
            wins[e] = win.clone()
            t_win = lo * dur / a.frames
            for q, kind, dist in questions(r["meta"], t_win, rng, a.n_scene, a.n_generic):
                try:
                    y, lg_full = teacher(model, processor, allf[:hi], q, fps, a)
                    if y.numel() == 0:
                        continue
                    idx, lp_full = topk_lp(lg_full, a.topk)
                    lg_short = forced_logits(model, processor, win, q, y, fps, a.device)
                    lp_short = gather_lp(lg_short, idx)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                w = kl_buckets(lp_full, lp_short)
                done["samples"].append({
                    "vid": vid, "end": e, "q": q, "kind": kind, "dist_s": dist,
                    "y": y.to(torch.int32), "idx": idx.cpu().to(torch.int32),
                    "lp_full": lp_full.cpu().half(), "lp_short": lp_short.cpu().half(),
                    "w": w.cpu(), "eval": is_eval(vid, a.eval_frac)})
                n_new += 1
        done["videos"][vid] = {"clips": clips, "windows": wins, "fps": fps, "dur": dur,
                               "eval": is_eval(vid, a.eval_frac)}
        el = time.time() - t0
        print(f"  [{a.shard}] video {vi + 1}/{len(rows)} {vid}  echantillons {len(done['samples'])}"
              f"  ({el / max(1, n_new):.2f} s/echantillon)", flush=True)
        if (vi + 1) % 20 == 0:
            torch.save(done, out_path)
    torch.save(done, out_path)
    print("TARGETS-FINIE", flush=True)


if __name__ == "__main__":
    main()
