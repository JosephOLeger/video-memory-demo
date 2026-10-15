"""Cibles du regime FLUX (15/10) : fenetre glissante a fps fixe, memoire du passe evince.

Objectif long terme : un agent assistant au contexte quasi infini. Le pretraitement de Qwen
(echantillonnage uniforme de toute la video) n'y a pas de sens. Ici :

* la video est lue a ``--fps`` FIXE (1 image/s), quelle que soit sa duree ;
* elle est decoupee en clips de ``--clip`` images (16 s) ; Qwen voit UN clip, la fenetre ``c`` ;
* la memoire recoit les clips ``0..c-1`` evinces, chacun encode seul (ses etats couche 24,
  un token sur 8). On stocke les etats BRUTS (2048) : la base PCA se choisit au chargement.

Questions (QCM a 4 options, reponse = « B. un triangle rouge ») :

* ``shape`` : une forme coloree est incrustee 8 s dans un clip ``p < c-1`` du flux reel.
  Options symetriques (bonne, meme couleur autre forme, meme forme autre couleur, aucune
  des deux) : aucun a priori de langue ne peut choisir.
* ``natural`` (liaison) : « pendant <activite annotee>, quel objet etait visible ? » ; les
  4 options sont des objets de la MEME video, tous du passe, de scenes differentes, absents
  de la fenetre ; on ecarte tout objet qui partage un mot avec l'activite (fuite lexicale).
  (Une premiere version tirait les distracteurs d'autres videos : le nom des personnages
  donnait la reponse, fenetre seule 3/3 au smoke.)

Cible latente (proposition de Joseph) : on ne copie pas tout l'etat local de Qwen, seulement
LA REPONSE. Un enseignant Qwen voit [clip de preuve p | fenetre c] (32 images) + la question,
reponse forcee ; on garde ses etats de derniere couche aux positions de la reponse. L'eleve
(fenetre + memoire) devra les retrouver. On garde aussi les logits des 4 lettres de
l'enseignant (plafond) et de Qwen sur la fenetre seule (plancher).

    PYTHONPATH=src python -m titans.graft.stream_targets --data data/finevideo \\
        --out ~/work/iclr/stream/tg --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time

import torch

LET = "ABCD"
SHAPES = ["circle", "square", "triangle", "star", "diamond", "cross"]
COLORS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (40, 80, 230),
          "yellow": (240, 210, 20), "purple": (150, 50, 190), "orange": (245, 130, 20)}
Q_SHAPE = "Earlier in this video, a colored shape appeared on the screen. Which one was it?"
TAIL = "Answer with the option's letter followed by the option text."


def is_eval(vid: str, frac: float = 0.1) -> bool:
    return (int(hashlib.md5(vid.encode()).hexdigest()[:8], 16) % 10000) / 10000 < frac


def shard_of(vid: str, n: int) -> int:
    return int(hashlib.md5(("s" + vid).encode()).hexdigest()[:8], 16) % n


def ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def items(meta, key, field):
    cm = (meta or {}).get("content_metadata") or {}
    out = []
    for sc in cm.get("scenes") or []:
        for x in sc.get(key) or []:
            try:
                t = x.get("timestamp") or x.get("timestamps") or {}
                a, b = ts(t["start_timestamp"]), ts(t["end_timestamp"])
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            txt = (x.get(field) or "").strip()
            if txt and len(txt) < 120:
                out.append((txt, a, b))
    return out


STOP = {"the", "and", "with", "from", "into", "onto", "their", "this", "that", "while", "them"}


def words(t: str) -> set:
    return {w.strip(".,;:!?\"'()").lower() for w in t.split()
            if len(w) > 3 and w.lower() not in STOP}


def shape_mask(kind, H, W, cy, cx, r):
    y = torch.arange(H).view(-1, 1).float()
    x = torch.arange(W).view(1, -1).float()
    dy, dx = y - cy, x - cx
    if kind == "circle":
        return dx ** 2 + dy ** 2 <= r ** 2
    if kind == "square":
        return (dx.abs() <= r * 0.85) & (dy.abs() <= r * 0.85)
    if kind == "diamond":
        return dx.abs() + dy.abs() <= r
    if kind == "triangle":
        t = (dy + r) / (2 * r)                                  # 0 en haut, 1 en bas
        return (t >= 0) & (t <= 1) & (dx.abs() <= t * r)
    if kind == "cross":
        return ((dx.abs() <= r / 3) & (dy.abs() <= r)) | ((dy.abs() <= r / 3) & (dx.abs() <= r))
    if kind == "star":
        rr = (dx ** 2 + dy ** 2).sqrt()
        th = torch.atan2(dy, dx) + math.pi / 2
        return rr <= r * (0.5 + 0.5 * torch.cos(5 * th).clamp(min=0) ** 2) * 1.05
    raise ValueError(kind)


def paste(frames, kind, color, rng, f0=4, f1=12):
    """Incruste la forme (bord noir) sur les images f0..f1-1 d'un clip uint8 [F,H,W,3]."""
    out = frames.clone()
    _, H, W, _ = out.shape
    r = min(H, W) / 6
    cy, cx = rng.uniform(r * 1.2, H - r * 1.2), rng.uniform(r * 1.2, W - r * 1.2)
    m = shape_mask(kind, H, W, cy, cx, r)
    edge = shape_mask(kind, H, W, cy, cx, r * 1.15) & ~m
    col = torch.tensor(COLORS[color], dtype=out.dtype)
    for f in range(f0, min(f1, out.shape[0])):
        out[f][edge] = 0
        out[f][m] = col
    return out


def read_fps(path, fps, max_side):
    import decord
    from .fit_basis import resize_frames
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path)
    src = float(vr.get_avg_fps())
    n = int(len(vr) / src * fps)
    idx = [min(len(vr) - 1, int(round(i * src / fps))) for i in range(n)]
    parts = [resize_frames(vr.get_batch(idx[i:i + 64]), max_side) for i in range(0, n, 64)]
    return torch.cat(parts), n / fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--min-past", type=int, default=3, dest="min_past", help="clips de passe minimum")
    ap.add_argument("--n-shape", type=int, default=3, dest="n_shape")
    ap.add_argument("--n-natural", type=int, default=3, dest="n_natural")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true",
                    help="parcourt l'eclat a l'envers (deux cartes sur un eclat se rejoignent au milieu)")
    ap.add_argument("--out", required=True, help="dossier ; un fichier par video")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host
    from .distill import forced_inputs
    from .fit_basis import video_inputs
    from .qwen import default_layer, find_decoder_layers

    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(line) for line in open(os.path.join(a.data, "manifest.jsonl"))]
    todo = [r for r in rows if shard_of(r["id"], a.nshards) == a.shard][:a.limit]
    if a.reverse:
        todo = todo[::-1]

    model = load_host(a.model, torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained(a.model)
    layers = find_decoder_layers(model)
    site = default_layer(len(layers))
    got = []
    layers[site].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    vtok = model.config.video_token_id
    lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]

    @torch.no_grad()
    def encode(frames):
        got.clear()
        inp = video_inputs(proc, frames.float(), "", a.fps)
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        model(**inp, use_cache=False)
        h = got[0][0][inp["input_ids"][0] == vtok]
        return h[::a.stride].half().cpu()

    @torch.no_grad()
    def forced(frames, prompt, y):
        inp, n0 = forced_inputs(proc, frames.float(), prompt, y, a.fps, a.device)
        out = model(**inp, use_cache=False, output_hidden_states=True)
        T = y.shape[0]
        h = out.hidden_states[-1][0, n0 - 1:n0 - 1 + T].half().cpu()
        return h, out.logits[0, n0 - 1, lid].float().cpu()

    @torch.no_grad()
    def letters(frames, prompt):
        inp = video_inputs(proc, frames.float(), prompt, a.fps)
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        return model(**inp, use_cache=False).logits[0, -1, lid].float().cpu()

    t0, done = time.time(), 0
    for r in todo:
        vid = r["id"]
        dst = os.path.join(a.out, vid + ".pt")
        if os.path.exists(dst):
            continue
        rng = random.Random(vid)
        try:
            frames, dur = read_fps(os.path.join(a.data, "videos", vid + ".mp4"), a.fps, a.max_side)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {vid} saute : {e}", flush=True)
            continue
        nclip = frames.shape[0] // a.clip
        if nclip < a.min_past + 2:
            continue
        clip = lambda c: frames[c * a.clip:(c + 1) * a.clip]    # noqa: E731
        clips = [encode(clip(c)) for c in range(nclip)]
        span = a.clip / a.fps                                   # secondes par clip
        cat = (r.get("meta") or {}).get("content_fine_category") or "?"
        qs, windows = [], {}

        def ask(kind, c, p, opts, ans, question, ev_frames, patch=None, meta=None):
            prompt = "\n".join([question] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL])
            y = proc.tokenizer(f"{LET[ans]}. {opts[ans]}", add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
            win = clip(c)
            hT, lT = forced(torch.cat([ev_frames, win]), prompt, y)
            lB = letters(win, prompt)
            windows[c] = win.clone()                            # une vue sauverait toute la video
            qs.append({"kind": kind, "c": c, "p": p, "prompt": prompt, "y": y, "answer": ans,
                       "options": opts, "h_teacher": hT, "lt_teacher": lT, "lt_window": lB,
                       "patch": patch, "meta": meta or {}})

        # formes : clip de preuve p <= c-2 (au moins un clip d'ecart avec la fenetre)
        cs = list(range(a.min_past, nclip))
        for _ in range(a.n_shape):
            c = rng.choice(cs)
            p = rng.randrange(0, c - 1)
            sh, co = rng.choice(SHAPES), rng.choice(list(COLORS))
            sh2 = rng.choice([s for s in SHAPES if s != sh])
            co2 = rng.choice([x for x in COLORS if x != co])
            opts = [f"a {co} {sh}", f"a {co} {sh2}", f"a {co2} {sh}", f"a {co2} {sh2}"]
            rng.shuffle(opts)
            ev = paste(clip(p), sh, co, rng)
            ask("shape", c, p, opts, opts.index(f"a {co} {sh}"), Q_SHAPE, ev, patch=encode(ev),
                meta={"shape": sh, "color": co})

        # naturel (liaison) : « pendant <activite A>, quel objet etait visible ? ». Les quatre
        # options sont des objets de la MEME video, tous du PASSE, chacun d'une scene differente :
        # ni « appartient a cette video » ni « tot / tard » ne designent la reponse.
        cm = (r.get("meta") or {}).get("content_metadata") or {}
        scenes = []
        for sc in cm.get("scenes") or []:
            acts = items({"content_metadata": {"scenes": [sc]}}, "activities", "description")
            props = items({"content_metadata": {"scenes": [sc]}}, "props", "name")
            if acts and props:
                scenes.append((acts, props))
        nat = []
        for si, (acts, props) in enumerate(scenes):
            for act, a0, a1 in acts:
                for prop, b0, b1 in props:
                    lo, hi = max(a0, b0), min(a1, b1)
                    if hi <= lo or words(prop) & words(act):
                        continue
                    p = int((lo + hi) / 2 // span)
                    nat.append((si, act, prop, p, hi))
        rng.shuffle(nat)
        used = set()
        for si, act, prop, p, end in nat:
            if sum(q["kind"] == "natural" for q in qs) >= a.n_natural:
                break
            c_ok = [c for c in cs if c - 1 >= p + 1 and end < c * span]
            if prop in used or not c_ok:
                continue
            c = rng.choice(c_ok)
            lo_w, hi_w = c * span, (c + 1) * span
            inwin = {t for _, pr in scenes for t, x, y_ in pr if x <= hi_w and y_ >= lo_w}
            mine = {t for t, _, _ in scenes[si][1]}
            others = sorted({t for sj, (_, pr) in enumerate(scenes) if sj != si
                             for t, x, y_ in pr if y_ < lo_w} - mine - inwin - {prop})
            others = [t for t in others if not words(t) & words(act)]
            if prop in inwin or len(others) < 3:
                continue
            used.add(prop)
            opts = rng.sample(others, 3) + [prop]
            rng.shuffle(opts)
            q = f'Earlier in this video, while the following happened: "{act}", which of these objects was visible?'
            ask("natural", c, p, opts, opts.index(prop), q, clip(p), meta={"activity": act})

        torch.save({"vid": vid, "dur": dur, "fps": a.fps, "cat": cat, "eval": is_eval(vid),
                    "clips": clips, "windows": windows, "qs": qs}, dst)
        done += 1
        if done % 10 == 0:
            print(f"  [{done}/{len(todo)}] {(time.time() - t0) / done:.1f} s/video", flush=True)
    print("STREAM-TARGETS-FINI", flush=True)


if __name__ == "__main__":
    main()
