"""21/10 : cibles du regime FLUX pour CG-Bench adapte (items.jsonl de build_cgb.py).

Meme regime que stream_targets.py (FineVideo) : video lue a --fps FIXE, clips de --clip images,
Qwen ne voit qu'UN clip (la fenetre c), la memoire recoit les clips 0..c-1 evinces, chacun
encode seul (etats BRUTS couche 24, un token sur --stride ; la base PCA se choisit au chargement).

Pour chaque question et chaque instant T de l'item :
  c = dernier clip entierement fini a T (la fenetre) ; la preuve doit finir avant le clip c
      (sinon l'instant est ecarte : la preuve serait dans la fenetre) ;
  p = clip qui contient la fin de la preuve (distance en clips = c - p) ;
  trois mesures de Qwen, logits des 4 lettres :
    lt_blind   : question + options SANS video (filtre 1 : a priori de langue) ;
    lt_window  : fenetre seule (plancher, filtre 2) ;
    lt_teacher : [16 images de l'intervalle de preuve | fenetre] (plafond, filtre 3).
Format identique a stream_targets.py (cles c, p, prompt, y, answer, lt_*), donc lisible par
train_stream.load. Pas d'etats latents de l'enseignant (perte sur le sens des reponses).

Un fichier par video, jamais reecrit s'il existe (reprise sur coupure).

    PYTHONPATH=src python cgb_targets.py --items ~/work/iclr/cgb/v1/items.jsonl \\
        --videos ~/work/iclr/cgb/videos --out ~/work/iclr/cgb/v1/tg --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, time
from collections import defaultdict

import torch

LET = "ABCD"
TAIL = "Answer with the option's letter followed by the option text."


def shard_of(vid: str, n: int) -> int:
    return int(hashlib.md5(("s" + vid).encode()).hexdigest()[:8], 16) % n


def find_video(root, uid):
    for ext in ("mp4", "mkv", "webm", "avi", "mov"):
        hit = glob.glob(os.path.join(root, "**", f"{uid}.{ext}"), recursive=True)
        if hit:
            return hit[0]
    return None


def evidence_frames(frames, fps, t0, t1, n):
    """n images reparties uniformement sur [t0, t1] (elargi a n/fps s si la preuve est plus courte)."""
    need = n / fps
    if t1 - t0 < need:
        mid = (t0 + t1) / 2
        t0, t1 = max(0.0, mid - need / 2), mid + need / 2
    idx = torch.linspace(t0 * fps, t1 * fps - 1e-6, n).long().clamp(0, frames.shape[0] - 1)
    return frames[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor
    from titans.graft.check_qwen import load_host
    from titans.graft.distill import forced_inputs, text_only_inputs
    from titans.graft.fit_basis import video_inputs
    from titans.graft.qwen import default_layer, find_decoder_layers
    from titans.graft.stream_targets import read_fps

    os.makedirs(a.out, exist_ok=True)
    by_vid = defaultdict(list)
    for line in open(os.path.expanduser(a.items)):
        it = json.loads(line)
        by_vid[it["video_uid"]].append(it)
    todo = sorted(v for v in by_vid if shard_of(v, a.nshards) == a.shard)[:a.limit]
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
    def letters(frames, prompt):
        inp = (text_only_inputs(proc, prompt) if frames is None             # texte seul
               else video_inputs(proc, frames.float(), prompt, a.fps))
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        return model(**inp, use_cache=False).logits[0, -1, lid].float().cpu()

    @torch.no_grad()
    def forced_letters(frames, prompt, y):
        inp, n0 = forced_inputs(proc, frames.float(), prompt, y, a.fps, a.device)
        return model(**inp, use_cache=False).logits[0, n0 - 1, lid].float().cpu()

    t0, done = time.time(), 0
    for vid in todo:
        dst = os.path.join(a.out, vid + ".pt")
        if os.path.exists(dst):
            continue
        path = find_video(a.videos, vid)
        if path is None:
            print(f"  {vid} : video absente", flush=True)
            continue
        try:
            frames, dur = read_fps(path, a.fps, a.max_side)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {vid} saute : {e}", flush=True)
            continue
        span = a.clip / a.fps
        nclip = frames.shape[0] // a.clip
        clip = lambda c: frames[c * a.clip:(c + 1) * a.clip]    # noqa: E731
        clips = [encode(clip(c)) for c in range(nclip)]
        qs, windows = [], {}
        for it in by_vid[vid]:
            prompt = "\n".join([it["question"]] + [f"{LET[i]}. {o}" for i, o in enumerate(it["options"])] + [TAIL])
            ans = it["answer_idx"]
            y = proc.tokenizer(f"{LET[ans]}. {it['options'][ans]}", add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
            t_ev0, t_ev1 = it["evidence_span"]
            ev = evidence_frames(frames, a.fps, t_ev0, t_ev1, a.clip)
            blind = letters(None, prompt)
            for T in it["query_times"]:
                c = min(int(T // span), nclip) - 1             # dernier clip fini a T
                if c < 1 or t_ev1 >= c * span:                 # preuve pas encore sortie de la fenetre
                    continue
                p = int(min(t_ev1, dur) // span)
                win = clip(c)
                if c not in windows:
                    windows[c] = win.clone()
                qs.append({"kind": "cgb", "c": c, "p": p, "prompt": prompt, "y": y, "answer": ans,
                           "options": it["options"], "lt_blind": blind, "lt_window": letters(win, prompt),
                           "lt_teacher": forced_letters(torch.cat([ev, win]), prompt, y),
                           "h_teacher": None, "patch": None,
                           "meta": {"qid": it["qid"], "T": T, "dist_s": round(T - t_ev1, 1),
                                    "sub_category": it["sub_category"], "domain": it["domain"]}})
        torch.save({"vid": vid, "dur": dur, "fps": a.fps, "cat": by_vid[vid][0]["domain"],
                    "eval": by_vid[vid][0]["split"] == "eval", "clips": clips, "windows": windows,
                    "qs": qs}, dst)
        done += 1
        print(f"  [{done}/{len(todo)}] {vid} {nclip} clips {len(qs)} q "
              f"{(time.time() - t0) / done:.1f} s/video", flush=True)
    print("CGB-TARGETS-FINI", flush=True)


if __name__ == "__main__":
    main()
