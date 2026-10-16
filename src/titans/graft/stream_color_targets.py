"""Cibles FLUX « couleur seule » : la tache couleur qui marche (07/10), en regime flux.

Reprend les videos deja precalculees par ``stream_targets`` (memes clips encodes, meme
decoupage eval/entrainement) et n'ajoute que des questions de COULEUR : l'ecran devient
uni (une couleur) pendant ``--needle`` images d'un clip ``p``, et la question demande
laquelle (4 options, 4 couleurs). Comme ``shape``, le clip de preuve est remplace par sa
version avec l'aiguille (``patch``), et l'enseignant voit [clip de preuve | fenetre].

``p`` est tire dans ``[c - max_gap, c - 2]`` : la preuve est au moins un clip avant la
fenetre et au plus ``max_gap`` clips (passe court d'abord : on veut d'abord savoir si le
flux peut apprendre). Seuls les champs ``qs`` et ``windows`` changent.

    PYTHONPATH=src python -m titans.graft.stream_color_targets --reuse ~/work/iclr/stream/tg \\
        --data data/finevideo --out ~/work/iclr/stream/tgc --shard 0 --nshards 1
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

import torch

from .stream_targets import COLORS, LET, TAIL, read_fps, shard_of

Q_COLOR = "Earlier in this video, the whole screen briefly turned a single color. Which color was it?"


def fill(frames, color, f0, f1):
    """Ecran uni ``color`` sur les images f0..f1-1 d'un clip uint8 [F,H,W,3]."""
    out = frames.clone()
    out[f0:min(f1, out.shape[0])] = torch.tensor(COLORS[color], dtype=out.dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--reuse", required=True, help="dossier des cibles flux deja calculees")
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--n-color", type=int, default=4, dest="n_color")
    ap.add_argument("--needle", type=int, default=4, help="images unies dans le clip de preuve")
    ap.add_argument("--max-gap", type=int, default=8, dest="max_gap")
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host
    from .distill import forced_inputs
    from .fit_basis import video_inputs
    from .qwen import default_layer, find_decoder_layers

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.reuse), "*.pt")))
    files = [f for f in files if shard_of(os.path.basename(f)[:-3], a.nshards) == a.shard][:a.limit]

    model = load_host(a.model, torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained(a.model)
    layers = find_decoder_layers(model)
    got = []
    layers[default_layer(len(layers))].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    vtok = model.config.video_token_id
    lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]

    t0, done = time.time(), 0
    for f in files:
        dst = os.path.join(a.out, os.path.basename(f))
        if os.path.exists(dst):
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        vid, fps = d["vid"], d["fps"]
        clen = a.clip
        try:
            frames, _ = read_fps(os.path.join(a.data, "videos", vid + ".mp4"), fps, a.max_side)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {vid} saute : {e}", flush=True)
            continue
        nclip = min(len(d["clips"]), frames.shape[0] // clen)
        clip = lambda c: frames[c * clen:(c + 1) * clen]       # noqa: E731
        if nclip < 4:
            continue

        @torch.no_grad()
        def encode(fr):
            got.clear()
            inp = video_inputs(proc, fr.float(), "", fps)
            inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
            model(**inp, use_cache=False)
            h = got[0][0][inp["input_ids"][0] == vtok]
            return h[::a.stride].half().cpu()

        @torch.no_grad()
        def forced(fr, prompt, y):
            inp, n0 = forced_inputs(proc, fr.float(), prompt, y, fps, a.device)
            out = model(**inp, use_cache=False, output_hidden_states=True)
            T = y.shape[0]
            return (out.hidden_states[-1][0, n0 - 1:n0 - 1 + T].half().cpu(),
                    out.logits[0, n0 - 1, lid].float().cpu())

        @torch.no_grad()
        def letters(fr, prompt):
            inp = video_inputs(proc, fr.float(), prompt, fps)
            inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
            return model(**inp, use_cache=False).logits[0, -1, lid].float().cpu()

        rng = random.Random("color" + vid)
        qs, windows = [], {}
        for _ in range(a.n_color):
            c = rng.randrange(2, nclip)
            p = rng.randrange(max(0, c - a.max_gap), c - 1)
            co = rng.choice(list(COLORS))
            opts = [co] + rng.sample([x for x in COLORS if x != co], 3)
            rng.shuffle(opts)
            ans = opts.index(co)
            f0 = rng.randrange(0, clen - a.needle + 1)
            ev = fill(clip(p), co, f0, f0 + a.needle)
            prompt = "\n".join([Q_COLOR] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL])
            y = proc.tokenizer(f"{LET[ans]}. {co}", add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
            win = clip(c)
            hT, lT = forced(torch.cat([ev, win]), prompt, y)
            windows[c] = win.clone()
            qs.append({"kind": "color", "c": c, "p": p, "prompt": prompt, "y": y, "answer": ans,
                       "options": opts, "h_teacher": hT, "lt_teacher": lT,
                       "lt_window": letters(win, prompt), "patch": encode(ev),
                       "meta": {"color": co, "f0": f0}})
        torch.save({**d, "windows": windows, "qs": qs}, dst)
        done += 1
        if done % 10 == 0:
            print(f"  [{done}/{len(files)}] {(time.time() - t0) / done:.1f} s/video", flush=True)
    print("STREAM-COLOR-FINI", flush=True)


if __name__ == "__main__":
    main()
