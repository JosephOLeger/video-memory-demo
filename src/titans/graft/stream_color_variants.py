"""Variantes de couleur pour la tache couleur en flux (20/10).

Diagnostic du 20/10 : les graines 1 et 2 apprennent par coeur « cette memoire (= cette video)
-> cette couleur » sur les 4274 questions d'entrainement, au lieu de lire la couleur. Ici, pour
chaque question d'ENTRAINEMENT de ``tgc`` (memes c, p, f0), on encode le clip de preuve avec
l'aiguille dans CHACUNE des 6 couleurs, avec ses options et sa reponse. ``train_stream
--variants`` tire une variante au hasard a chaque passage : la meme video donne des reponses
differentes, l'identite de la video ne sert plus. Les questions d'eval ne changent pas.

    PYTHONPATH=src python -m titans.graft.stream_color_variants --tgc runs/stream/tgc \\
        --data data/finevideo --out runs/stream/tgv --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

import torch

from .stream_color_targets import Q_COLOR, fill
from .stream_targets import COLORS, LET, TAIL, read_fps, shard_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--tgc", required=True)
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--needle", type=int, default=4)
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
    from .fit_basis import video_inputs
    from .qwen import default_layer, find_decoder_layers

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.tgc), "*.pt")))
    files = [f for f in files if shard_of(os.path.basename(f)[:-3], a.nshards) == a.shard][:a.limit]
    model = load_host(a.model, torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained(a.model)
    layers = find_decoder_layers(model)
    got = []
    layers[default_layer(len(layers))].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    vtok = model.config.video_token_id

    t0, done = time.time(), 0
    for f in files:
        dst = os.path.join(a.out, os.path.basename(f))
        if os.path.exists(dst):
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d["eval"]:
            continue
        vid, fps, clen = d["vid"], d["fps"], a.clip
        try:
            frames, _ = read_fps(os.path.join(a.data, "videos", vid + ".mp4"), fps, a.max_side)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {vid} saute : {e}", flush=True)
            continue

        @torch.no_grad()
        def encode(fr):
            got.clear()
            inp = video_inputs(proc, fr.float(), "", fps)
            inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
            model(**inp, use_cache=False)
            h = got[0][0][inp["input_ids"][0] == vtok]
            return h[::a.stride].half().cpu()

        var = {}
        for i, q in enumerate(d["qs"]):
            p, f0 = q["p"], q["meta"]["f0"]
            base = frames[p * clen:(p + 1) * clen]
            rows = []
            for co in COLORS:
                rng = random.Random(f"{vid}#{i}#{co}")
                opts = [co] + rng.sample([x for x in COLORS if x != co], 3)
                rng.shuffle(opts)
                ans = opts.index(co)
                prompt = "\n".join([Q_COLOR] + [f"{LET[j]}. {o}" for j, o in enumerate(opts)] + [TAIL])
                y = proc.tokenizer(f"{LET[ans]}. {co}", add_special_tokens=False,
                                   return_tensors="pt").input_ids[0]
                rows.append({"color": co, "options": opts, "prompt": prompt, "y": y, "answer": ans,
                             "patch": encode(fill(base, co, f0, f0 + a.needle))})
            var[i] = rows
        torch.save({"vid": vid, "variants": var}, dst)
        done += 1
        if done % 10 == 0:
            print(f"  [{done}/{len(files)}] {(time.time() - t0) / done:.1f} s/video", flush=True)
    print("STREAM-VARIANTS-FINI", flush=True)


if __name__ == "__main__":
    main()
