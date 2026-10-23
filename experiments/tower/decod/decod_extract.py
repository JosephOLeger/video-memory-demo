"""Sonde de decodabilite, etape 1 : etats bruts couche 24 des clips FineVideo + embeddings des options QCM.

Meme echantillonnage que distill_targets (256 images uniformes, max_side 256, clips de 16,
chaque clip encode seul). Par clip 0..10 (le passe maximal des QCM, fin 12) on garde les
tokens video bruts (2048) de deux facons :
    sub  : un token sur 8 (ce que la memoire recoit aujourd'hui, avant PCA)
    pool : moyenne de chaque groupe de 8 tokens consecutifs (meme nombre de tokens)
Options : etat couche 24 moyen sur les tokens du texte de l'option seul.

    PYTHONPATH=src python decod_extract.py --mcq mcq3.jsonl --data data/finevideo --out DIR
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoProcessor

from titans.graft.check_qwen import load_host
from titans.graft.fit_basis import resize_frames, video_inputs
from titans.graft.qwen import default_layer, find_decoder_layers
from titans.worldmodel.data import load_video_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--nclips", type=int, default=11)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by = defaultdict(list)
    for line in open(a.mcq):
        m = json.loads(line)
        by[m["vid"]].append(m)
    vids = sorted(by)[:a.limit]
    model = load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    layers = find_decoder_layers(model)
    site = default_layer(len(layers))
    got = []
    layers[site].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    vtok = model.config.video_token_id
    t0, n = time.time(), 0
    for vid in vids:
        dst = os.path.join(a.out, vid + ".pt")
        if os.path.exists(dst):
            continue
        ms = by[vid]
        try:
            allf = resize_frames(load_video_frames(os.path.join(a.data, "videos", vid + ".mp4"),
                                                   num_frames=a.frames), a.max_side)
        except Exception as e:
            print(f"  {vid} saute : {e}", flush=True)
            continue
        fps = a.frames / ms[0]["dur"]
        sub, pool = [], []
        with torch.no_grad():
            for c in range(a.nclips):
                got.clear()
                inp = video_inputs(proc, allf[c * a.clip:(c + 1) * a.clip], "", fps)
                inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
                model(**inp, use_cache=False)
                h = got[0][0][inp["input_ids"][0] == vtok].float()          # [n_tok, 2048]
                m = (h.shape[0] // a.stride) * a.stride
                sub.append(h[::a.stride].half().cpu())
                pool.append(h[:m].reshape(-1, a.stride, h.shape[1]).mean(1).half().cpu())
            opts = {}
            for m_ in ms:
                for o in m_["options"]:
                    if o in opts:
                        continue
                    got.clear()
                    ids = proc.tokenizer(o, return_tensors="pt").input_ids.to(a.device)
                    model(input_ids=ids, use_cache=False)
                    opts[o] = got[0][0].float().mean(0).half().cpu()
        torch.save({"sub": sub, "pool": pool, "opts": opts, "mcq": ms}, dst)
        n += 1
        if n % 20 == 0:
            print(f"  [{n}/{len(vids)}] {(time.time() - t0) / n:.1f} s/video", flush=True)
    print("DECOD-EXTRACT-FINI", flush=True)


if __name__ == "__main__":
    main()
