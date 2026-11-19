"""Fenetre glissante : contexte tronque (fenetre recente) vs. + la memoire du passe.

Le regime fidele a la claim (E1 : la memoire comble la taille bornee de l'attention).
On echantillonne ``--total-frames`` images sur TOUTE la video, puis on coupe :

    contexte (ce que l'hote voit) = les ``--window`` images LES PLUS RECENTES
    memoire  (le passe)           = toutes les images AVANT la fenetre

Les deux bras paient le meme cout d'attention (la meme fenetre recente). La seule
difference est que le bras memoire a lu, hors contexte, ce qui a defile avant. La
baseline ``base`` EST donc "Qwen a contexte tronque" ; ``mem`` est "meme contexte
tronque + souvenir du passe". Contrairement a eval_lvb (sous-echantillonnage de
toute la video dans le contexte), ici la video peut depasser le contexte en temps :
on ne tient jamais que la fenetre, le reste vit dans la memoire.

Controle ``--memory-source mismatched`` : le passe d'une AUTRE video (meme cout,
meme nombre d'ecritures, mauvais contenu) -> isole ce que la memoire RETIENT du
contenu, pas l'apport de la branche en plus.

Score par logit de la lettre (comme eval_lvb), pas par generation.

    python -m titans.graft.eval_stream --data data/longvideobench \
        --ckpt runs/graft_usq_s8.pt --basis cache/qwen_l24_pca128.pt \
        --total-frames 128 --window 16 --duration-group 600 --limit 30 \
        --out results/graft/stream_600.json

Sans --ckpt : seule la fenetre tronquee est evaluee (le plancher a battre).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .eval_lvb import (_nullcm, build_prompt, donor_indices, format_report,
                       letter_logits, summarise, write_memory)
from .qwen import default_layer, disabled, find_decoder_layers, graft


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True, help="racine : lvb_val.json + videos/")
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--ckpt", default=None, help="sans lui : seul le contexte "
                                                 "tronque (plancher) est evalue")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--total-frames", type=int, default=128, dest="total_frames",
                    help="images echantillonnees sur TOUTE la video (fenetre + passe)")
    ap.add_argument("--window", type=int, default=16,
                    help="images RECENTES visibles par l'hote (contexte tronque)")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=None, dest="write_stride")
    ap.add_argument("--memory-source", default="video", dest="memory_source",
                    choices=["video", "mismatched"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--duration-group", type=int, default=None, dest="duration_group")
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from ..worldmodel.data import load_video_frames
    from .check_qwen import load_host
    from .fit_basis import resize_frames

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    data = json.load(open(os.path.join(a.data, a.split)))
    if a.duration_group:
        data = [d for d in data if d.get("duration_group") == a.duration_group]
    if a.limit:
        data = data[:a.limit]
    print(f"{len(data)} questions ; fenetre {a.window} / {a.total_frames} images",
          flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer

    branch = wrapper = None
    if a.ckpt:
        blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        args = blob.get("args", {})
        pca = (torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
               if a.basis else None)
        d_model = int(model.config.text_config.hidden_size)
        mem_kw = memory_kwargs_from_args(args)
        if a.write_stride is None:
            a.write_stride = int(args.get("write_stride", 2))
        branch = MemoryBranch(d_model, args.get("d_mem", 128), chunk_size=a.chunk,
                              down_pca=pca, **mem_kw).to(a.device)
        branch.load_state_dict(blob["branch"])
        branch.eval()
        wrapper = graft(model, branch, layer=site)
        wrapper.writes_enabled = False        # la memoire est ecrite hors contexte
        print(f"greffe {a.ckpt} (pas {blob.get('step')}) "
              f"muon={mem_kw['orthogonalize_momentum']} "
              f"pas-ecriture={a.write_stride} [PLOMBERIE : ckpt distillation, "
              f"pas la baseline scientifique]", flush=True)
    elif a.write_stride is None:
        a.write_stride = 1

    every = max(1, min(25, len(data) // 8))
    donors = donor_indices(data) if a.memory_source == "mismatched" else []
    if a.memory_source != "video":
        print(f"CONTROLE : memoire = passe d'une AUTRE video", flush=True)
    rows, t0 = [], time.time()
    for i, item in enumerate(data):
        path = os.path.join(a.data, "videos", item["video_path"])
        if not os.path.exists(path):
            continue
        n_opt = len(item["candidates"])
        prompt = build_prompt(item)
        dur = float(item["duration"])
        try:
            allf = resize_frames(load_video_frames(path, num_frames=a.total_frames),
                                 a.max_side)
            T = allf.shape[0]
            W = min(a.window, T)
            ctx = allf[-W:]                       # fenetre recente = contexte tronque
            fps = a.total_frames / dur            # cadence d'echantillonnage uniforme
            with (disabled(wrapper) if wrapper else _nullcm()):
                base = letter_logits(model, processor, ctx, prompt, n_opt, fps,
                                     a.device)
            row = {"id": item["id"], "duration_group": item.get("duration_group"),
                   "duration": dur, "gold": item["correct_choice"],
                   "base": int(base.argmax())}
            if wrapper is not None:
                if a.memory_source == "mismatched":
                    other = data[donors[i]]
                    of = resize_frames(load_video_frames(
                        os.path.join(a.data, "videos", other["video_path"]),
                        num_frames=a.total_frames), a.max_side)
                    past = of[:max(0, of.shape[0] - W)]      # passe d'une AUTRE video
                    fps_w = a.total_frames / float(other["duration"])
                    row["donor"] = other["video_path"]
                else:
                    past = allf[:max(0, T - W)]              # passe de CETTE video
                    fps_w = fps
                with disabled(wrapper):
                    state, S, nw = write_memory(branch, model, processor, past, "",
                                                fps_w, a.device, site, a.chunk,
                                                a.write_stride)
                wrapper.carry, wrapper.token_offset = (state, S), 0
                mem = letter_logits(model, processor, ctx, prompt, n_opt, fps,
                                    a.device)
                row["mem"] = int(mem.argmax())
                row["writes"] = nw
            rows.append(row)
        except Exception as e:
            print(f"  saut {item['id']}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % every == 0:
            acc_b = sum(r["base"] == r["gold"] for r in rows) / max(1, len(rows))
            msg = f"  [{i + 1}/{len(data)}] fenetre-seule {100 * acc_b:.1f} %"
            if wrapper is not None and rows and "mem" in rows[0]:
                acc_m = sum(r["mem"] == r["gold"] for r in rows) / len(rows)
                msg += f" | +memoire {100 * acc_m:.1f} %"
            print(msg + f"  ({(time.time() - t0) / (i + 1):.1f} s/q)", flush=True)

    report = summarise(rows)
    print("\n" + format_report(report))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "report": report, "rows": rows},
                  open(a.out, "w"), indent=1)
        print("json:", a.out)
    print("STREAM-EVAL-FINIE", flush=True)


if __name__ == "__main__":
    main()
