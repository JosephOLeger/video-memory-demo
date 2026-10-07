"""Banc aiguille : la memoire recupere-t-elle un contenu passe precis ?

NIAH (needle-in-a-haystack) adapte a la video. On fabrique une "video" = une vraie
video longue (distracteurs) + une AIGUILLE monochrome inseree a une profondeur
controlee : ``--needle-frames`` images d'un aplat de couleur. La question est un QCM
ferme sur 8 couleurs. L'hote ne voit que la fenetre recente, donc l'aiguille est
dans le passe : il ne peut pas repondre, seule la memoire le peut.

Quatre conditions par echantillon (le metre-etalon) :
  off      : fenetre recente seule, memoire eteinte         -> PLANCHER (~hasard)
  ceiling  : fenetre contenant l'aiguille, memoire eteinte  -> PLAFOND  (~100 %)
  on       : fenetre + memoire du passe (qui contient l'aiguille) -> la MESURE
  wrong    : fenetre + memoire d'un passe a une AUTRE couleur -> CONTROLE (~hasard)

on >> wrong ~= off ~= hasard, avec ceiling haut : "la memoire recupere un contenu
passe precis" est demontre proprement ([[lvb-gain-is-not-memory-content]]). La bonne
couleur est equilibree (cycle sur la palette) pour qu'un a priori donne pile le
hasard. On balaye la PROFONDEUR de l'aiguille -> courbe rappel vs distance.

Lance avec le vieux checkpoint = validation du metre-etalon : off/ceiling/wrong
atterrissent ou ils doivent, et on=hasard tant que la Phase 2 n'a pas entraine la
lecture memoire.

    python -m titans.graft.needle --data data/longvideobench \
        --ckpt runs/graft_usq_s8.pt --basis cache/qwen_l24_pca128.pt \
        --duration-group 600 --num-videos 8 --depths 0.1,0.5,0.9 \
        --total-frames 128 --window 16 --needle-frames 4 \
        --out results/graft/needle.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .eval_lvb import letter_logits, write_memory
from .qwen import default_layer, disabled, find_decoder_layers, graft

#: 8 couleurs nommables et visuellement distinctes. L'ordre EST l'ordre des options
#: (red=A, ..., black=H), donc equilibrer la couleur-aiguille equilibre la lettre.
COLORS = [("red", (255, 0, 0)), ("green", (0, 128, 0)), ("blue", (0, 0, 255)),
          ("yellow", (255, 255, 0)), ("orange", (255, 165, 0)),
          ("purple", (128, 0, 128)), ("white", (255, 255, 255)),
          ("black", (0, 0, 0))]
LETTERS = "ABCDEFGH"


def build_prompt() -> str:
    lines = ["A frame of a single solid color was shown at one point in the video. "
             "What color was it?"]
    lines += [f"{LETTERS[i]}. {name}" for i, (name, _) in enumerate(COLORS)]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def make_needle(k: int, h: int, w: int, rgb) -> torch.Tensor:
    """``[k, h, w, 3]`` uint8 : ``k`` images d'un aplat de couleur ``rgb``."""
    f = torch.zeros(k, h, w, 3, dtype=torch.uint8)
    for c in range(3):
        f[..., c] = rgb[c]
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True, help="racine : lvb_val.json + videos/")
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--basis", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--total-frames", type=int, default=128, dest="total_frames")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--needle-frames", type=int, default=4, dest="needle_frames")
    ap.add_argument("--depths", default="0.1,0.5,0.9",
                    help="fraction du passe (0 = debut/profond, 1 = juste avant la fenetre)")
    ap.add_argument("--num-videos", type=int, default=8, dest="num_videos")
    ap.add_argument("--duration-group", type=int, default=600, dest="duration_group")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=1, dest="write_stride",
                    help="1 par defaut : l'aiguille doit etre ecrite (D5)")
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    depths = [float(x) for x in a.depths.split(",") if x]

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
    # une question par video suffit : on ne se sert que de la video comme distracteur
    seen, vids = set(), []
    for d in data:
        if d["video_path"] not in seen:
            seen.add(d["video_path"])
            vids.append(d)
        if len(vids) >= a.num_videos:
            break
    print(f"{len(vids)} videos distracteurs (tranche {a.duration_group} s), "
          f"profondeurs {depths}", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer

    if not a.ckpt:
        raise SystemExit("--ckpt requis (il faut une branche pour on/wrong)")
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    pca = (torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
           if a.basis else None)
    d_model = int(model.config.text_config.hidden_size)
    mem_kw = memory_kwargs_from_args(args)
    branch = MemoryBranch(d_model, args.get("d_mem", 128), chunk_size=a.chunk,
                          down_pca=pca, **mem_kw).to(a.device)
    branch.load_state_dict(blob["branch"])
    branch.eval()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    print(f"greffe {a.ckpt} muon={mem_kw['orthogonalize_momentum']} "
          f"[on=hasard attendu tant que non entraine]", flush=True)

    prompt = build_prompt()
    nf = a.needle_frames
    rows, s, t0 = [], 0, time.time()

    @torch.no_grad()
    def answer(frames) -> int:
        return int(letter_logits(model, processor, frames, prompt, len(COLORS),
                                 fps, a.device).argmax())

    for item in vids:
        path = os.path.join(a.data, "videos", item["video_path"])
        if not os.path.exists(path):
            continue
        dur = float(item["duration"])
        try:
            allf = resize_frames(load_video_frames(path, num_frames=a.total_frames),
                                 a.max_side)
        except Exception as e:
            print(f"  saut {item['video_path']}: {type(e).__name__}: {e}", flush=True)
            continue
        T, h, w, _ = allf.shape
        W = min(a.window, T)
        P = T - W
        windowf = allf[-W:]
        pastf = allf[:P]
        fps = a.total_frames / dur
        for frac in depths:
            ci, wi = s % len(COLORS), (s + len(COLORS) // 2) % len(COLORS)
            _, rgb = COLORS[ci]
            _, wrgb = COLORS[wi]
            pos = min(P, max(0, int(round(frac * P))))
            needle = make_needle(nf, h, w, rgb)
            wneedle = make_needle(nf, h, w, wrgb)
            past_r = torch.cat([pastf[:pos], needle, pastf[pos:]], 0)
            past_w = torch.cat([pastf[:pos], wneedle, pastf[pos:]], 0)
            try:
                with disabled(wrapper):
                    off = answer(windowf)
                    ceil = answer(torch.cat([windowf[:-nf], needle], 0))
                with disabled(wrapper):
                    st, S, nw = write_memory(branch, model, processor, past_r, "",
                                             fps, a.device, site, a.chunk,
                                             a.write_stride)
                wrapper.carry, wrapper.token_offset = (st, S), 0
                on = answer(windowf)
                with disabled(wrapper):
                    st, S, _ = write_memory(branch, model, processor, past_w, "",
                                            fps, a.device, site, a.chunk,
                                            a.write_stride)
                wrapper.carry, wrapper.token_offset = (st, S), 0
                wrong = answer(windowf)
            except Exception as e:
                print(f"  saut {item['video_path']} f={frac}: "
                      f"{type(e).__name__}: {e}", flush=True)
                s += 1
                continue
            rows.append({"video": item["video_path"], "frac": frac,
                         "dist_frames": P - pos, "gold": ci, "writes": nw,
                         "off": off, "ceiling": ceil, "on": on, "wrong": wrong})
            s += 1
        if rows:
            acc = lambda k: sum(r[k] == r["gold"] for r in rows) / len(rows)
            print(f"  [{len(rows)}] off {100*acc('off'):.0f} plafond "
                  f"{100*acc('ceiling'):.0f} on {100*acc('on'):.0f} mauvais "
                  f"{100*acc('wrong'):.0f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n=== banc aiguille (hasard = 12.5 %) ===")
    print(summarise(rows, depths))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
        print("json:", a.out)
    print("NEEDLE-FINIE", flush=True)


def summarise(rows: list, depths: list) -> str:
    if not rows:
        return "(aucun echantillon)"
    def acc(sel, k):
        return 100 * sum(r[k] == r["gold"] for r in sel) / len(sel) if sel else 0.0
    head = f"{'profondeur':>11}{'n':>5}{'off':>8}{'plafond':>9}{'on':>7}{'mauvais':>9}"
    lines = [head]
    for key, sel in ([("tout", rows)]
                     + [(f"f={d}", [r for r in rows if r["frac"] == d]) for d in depths]):
        if not sel:
            continue
        lines.append(f"{key:>11}{len(sel):>5}{acc(sel,'off'):>7.0f}%"
                     f"{acc(sel,'ceiling'):>8.0f}%{acc(sel,'on'):>6.0f}%"
                     f"{acc(sel,'wrong'):>8.0f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
