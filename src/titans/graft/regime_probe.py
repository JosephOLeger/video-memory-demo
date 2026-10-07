"""Phase 0 du plan greffe : y a-t-il un regime ou l'attention bornee echoue ?

La claim (E1) est que la memoire comble la taille bornee de l'attention. Elle n'a
de sens que s'il existe un regime ou l'hote SEUL, a budget d'images restreint,
echoue sur du contenu long. Ce probe le mesure, sans aucune memoire : pour chaque
tranche de duree de LongVideoBench et chaque budget d'images, la precision de
reponse de l'hote gele (score par logit de la lettre, exactement comme eval_lvb).

Lecture : si la precision est plate en duree et en images, il n'y a pas de travail
pour la memoire (et la these est en danger). Si elle chute avec la duree a petit
budget et remonte avec plus d'images, cet ecart est precisement ce que la memoire
doit recuperer en Phase 1 (meme budget d'images visible, + une memoire ecrite
depuis un echantillonnage dense).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m titans.graft.regime_probe \
        --data data/longvideobench --frames 8,16,32,64 \
        --groups 15,60,600,3600 --limit 60 --out results/graft/regime.json

Hote seul : aucun checkpoint, aucune greffe. Charge Qwen une seule fois.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True, help="racine : lvb_val.json + videos/")
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--frames", default="8,16,32,64",
                    help="budgets d'images DANS le contexte, balayes")
    ap.add_argument("--groups", default="15,60,600,3600",
                    help="tranches de duree a evaluer")
    ap.add_argument("--limit", type=int, default=60,
                    help="questions par tranche (les premieres)")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--guard-gb", type=float, default=9.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames_list = [int(x) for x in a.frames.split(",") if x]
    groups = [int(x) for x in a.groups.split(",") if x]

    from transformers import AutoProcessor

    from ..worldmodel.data import load_video_frames
    from .check_qwen import load_host
    from .eval_lvb import build_prompt, letter_logits
    from .fit_basis import resize_frames

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    data = json.load(open(os.path.join(a.data, a.split)))
    by_group = {g: [d for d in data if d.get("duration_group") == g][:a.limit]
                for g in groups}
    for g in groups:
        print(f"tranche {g:>4} s : {len(by_group[g])} questions", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)

    # Cache des images par (video, budget) : une meme question est scoree a
    # plusieurs budgets, mais les tranches ne partagent pas de video, donc le
    # cache ne sert qu'a ne pas re-decoder entre budgets voisins si on le voulait.
    rows = []
    t0 = time.time()
    done = 0
    total_q = sum(len(by_group[g]) for g in groups) * len(frames_list)
    for g in groups:
        items = by_group[g]
        for F in frames_list:
            ok = n = 0
            for item in items:
                path = os.path.join(a.data, "videos", item["video_path"])
                if not os.path.exists(path):
                    continue
                try:
                    n_opt = len(item["candidates"])
                    prompt = build_prompt(item)
                    dur = float(item["duration"])
                    ctx = resize_frames(load_video_frames(path, num_frames=F),
                                        a.max_side)
                    with torch.no_grad():
                        lg = letter_logits(model, processor, ctx, prompt, n_opt,
                                           F / dur, a.device)
                    ok += int(int(lg.argmax()) == item["correct_choice"])
                    n += 1
                except Exception as e:
                    print(f"  saut {item.get('id')}: {type(e).__name__}: {e}",
                          flush=True)
                done += 1
            acc = ok / max(1, n)
            # plancher de hasard moyen sur les questions scorees (nb d'options varie)
            chance = (sum(1.0 / len(it["candidates"]) for it in items[:n])
                      / max(1, n))
            rows.append({"group": g, "frames": F, "n": n, "acc": acc,
                         "chance": chance})
            print(f"  [{done}/{total_q}] tranche {g:>4} s  images {F:>3}  "
                  f"n={n:>3}  base={100 * acc:4.1f} %  (hasard {100 * chance:4.1f} %)  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    print("\n=== precision de l'hote seul (base), par tranche x budget d'images ===")
    hdr = "tranche(s) " + "".join(f"{F:>8}img" for F in frames_list)
    print(hdr)
    for g in groups:
        line = f"{g:>9}  "
        for F in frames_list:
            r = next((x for x in rows if x["group"] == g and x["frames"] == F), None)
            line += f"{100 * r['acc']:>8.1f}%" if r else f"{'--':>9}"
        print(line)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
        print("json:", a.out, flush=True)
    print("REGIME-PROBE-FINIE", flush=True)


if __name__ == "__main__":
    main()
