"""La distillation, sous les memes controles que LongVideoBench.

Pourquoi ce fichier existe
--------------------------
Sur LongVideoBench, ecrire la memoire depuis une AUTRE video ne change pas le
score : le gain de la greffe n'y vient pas de ce que la memoire retient. Restait
l'argument que la distillation, elle, serait specifique au contenu << par
construction >>, puisque la sonde interroge le debut de la fenetre que l'eleve ne
voit pas.

Cet argument n'a jamais ete teste, et il a exactement la forme de celui que les
controles LVB viennent d'invalider. Une branche peut apprendre la *distribution*
des reponses du professeur -- les tournures, les activites typiques d'EgoLife --
et faire baisser l'entropie croisee sans jamais lire la video. Cela produirait le
meme +28 %, avec zero specificite au contenu.

Les trois sources
-----------------
``context``     la vraie fenetre precedente (la mesure)
``mismatched``  une fenetre d'un AUTRE jour, tronquee au meme nombre de jetons,
                donc au meme nombre d'ecritures : memes parametres, meme cout,
                contenu faux
``none``        la branche est active mais rien n'est ecrit : le carry reste a
                l'init. Isole la projection entrainee de l'ecriture elle-meme.

Le plancher (``sans``) reste le meme dans les trois cas -- l'hote intact sur le
seul clip de l'eleve -- pour que les pourcentages soient comparables.

    python -m titans.graft.eval_distill --ckpt runs/g7_unit_s0.pt \
        --z cache/jake_day*_z.pt --basis cache/qwen_l24_pca128.pt \
        --memory-source mismatched
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .distill import (forced_inputs, forced_loss, load_window_frames,
                      split_windows, windows, write_context)
from .qwen import default_layer, find_decoder_layers, graft


def donor_span(days: list, d: int, tok_lo: int, tok_hi: int) -> tuple:
    """A token range on a DIFFERENT day, of exactly the same length.

    Same length is the point: the memory's state grows with how often it is
    written, so a donor of a different size would confound "wrong content" with
    "less writing". Deterministic -- the next day round-robin, offset to the middle
    of it -- so a rerun gives the same pairing.
    """
    n = len(days)
    d2 = (d + 1) % n
    span = tok_hi - tok_lo
    total = int(days[d2]["z"].shape[0])
    lo2 = max(0, min(total - span, (total - span) // 2))
    return d2, lo2, lo2 + span


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--z", required=True, nargs="+")
    ap.add_argument("--targets", default=None, nargs="+")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--memory-source", default="context", dest="memory_source",
                    choices=["context", "mismatched", "none"])
    ap.add_argument("--student-view", default="clip", dest="student_view",
                    choices=["clip", "none", "mismatched"],
                    help="ce que l'ELEVE voit, avant toute memoire. clip : son "
                         "dernier clip (la mesure). none : rien du tout -- lire "
                         "alors la colonne 'sans', c'est l'a priori de langage "
                         "pur. mismatched : un clip d'un AUTRE jour, ce qui "
                         "separe 'voir une video' de 'voir LA video'.")
    ap.add_argument("--eval-n", type=int, default=0, dest="eval_n",
                    help="0 = toute la validation")
    ap.add_argument("--guard-gb", type=float, default=8.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = blob.get("args", {})
    span, hop = int(targs.get("span", 12)), int(targs.get("hop", 4))
    stride = int(targs.get("write_stride", 2))
    probe = targs.get("probe")
    sclips = int(targs.get("student_clips", 1))

    days, wins = [], []
    for d, p in enumerate(a.z):
        b = torch.load(p, map_location="cpu", weights_only=False)
        days.append({"z": b["z"], "offsets": b["offsets"], "paths": b["clips"],
                     "fpc": b["frames_per_clip"], "max_side": b["max_side"],
                     "fps": b["frames_per_clip"] / 30.0})
        n = len(b["offsets"]) - 1
        wins += [(d, lo, hi) for lo, hi in windows(n, span, hop)]

    tgt_paths = a.targets or [p.replace("_z.pt", "_tgt.pt") for p in a.z]
    tgt = {}
    for d, tp in enumerate(tgt_paths):
        tb = torch.load(tp, map_location="cpu", weights_only=False)
        for k, v in tb["targets"].items():
            lo, hi = (int(x) for x in k.split("-"))
            tgt[(d, lo, hi)] = v

    embargo = max(0, math.ceil(span / hop) - 1)
    _, val_w = split_windows(wins, val_frac=0.1, embargo=embargo)
    val_w = [w for w in val_w if w in tgt]
    if a.eval_n:
        val_w = val_w[:a.eval_n]
    print(f"{len(val_w)} fenetres de validation, source memoire : "
          f"{a.memory_source}, vue de l'eleve : {a.student_view}", flush=True)
    if a.memory_source != "context":
        print("CONTROLE -- ce n'est pas la mesure", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model)))
    pca = (torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
           if a.basis else None)
    branch = MemoryBranch(int(model.config.text_config.hidden_size),
                          targs.get("d_mem", 128), chunk_size=targs.get("chunk", 8),
                          write_stride=1, down_pca=pca,
                          **memory_kwargs_from_args(targs)).to(a.device)
    branch.load_state_dict(blob["branch"])
    branch.eval()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    print(f"greffe {a.ckpt} (pas {blob.get('step')}, val {blob.get('val_mem'):.4f}) "
          f"muon={targs.get('muon')}", flush=True)

    rows, t0 = [], time.time()
    for i, win in enumerate(val_w):
        d, lo, hi = win
        day = days[d]
        y = tgt[win]
        s_lo = hi - sclips
        if a.student_view == "none":
            frames = None
        elif a.student_view == "mismatched":
            d2 = (d + 1) % len(days)
            n2 = len(days[d2]["paths"])
            c2 = min(n2 - sclips, int(s_lo * n2 / len(day["paths"])))
            frames = load_window_frames(days[d2]["paths"], max(0, c2),
                                        max(0, c2) + sclips, days[d2]["fpc"],
                                        days[d2]["max_side"])
        else:
            frames = load_window_frames(day["paths"], s_lo, hi, day["fpc"],
                                        day["max_side"])
        inputs, n_prompt = forced_inputs(processor, frames, probe, y, day["fps"],
                                         a.device)
        tok_lo, tok_hi = day["offsets"][lo], day["offsets"][s_lo]
        wrapper.reset()
        with torch.no_grad():
            if a.memory_source == "none":
                state, S = branch.init_carry(1)
                nw = 0
            elif a.memory_source == "mismatched":
                d2, l2, h2 = donor_span(days, d, tok_lo, tok_hi)
                state, S, nw = write_context(branch, days[d2]["z"], l2, h2,
                                             stride, a.device)
            else:
                state, S, nw = write_context(branch, day["z"], tok_lo, tok_hi,
                                             stride, a.device)
            wrapper.carry, wrapper.token_offset = (state, S), 0
            mem = float(forced_loss(model, inputs, n_prompt, y))
            wrapper.enabled = False
            short = float(forced_loss(model, inputs, n_prompt, y))
            wrapper.enabled = True
        rows.append({"win": list(win), "mem": mem, "short": short, "writes": nw})
        if (i + 1) % max(1, len(val_w) // 8) == 0:
            m = sum(r["mem"] for r in rows) / len(rows)
            s = sum(r["short"] for r in rows) / len(rows)
            print(f"  [{i + 1}/{len(val_w)}] memoire {m:.4f} | sans {s:.4f} | "
                  f"recupere {100 * (s - m) / s:+.1f} %  "
                  f"({(time.time() - t0) / (i + 1):.1f} s/fenetre)", flush=True)

    m = sum(r["mem"] for r in rows) / len(rows)
    s = sum(r["short"] for r in rows) / len(rows)
    better = sum(r["mem"] < r["short"] for r in rows)
    print(f"\nsource {a.memory_source} : memoire {m:.4f} | sans {s:.4f} | "
          f"recupere {100 * (s - m) / s:+.1f} % | "
          f"{better}/{len(rows)} fenetres ou la memoire aide | "
          f"ecritures {sum(r['writes'] for r in rows) / len(rows):.0f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "mem": m, "short": s, "rows": rows},
                  open(a.out, "w"), indent=1)
        print("json:", a.out)


if __name__ == "__main__":
    main()
