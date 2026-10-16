"""Charge : rappel associatif multi-paires (le MQAR du papier, sur un vrai VLM).

Le jalon couleur (``train_color``) est sature : une seule aiguille, ``on`` = 100 %.
Il ne dit rien de la CAPACITE. Ici le passe contient ``K`` paires cle->valeur :
chaque paire = ``--pair-frames`` images d'un aplat de couleur (la valeur) portant un
nombre a deux chiffres en gros (la cle). La question nomme UNE cle :

    "What was the background color of the frame showing the number 37?"

Le hasard reste 1/8 quel que soit ``K``, donc la courbe ``on(K)`` mesure directement
combien d'associations la memoire tient. C'est l'analogue VLM du MQAR
([[mqar-harness-exists]], [[echo-recall-load-dependent-0929]]) : a forte charge, une
memoire diverse devrait battre une memoire effondree. D'ou les deux bras a comparer,
``--muon unit`` (s'effondre sur les longs flux, pr 3-10) et ``--muon unit_squash``
(reste divers, pr 31-37) ([[long-probe-corrects-step0-0929]]).

Controles, comme le banc aiguille :
  off   : fenetre seule                                   -> hasard
  on    : fenetre + memoire du passe aux K paires         -> la MESURE
  wrong : memes cles, memes positions, couleurs retirees,
          la cle interrogee ayant une AUTRE couleur        -> hasard (ou 0 si la
          reponse suit le contenu de la memoire)

Ce que ca teste en plus du jalon couleur : la lecture doit etre ADRESSEE par la cle
(le nombre lu dans la question), pas seulement "la couleur saillante du passe". Avec
``K = 1`` on retombe sur le jalon couleur, la cle en plus.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m titans.graft.train_pairs \\
        --dataset finevideo --data data/finevideo --dur-min 400 --dur-max 800 \\
        --basis cache/qwen_l24_pca128.pt --write-stride 8 --accum 8 --lr 3e-3 \\
        --muon unit --train-k 1,2,4,8,16 --eval-k 1,4,16 \\
        --steps 4800 --eval-every 600 --out runs/pairs_unit_s0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .branch import MemoryBranch, memory_kwargs_from_args, add_memory_flags
from .needle import COLORS, LETTERS
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_color import (answer_logits, host_states, load_split, video_pool,
                          write_grad)

#: Cles : nombres a deux chiffres, assez pour K = 32 sans repetition.
KEYS = list(range(10, 100))
#: Couleurs claires -> chiffre noir, sinon blanc (lisibilite du nombre).
LIGHT = {"yellow", "orange", "white"}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def pairs_prompt(key: int) -> str:
    lines = ["Several frames in the video showed a number on a solid colored "
             f"background. What was the background color of the frame showing "
             f"the number {key}?"]
    lines += [f"{LETTERS[i]}. {name}" for i, (name, _) in enumerate(COLORS)]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


_FONTS: dict = {}


def pair_frames(key: int, ci: int, nf: int, h: int, w: int) -> torch.Tensor:
    """``[nf, h, w, 3]`` uint8 : aplat ``COLORS[ci]`` portant le nombre ``key``."""
    from PIL import Image, ImageDraw, ImageFont
    size = int(0.6 * min(h, w))
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.truetype(FONT_PATH, size)
        except OSError:
            _FONTS[size] = ImageFont.load_default(size=size)
    name, rgb = COLORS[ci]
    img = Image.new("RGB", (w, h), rgb)
    d = ImageDraw.Draw(img)
    fg = (0, 0, 0) if name in LIGHT else (255, 255, 255)
    d.text((w / 2, h / 2), str(key), fill=fg, font=_FONTS[size], anchor="mm")
    f = torch.from_numpy(np.asarray(img).copy())
    return f.unsqueeze(0).expand(nf, h, w, 3).contiguous()


def spliced_pairs(past: torch.Tensor, keys, colors, positions, nf: int):
    """Le passe avec chaque paire inseree a sa position (indices du passe d'origine)."""
    _, h, w, _ = past.shape
    out, prev = [], 0
    for pos, k, c in sorted(zip(positions, keys, colors)):
        out += [past[prev:pos], pair_frames(k, c, nf, h, w)]
        prev = pos
    out.append(past[prev:])
    return torch.cat(out, 0)


def draw_sample(rng: random.Random, K: int, P: int, gold: int | None = None):
    """Cles, couleurs, positions, indice interroge, et couleurs du controle ``wrong``.

    Positions paires (alignement sur le patch temporel de 2 images de Qwen2.5-VL).
    ``wrong`` : couleurs retirees, celle de la cle interrogee decalee de 4 (jamais la
    bonne).
    """
    keys = rng.sample(KEYS, K)
    colors = [rng.randrange(len(COLORS)) for _ in range(K)]
    q = rng.randrange(K)
    if gold is not None:
        colors[q] = gold
    slots = list(range(0, P + 1, 2))
    positions = rng.sample(slots, K) if K <= len(slots) else [rng.choice(slots)
                                                              for _ in range(K)]
    wcolors = [rng.randrange(len(COLORS)) for _ in range(K)]
    wcolors[q] = (colors[q] + len(COLORS) // 2) % len(COLORS)
    return keys, colors, positions, q, wcolors


@torch.no_grad()
def pairs_eval(model, processor, branch, wrapper, vids, cache, ks, reps, site,
               a, letter_ids) -> dict:
    branch.eval()
    rng = random.Random(1234)                     # memes echantillons a chaque eval

    def pred(frames, fps, prompt):
        return int(answer_logits(model, processor, frames, prompt, fps, a.device,
                                 letter_ids).argmax())

    rows, s = [], 0
    for K in ks:
        for vid in vids:
            try:
                win, past, fps = load_split(vid, cache, a.total_frames, a.window,
                                            a.max_side)
            except Exception:
                continue
            for _ in range(reps):
                gold = s % len(COLORS)
                keys, cols, pos, q, wcols = draw_sample(rng, K, past.shape[0], gold)
                prompt = pairs_prompt(keys[q])
                res = {}
                with disabled(wrapper):
                    res["off"] = pred(win, fps, prompt)
                for tag, cc in (("on", cols), ("wrong", wcols)):
                    pr = spliced_pairs(past, keys, cc, pos, a.pair_frames)
                    with disabled(wrapper):
                        st = host_states(model, processor, pr, fps, a.device, site)
                    state, S, _ = write_grad(branch, st, a.chunk, a.write_stride)
                    wrapper.carry, wrapper.token_offset = (state, S), 0
                    res[tag] = pred(win, fps, prompt)
                rows.append({"K": K, "gold": gold, **res})
                s += 1
    branch.train()
    out = {}
    for K in ks:
        r = [x for x in rows if x["K"] == K]
        if r:
            out[str(K)] = {"n": len(r), **{c: sum(x[c] == x["gold"] for x in r) / len(r)
                                           for c in ("off", "on", "wrong")}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--dataset", default="lvb", choices=["lvb", "finevideo"])
    ap.add_argument("--dur-min", type=float, default=None, dest="dur_min")
    ap.add_argument("--dur-max", type=float, default=None, dest="dur_max")
    ap.add_argument("--duration-group", type=int, default=600, dest="duration_group")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--init-ckpt", default=None, dest="init_ckpt")
    ap.add_argument("--layer", type=int, default=None)
    # --- tache ---
    ap.add_argument("--total-frames", type=int, default=128, dest="total_frames")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--pair-frames", type=int, default=4, dest="pair_frames")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--eval-videos", type=int, default=12, dest="eval_videos")
    ap.add_argument("--max-train-videos", type=int, default=60, dest="max_train_videos")
    ap.add_argument("--train-k", default="1,2,4,8,16", dest="train_k",
                    help="charges tirees uniformement a l'entrainement")
    ap.add_argument("--eval-k", default="1,4,16", dest="eval_k")
    ap.add_argument("--curriculum", action="store_true",
                    help="K croissant : on ne passe au palier suivant de --train-k que "
                         "quand la precision d'entrainement au palier courant depasse "
                         "--advance-acc (sur les --advance-window derniers echantillons)")
    ap.add_argument("--advance-acc", type=float, default=0.6, dest="advance_acc")
    ap.add_argument("--advance-window", type=int, default=160, dest="advance_window")
    ap.add_argument("--min-stage", type=int, default=200, dest="min_stage",
                    help="pas minimum par palier")
    ap.add_argument("--eval-reps", type=int, default=3, dest="eval_reps",
                    help="echantillons par video et par K a l'eval")
    # --- memoire ---
    add_memory_flags(ap, write_stride=8)
    # --- optimisation ---
    ap.add_argument("--steps", type=int, default=4800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=600, dest="eval_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train_k = [int(x) for x in a.train_k.split(",") if x]
    eval_k = [int(x) for x in a.eval_k.split(",") if x]
    assert max(train_k + eval_k) <= len(KEYS)

    from transformers import AutoProcessor

    from .check_qwen import load_host

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    dur_band = ((a.dur_min, a.dur_max) if a.dataset == "finevideo"
                and a.dur_min is not None and a.dur_max is not None else None)
    pool = video_pool(a.data, a.split, a.duration_group,
                      a.eval_videos + a.max_train_videos, a.dataset, dur_band)
    if len(pool) <= a.eval_videos:
        raise SystemExit(f"seulement {len(pool)} videos")
    eval_vids, train_vids = pool[:a.eval_videos], pool[a.eval_videos:]
    print(f"{len(train_vids)} videos d'entrainement, {len(eval_vids)} d'eval "
          f"(disjointes) ; K entrainement {train_k}, eval {eval_k}", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    mem_kw = memory_kwargs_from_args(a)
    branch = MemoryBranch(int(model.config.text_config.hidden_size), a.d_mem,
                          chunk_size=a.chunk, write_stride=a.write_stride,
                          down_pca=pca, **mem_kw).to(a.device)
    if a.init_ckpt:
        branch.load_state_dict(torch.load(a.init_ckpt, map_location="cpu",
                                          weights_only=False)["branch"])
        print(f"branche initialisee depuis {a.init_ckpt}", flush=True)
    branch.train()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    print(f"greffe couche {site} : {sum(p.numel() for p in params)} params, "
          f"identite a l'init : {branch.is_identity()} "
          f"muon={mem_kw['orthogonalize_momentum']}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    letter_ids = torch.tensor(
        [processor.tokenizer.encode(LETTERS[i], add_special_tokens=False)[0]
         for i in range(len(COLORS))], device=a.device)

    # Curriculum : palier courant = indice dans train_k trie. La moitie des
    # echantillons au palier courant (le signal neuf), l'autre moitie sur les paliers
    # deja acquis (pour ne pas les oublier).
    train_k = sorted(train_k)
    level = 0 if a.curriculum else len(train_k) - 1
    stage_start, recent = 0, []

    def draw_k():
        if not a.curriculum or level == 0 or rng.random() < 0.5:
            return train_k[level] if a.curriculum else rng.choice(train_k)
        return rng.choice(train_k[:level])

    frame_cache, hist, best, t0 = {}, [], -1.0, time.time()
    micro, last_gn = 0, float("nan")
    opt.zero_grad(set_to_none=True)
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        vid = train_vids[rng.randrange(len(train_vids))]
        K = draw_k()
        try:
            win, past, fps = load_split(vid, frame_cache, a.total_frames, a.window,
                                        a.max_side)
            keys, cols, pos, q, _ = draw_sample(rng, K, past.shape[0])
            pr = spliced_pairs(past, keys, cols, pos, a.pair_frames)
            with disabled(wrapper):
                st = host_states(model, processor, pr, fps, a.device, site)
            state, S, nw = write_grad(branch, st, a.chunk, a.write_stride)
            wrapper.carry, wrapper.token_offset = (state, S), 0
            cls = answer_logits(model, processor, win, pairs_prompt(keys[q]), fps,
                                a.device, letter_ids)
            loss = F.cross_entropy(cls.unsqueeze(0),
                                   torch.tensor([cols[q]], device=a.device))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  pas {step} saute ({vid['name']}): {type(e).__name__}: {e}",
                  flush=True)
            continue
        if a.curriculum and K == train_k[level]:
            recent = (recent + [int(int(cls.argmax()) == cols[q])])[-a.advance_window:]
            if (level < len(train_k) - 1 and step - stage_start >= a.min_stage
                    and len(recent) == a.advance_window
                    and sum(recent) / len(recent) >= a.advance_acc):
                level, stage_start, recent = level + 1, step, []
                hist.append({"step": step, "curriculum_K": train_k[level]})
                print(f"  ** palier : K max = {train_k[level]} (pas {step})", flush=True)
        (loss / a.accum).backward()
        micro += 1
        if micro % a.accum == 0:
            last_gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            hit = int(cls.argmax()) == cols[q]
            print(f"  {step:5d}  K {K:2d}/{train_k[level]:2d}  perte {float(loss):.3f}  "
                  f"{'bon' if hit else '.':>3}  |g| {last_gn:.2f}  ecritures {nw}  "
                  f"{(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)

        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ev = pairs_eval(model, processor, branch, wrapper, eval_vids, frame_cache,
                            eval_k, a.eval_reps, site, a, letter_ids)
            hist.append({"step": step + 1, "by_K": ev, "curriculum_K": train_k[level]})
            print(f"  == eval {step + 1}: " + "  ".join(
                f"K={k}: on {100*v['on']:.0f} wr {100*v['wrong']:.0f} "
                f"off {100*v['off']:.0f}" for k, v in ev.items())
                + f"  (n/K={next(iter(ev.values()))['n']})", flush=True)
            score = sum(v["on"] for v in ev.values()) / max(1, len(ev))
            if score > best and a.out:
                best = score
                torch.save({"branch": branch.state_dict(), "step": step + 1,
                            "eval": ev, "args": vars(a)}, a.out + ".pt")
            if a.out:
                with open(a.out + ".json", "w") as f:
                    json.dump({"args": vars(a), "history": hist,
                               "elapsed_s": time.time() - t0}, f, indent=1)
    print("TRAIN-PAIRS-FINIE", flush=True)


if __name__ == "__main__":
    main()
