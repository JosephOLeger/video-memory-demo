"""Jalon couleur : apprendre a la memoire a rappeler une couleur sortie du contexte.

Le metre-etalon (``needle.py``) est pose : ``ceiling`` ~96 %, ``off``/``wrong`` au
hasard, ``on`` au hasard tant que la lecture memoire n'est pas entrainee. Ce module
ferme ce dernier ecart. C'est le JALON intermediaire de la Phase 2
([[graft-sliding-window-needle]]) : une tache auto-generee, a reservoir de donnees
infini, qui verifie de bout en bout que la retropropagation-a-travers-l'ecriture
sait faconner un contenu rappelable, AVANT de depenser du calcul sur la tache
generale.

Ce qui est entraine, et pourquoi ca suffit
------------------------------------------
L'hote est gele. ``down`` est la base PCA, gelee elle aussi. Restent entrainables :
les cartes cle/valeur et les portes de la boucle interne (via ``memory.update``),
la requete ``q_proj``, la norme de lecture et surtout ``up`` -- le chemin qui ecrit
dans le flux residuel. C'est exactement la nuance du curriculum : la boucle interne
STOCKE deja k->v toute seule ; l'entrainement exterieur faconne ces projections
pour que le contenu stocke soit RAPPELABLE par la tache ([[graft-sliding-window-needle]]).

distill.py s'inquietait qu'une lettre = ~1.4 nats ne suffise pas a entrainer un
demi-million de parametres : c'etait vrai parce qu'EgoLifeQA n'a que 101 questions.
Ici le reservoir est infini -- n'importe quelle video x n'importe quelle couleur x
n'importe quelle profondeur -- donc le signal total est ``pas x log(8)`` et la
faiblesse par exemple ne mord pas.

La perte est le "classifieur" QCM
---------------------------------
On lit les 8 logits de lettre a la premiere position de reponse (ce que
``eval_lvb.letter_logits`` mesure) et on fait une entropie croisee a 8 voies contre
la bonne couleur. La perte entraine donc pile la quantite que le banc evalue, sans
dependre du decodage. L'ecriture (le passe + l'aiguille) est construite a la main,
graphe garde ; la reponse lit la memoire (``writes_enabled=False``) -- le graphe
BPTT couvre update->retrieve, la profondeur que le micro-run a montree tenable.

La tete de langage reste GELEE par defaut (``--train-head none``)
----------------------------------------------------------------
On commence par isoler une question : la memoire peut-elle encoder une couleur que
la tete DEJA EN PLACE sait lire ? Si ``on`` monte vers ``ceiling`` sans toucher
l'hote, le chemin memoire marche de bout en bout. Sinon, ``--train-head full``
degele la tete (ce que Josep veut pour la tache generale : qu'elle apprenne a faire
confiance au latent ajoute) -- mais alors l'evaluation externe doit charger la tete
entrainee, pas seulement la branche.

Les videos d'entrainement sont DISJOINTES des videos d'evaluation (``--eval-videos``
en tete de la tranche, entrainement sur le reste) : le banc mesure une competence
generale de lecture memoire, pas la memorisation d'un flux ([[embargo-at-the-seam]]).

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python -m titans.graft.train_color \\
        --data data/longvideobench --duration-group 600 \\
        --basis cache/qwen_l24_pca128.pt \\
        --eval-videos 6 --max-train-videos 40 \\
        --total-frames 128 --window 16 --needle-frames 4 \\
        --steps 1500 --eval-every 150 --out runs/color_unit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .branch import MemoryBranch, memory_kwargs_from_args
from .needle import COLORS, build_prompt, make_needle
from .qwen import (default_layer, disabled, find_decoder_layers, graft)


# --------------------------------------------------------------------------- #
# passes
# --------------------------------------------------------------------------- #
def host_states(model, processor, frames, fps: float, device: str,
                layer: int) -> torch.Tensor:
    """Sortie couche-L de l'hote gele sur ``frames``, detachee. SANS graphe.

    Doit tourner degreffe (l'appelant est dans ``disabled(wrapper)``) : la branche
    lit cette activation, qui ne depend pas d'elle, donc la faire tourner greffee
    avancerait la recurrence deux fois. L'hote etant gele, on detache ici -- seules
    les cartes de la memoire, en aval, portent un gradient.
    """
    from .fit_basis import video_inputs
    layers = find_decoder_layers(model)
    got = []

    def hook(_m, _i, out):
        got.append((out[0] if isinstance(out, tuple) else out).detach())

    h = layers[layer].register_forward_hook(hook)
    try:
        inp = video_inputs(processor, frames, "", fps)
        inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            model(**inp, use_cache=False)
    finally:
        h.remove()
    return got[0].float()                                     # [1, N_tok, d_model]


def write_grad(branch: MemoryBranch, states: torch.Tensor, chunk: int, stride: int):
    """Ecrit ``states`` dans la memoire EN GARDANT le graphe.

    ``down`` est gelee (PCA), donc ``z`` est une constante ; mais ``memory.update``
    applique ses cartes cle/valeur et ses portes entrainables, et c'est par la que
    le gradient entre. On construit l'etat a la main, comme ``distill.write_context``,
    puis la passe de reponse le lira via le forward greffe.
    """
    z = branch.down(states)[:, ::stride]                     # [1, n_w, d_mem]
    state, S = branch.init_carry(1)
    if z.shape[1]:
        state, S = branch.memory.update(z, state, S, min(chunk, z.shape[1]))
    return state, S, int(z.shape[1])


def answer_logits(model, processor, frames, prompt: str, fps: float, device: str,
                  letter_ids: torch.Tensor) -> torch.Tensor:
    """Les ``n_opt`` logits de lettre a la premiere position de reponse, AVEC graphe.

    Memes position et tokens que ``eval_lvb.letter_logits`` (``logits[0, -1]``), donc
    l'entropie croisee entraine exactement ce que le banc evalue. Le forward est
    greffe (``enabled=True, writes_enabled=False``) : la branche LIT l'etat que
    ``write_grad`` a construit, sans rien ecrire.
    """
    from .fit_basis import video_inputs
    inp = video_inputs(processor, frames, prompt, fps)
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    logits = model(**inp, use_cache=False).logits[0, -1].float()
    return logits[letter_ids]


# --------------------------------------------------------------------------- #
# donnees
# --------------------------------------------------------------------------- #
def video_pool(data_root: str, split: str, duration_group, n_want: int,
               dataset: str = "lvb", dur_band=None) -> list:
    """Videos distinctes (une question/video suffit : la video n'est qu'un distracteur).

    ``dataset='lvb'`` : ``<split>`` (json) + ``videos/<video_path>``, filtre
    ``duration_group``. ``dataset='finevideo'`` : ``manifest.jsonl`` (cles ``id`` /
    ``duration``) + ``videos/<id>.mp4``, filtre optionnel ``dur_band=(min,max)`` en s.
    """
    seen, vids = set(), []
    if dataset == "finevideo":
        with open(os.path.join(data_root, "manifest.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                vid = d["id"]
                try:
                    dur = float(d.get("duration") or d["meta"]["duration_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if dur_band and not (dur_band[0] <= dur <= dur_band[1]):
                    continue
                p = os.path.join(data_root, "videos", vid + ".mp4")
                if vid in seen or not os.path.exists(p):
                    continue
                seen.add(vid)
                vids.append({"path": p, "name": vid + ".mp4", "dur": dur})
                if len(vids) >= n_want:
                    break
        return vids
    data = json.load(open(os.path.join(data_root, split)))
    if duration_group:
        data = [d for d in data if d.get("duration_group") == duration_group]
    for d in data:
        p = os.path.join(data_root, "videos", d["video_path"])
        if d["video_path"] in seen or not os.path.exists(p):
            continue
        seen.add(d["video_path"])
        vids.append({"path": p, "name": d["video_path"], "dur": float(d["duration"])})
        if len(vids) >= n_want:
            break
    return vids


def load_split(vid: dict, cache: dict, total_frames: int, window: int, max_side: int):
    """``(window_frames, past_frames, fps)`` -- le passe part a la memoire.

    Les images decodees sont mises en cache par video : le pool est petit et fixe,
    et le decodage d'une video de 600 s domine le cout par pas (distill.py le
    contournait en pre-calculant z). On ne decode donc chaque video qu'une fois.
    """
    allf = cache.get(vid["name"])
    if allf is None:
        from ..worldmodel.data import load_video_frames
        from .fit_basis import resize_frames
        allf = resize_frames(load_video_frames(vid["path"], num_frames=total_frames),
                             max_side)
        cache[vid["name"]] = allf
    T = allf.shape[0]
    W = min(window, T)
    return allf[-W:], allf[:T - W], total_frames / vid["dur"]


def spliced_past(past: torch.Tensor, rgb, frac: float, nf: int) -> torch.Tensor:
    """Le passe avec ``nf`` images d'un aplat ``rgb`` insere a la fraction ``frac``."""
    _, h, w, _ = past.shape
    pos = min(past.shape[0], max(0, int(round(frac * past.shape[0]))))
    needle = make_needle(nf, h, w, rgb)
    return torch.cat([past[:pos], needle, past[pos:]], 0)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def needle_eval(model, processor, branch, wrapper, vids: list, cache: dict,
                depths: list, site: int, chunk: int, stride: int, nf: int,
                max_side: int, total_frames: int, window: int, prompt: str,
                letter_ids: torch.Tensor, device: str) -> dict:
    """``off``/``on``/``wrong`` sur les videos tenues a l'ecart (cf. ``needle.py``)."""
    branch.eval()

    def pred(frames, fps):
        return int(answer_logits(model, processor, frames, prompt, fps, device,
                                 letter_ids).argmax())

    rows, s = [], 0
    for vid in vids:
        try:
            win, past, fps = load_split(vid, cache, total_frames, window, max_side)
        except Exception:
            continue
        for frac in depths:
            ci = s % len(COLORS)
            wi = (s + len(COLORS) // 2) % len(COLORS)
            past_r = spliced_past(past, COLORS[ci][1], frac, nf)
            past_w = spliced_past(past, COLORS[wi][1], frac, nf)
            with disabled(wrapper):
                off = pred(win, fps)
                st = host_states(model, processor, past_r, fps, device, site)
            state, S, _ = write_grad(branch, st, chunk, stride)
            wrapper.carry, wrapper.token_offset = (state, S), 0
            on = pred(win, fps)
            with disabled(wrapper):
                st = host_states(model, processor, past_w, fps, device, site)
            state, S, _ = write_grad(branch, st, chunk, stride)
            wrapper.carry, wrapper.token_offset = (state, S), 0
            wrong = pred(win, fps)
            rows.append({"gold": ci, "off": off, "on": on, "wrong": wrong})
            s += 1
    branch.train()
    if not rows:
        return {"n": 0}
    acc = lambda k: sum(r[k] == r["gold"] for r in rows) / len(rows)
    return {"n": len(rows), "off": acc("off"), "on": acc("on"), "wrong": acc("wrong")}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True, help="racine : <split> + videos/")
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--dataset", default="lvb", choices=["lvb", "finevideo"],
                    help="lvb : <split> json. finevideo : manifest.jsonl + videos/<id>.mp4")
    ap.add_argument("--dur-min", type=float, default=None, dest="dur_min")
    ap.add_argument("--dur-max", type=float, default=None, dest="dur_max")
    ap.add_argument("--duration-group", type=int, default=600, dest="duration_group")
    ap.add_argument("--basis", default=None, help="base PCA (cache/qwen_l24_pca128.pt)")
    ap.add_argument("--init-ckpt", default=None, dest="init_ckpt",
                    help="repartir d'une branche (defaut : fraiche, up=0)")
    ap.add_argument("--layer", type=int, default=None)
    # --- tache ---
    ap.add_argument("--total-frames", type=int, default=128, dest="total_frames")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--needle-frames", type=int, default=4, dest="needle_frames")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--eval-videos", type=int, default=6, dest="eval_videos",
                    help="videos en tete de tranche tenues a l'ecart (disjointes)")
    ap.add_argument("--max-train-videos", type=int, default=40, dest="max_train_videos")
    ap.add_argument("--eval-depths", default="0.1,0.5,0.9", dest="eval_depths")
    ap.add_argument("--depth-min", type=float, default=0.0, dest="depth_min",
                    help="borne basse de la profondeur d'aiguille tiree a l'entrainement")
    ap.add_argument("--depth-max", type=float, default=1.0, dest="depth_max",
                    help="borne haute (ex. 0.8..1.0 = aiguille proche de la fenetre, cas facile)")
    # --- memoire ---
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--memory-hidden", type=int, default=64, dest="memory_hidden")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=1, dest="write_stride",
                    help="1 : l'aiguille doit etre ecrite")
    ap.add_argument("--alpha-init", type=float, default=1.27e-4, dest="alpha_init")
    ap.add_argument("--muon", default="unit",
                    choices=["rescale", "unit", "clip", "squash", "unit_squash",
                             "softclip"])
    ap.add_argument("--muon-scale", type=float, default=None, dest="muon_scale")
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    # --- optimisation ---
    ap.add_argument("--train-head", default="none", dest="train_head",
                    choices=["none", "full"],
                    help="none : branche seule (defaut). full : degele aussi la tete "
                         "de langage -- l'eval externe devra alors la recharger")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=1e-5, dest="head_lr")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--accum", type=int, default=1,
                    help="echantillons par pas d'optimisation (debruite le batch-1)")
    ap.add_argument("--eval-every", type=int, default=150, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=0, dest="save_every",
                    help="sauver aussi la branche tous les N pas (<out>_at<pas>.pt), "
                         "pour suivre les portes au cours de l'entrainement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="prefixe (.json, .pt)")
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = torch.Generator().manual_seed(a.seed)
    depths = [float(x) for x in a.eval_depths.split(",") if x]

    from transformers import AutoProcessor

    from .check_qwen import load_host

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    # --- videos, split disjoint ------------------------------------------------
    dur_band = ((a.dur_min, a.dur_max) if a.dataset == "finevideo"
                and a.dur_min is not None and a.dur_max is not None else None)
    pool = video_pool(a.data, a.split, a.duration_group,
                      a.eval_videos + a.max_train_videos, a.dataset, dur_band)
    if len(pool) <= a.eval_videos:
        raise SystemExit(f"seulement {len(pool)} videos : pas de quoi separer "
                         f"{a.eval_videos} d'eval et au moins une d'entrainement")
    eval_vids, train_vids = pool[:a.eval_videos], pool[a.eval_videos:]
    print(f"{len(train_vids)} videos d'entrainement, {len(eval_vids)} d'eval "
          f"(disjointes), tranche {a.duration_group} s", flush=True)

    # --- hote + greffe ---------------------------------------------------------
    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer

    pca = (torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
           if a.basis else None)
    mem_kw = memory_kwargs_from_args(a)
    d_model = int(model.config.text_config.hidden_size)
    branch = MemoryBranch(d_model, a.d_mem, chunk_size=a.chunk,
                          write_stride=a.write_stride, down_pca=pca, **mem_kw).to(a.device)
    if a.init_ckpt:
        blob = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        branch.load_state_dict(blob["branch"])
        print(f"branche initialisee depuis {a.init_ckpt}", flush=True)
    branch.train()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False            # la reponse LIT, elle n'ecrit pas
    n_tr = sum(p.numel() for p in branch.trainable_parameters())
    print(f"greffe couche {site} : {n_tr} params entrainables, identite a l'init : "
          f"{branch.is_identity()} muon={mem_kw['orthogonalize_momentum']}", flush=True)

    params = [{"params": list(branch.trainable_parameters()), "lr": a.lr}]
    if a.train_head == "full":
        head = model.get_output_embeddings()
        for p in head.parameters():
            p.requires_grad_(True)
        params.append({"params": list(head.parameters()), "lr": a.head_lr})
        tied = bool(getattr(model.config, "tie_word_embeddings", False))
        print(f"tete de langage DEGELEE (lr {a.head_lr:g})"
              + ("  ATTENTION : embeddings lies -> l'entree bouge aussi" if tied else ""),
              flush=True)
    opt = torch.optim.AdamW(params, weight_decay=0.01)

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    base_lrs = [g["lr"] for g in opt.param_groups]
    prompt = build_prompt()
    letter_ids = torch.tensor(
        [processor.tokenizer.encode("ABCDEFGH"[i], add_special_tokens=False)[0]
         for i in range(len(COLORS))], device=a.device)

    # --- boucle ----------------------------------------------------------------
    frame_cache = {}                          # images decodees, remplies a la volee
    hist, best, t0 = [], -1.0, time.time()
    micro, last_gn = 0, float("nan")
    opt.zero_grad(set_to_none=True)
    for step in range(a.steps):
        for g, bl in zip(opt.param_groups, base_lrs):
            g["lr"] = bl * lr_scale(step)
        vid = train_vids[int(torch.randint(len(train_vids), (1,), generator=rng))]
        ci = int(torch.randint(len(COLORS), (1,), generator=rng))
        frac = a.depth_min + (a.depth_max - a.depth_min) * float(
            torch.rand(1, generator=rng))
        try:
            win, past, fps = load_split(vid, frame_cache, a.total_frames, a.window,
                                        a.max_side)
            past_r = spliced_past(past, COLORS[ci][1], frac, a.needle_frames)
            with disabled(wrapper):
                st = host_states(model, processor, past_r, fps, a.device, site)
            state, S, nw = write_grad(branch, st, a.chunk, a.write_stride)
            wrapper.carry, wrapper.token_offset = (state, S), 0
            cls = answer_logits(model, processor, win, prompt, fps, a.device, letter_ids)
            loss = F.cross_entropy(cls.unsqueeze(0),
                                   torch.tensor([ci], device=a.device))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  pas {step} saute ({vid['name']}): {type(e).__name__}: {e}",
                  flush=True)
            continue
        # Accumulation : on moyenne le gradient sur `accum` echantillons avant de
        # bouger, pour debruiter le batch-1. Chaque echantillon a son propre graphe,
        # libere apres son backward, donc le pic VRAM ne change pas.
        (loss / a.accum).backward()
        micro += 1
        if micro % a.accum == 0:
            last_gn = float(torch.nn.utils.clip_grad_norm_(
                [p for grp in opt.param_groups for p in grp["params"]], a.clip_grad))
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            hit = int(cls.argmax()) == ci
            print(f"  {step:5d}  perte {float(loss):.3f}  {'bon' if hit else '.':>3}  "
                  f"|g| {last_gn:.2f}  ecritures {nw}  "
                  f"{(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)

        if a.save_every and a.out and (step + 1) % a.save_every == 0:
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a)},
                       f"{a.out}_at{step + 1}.pt")

        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ev = needle_eval(model, processor, branch, wrapper, eval_vids,
                             frame_cache, depths, site, a.chunk, a.write_stride,
                             a.needle_frames, a.max_side, a.total_frames, a.window,
                             prompt, letter_ids, a.device)
            ev["step"] = step + 1
            hist.append(ev)
            if ev.get("n"):
                print(f"  == eval {step + 1}: off {100*ev['off']:.0f} on "
                      f"{100*ev['on']:.0f} mauvais {100*ev['wrong']:.0f} "
                      f"(hasard {100/len(COLORS):.0f}, n={ev['n']})", flush=True)
                if ev["on"] > best and a.out:
                    best = ev["on"]
                    save = {"branch": branch.state_dict(), "step": step + 1,
                            "eval": ev, "args": vars(a)}
                    if a.train_head == "full":
                        save["lm_head"] = model.get_output_embeddings().state_dict()
                    torch.save(save, a.out + ".pt")

    if a.out:
        with open(a.out + ".json", "w") as f:
            json.dump({"args": vars(a), "history": hist,
                       "elapsed_s": time.time() - t0}, f, indent=1)
        print(f"\nmeilleur on {100*best:.0f} % -> {a.out}.pt / {a.out}.json")
    print("TRAIN-COLOR-FINIE", flush=True)


if __name__ == "__main__":
    main()
