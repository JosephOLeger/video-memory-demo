"""Charge, surchargee : paires cle->couleur sur un passe ASSEMBLE, + loss dense optionnelle.

Pourquoi un nouveau script
--------------------------
``train_pairs`` fait passer tout le passe dans l'hote a chaque pas (~4 s/pas) et
plafonne a K = 2 en 8000 pas : trop cher pour surcharger la memoire (K jusqu'a 64,
20-30k pas). Or l'hote ecrit dans la memoire sans que la memoire agisse sur lui :
les etats couche-L du passe ne dependent pas de la branche. On les PRE-CALCULE donc
par morceaux et on assemble les passes a la volee :

  * paire   : ``--pair-frames`` images d'un aplat portant un nombre, encodees SEULES
              (une par (cle, couleur, variante)) ;
  * filler  : clips de ``--clip`` images de videos FineVideo, encodes seuls (l'hote
              en flux par petits clips, le regime fenetre glissante + memoire) ;
  * fenetre : les ``--window`` dernieres images d'une video, gardees en pixels pour
              la passe de reponse (la seule qui traverse encore l'hote, avec graphe).

Un passe = ``--fillers`` clips + K paires inserees entre eux, au hasard. Seuls les
tokens VIDEO sont gardes (le gabarit de chat se repeterait a chaque morceau).

Loss
----
``L = CE(lettre) + lambda * L_dense``. ``L_dense`` : a chaque ecriture t, la lecture
``M_t(q*)`` (apres ecriture, requete constante apprise) doit predire le latent suivant
``z_{t+1}`` via une tete lineaire. C'est l'objectif du world model, qui recompense la recence et y fait
collapser ``unit`` ([[collapse-is-horizon-retreat]]). Les deux bras ``--muon unit`` /
``unit_squash`` partagent l'objectif : seule la regle d'ecriture differe.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m titans.graft.train_load \\
        --data data/finevideo --basis cache/qwen_l24_pca128.pt \\
        --bank ~/work/iclr/load/bank.pt --muon unit --dense-lambda 1.0 \\
        --steps 4000 --out ~/work/iclr/load/unit_l1_s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .branch import MemoryBranch, memory_kwargs_from_args
from .needle import COLORS, LETTERS
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_color import answer_logits, video_pool
from .train_pairs import FONT_PATH, KEYS, LIGHT, pairs_prompt

NC = len(COLORS)


# --------------------------------------------------------------------------- #
# banque
# --------------------------------------------------------------------------- #
_FONTS: dict = {}


def pair_image(key: int, ci: int, nf: int, h: int, w: int, variant: int) -> torch.Tensor:
    """Comme ``train_pairs.pair_frames``, avec une gigue de taille / position par variante."""
    from PIL import Image, ImageDraw, ImageFont
    r = random.Random(variant * 100003 + key * 101 + ci)
    size = int(min(h, w) * (0.6 if variant == 0 else r.uniform(0.45, 0.7))
               * min(1.0, 2 / len(str(key))))           # 3 chiffres : meme largeur que 2
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.truetype(FONT_PATH, size)
        except OSError:
            _FONTS[size] = ImageFont.load_default(size=size)
    name, rgb = COLORS[ci]
    img = Image.new("RGB", (w, h), rgb)
    dx = 0 if variant == 0 else r.uniform(-0.15, 0.15) * w
    dy = 0 if variant == 0 else r.uniform(-0.12, 0.12) * h
    fg = (0, 0, 0) if name in LIGHT else (255, 255, 255)
    ImageDraw.Draw(img).text((w / 2 + dx, h / 2 + dy), str(key), fill=fg,
                             font=_FONTS[size], anchor="mm")
    f = torch.from_numpy(np.asarray(img).copy())
    return f.unsqueeze(0).expand(nf, h, w, 3).contiguous()


@torch.no_grad()
def video_z(model, processor, frames, fps, site, down, device, stride) -> torch.Tensor:
    """``down`` des etats couche-``site`` des seuls tokens video, au pas ``stride``."""
    from .fit_basis import video_inputs
    layers = find_decoder_layers(model)
    got = []
    h = layers[site].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    try:
        inp = video_inputs(processor, frames, "", fps)
        inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
        model(**inp, use_cache=False)
    finally:
        h.remove()
    vid_tok = model.config.video_token_id
    mask = inp["input_ids"][0] == vid_tok
    z = down(got[0][0][mask].float())                         # [n_tok, d_mem]
    return z[::stride].half().cpu()


def add_pairs(model, processor, site, down, bank, keys, a, variants):
    """Encode les paires (cle, couleur, variante) manquantes dans ``bank["pairs"]``."""
    h, w = bank["windows"][0]["frames"].shape[1:3]
    fps = bank["windows"][0]["fps"]
    n = 0
    for key in keys:
        for ci in range(NC):
            for v in variants:
                if (key, ci, v) in bank["pairs"]:
                    continue
                fr = pair_image(key, ci, a.pair_frames, h, w, v).float()
                bank["pairs"][(key, ci, v)] = video_z(model, processor, fr, fps, site,
                                                      down, a.device, a.write_stride)
                n += 1
        if key % 20 == 0:
            print(f"  banque : paires cle {key} ({n} nouvelles)", flush=True)


_KEYS_OF: dict = {}


def bank_keys(pairs) -> list:
    """Les cles presentes dans la banque (90 a l'origine ; plus avec ``--n-keys``)."""
    if id(pairs) not in _KEYS_OF:
        _KEYS_OF[id(pairs)] = sorted({k for (k, _, _) in pairs})
    return _KEYS_OF[id(pairs)]


def build_bank(model, processor, site, down, vids, a, *, variants, with_pairs):
    from ..worldmodel.data import load_video_frames
    from .fit_basis import resize_frames
    bank = {"fillers": [], "windows": [], "pairs": {}}
    nfr = a.window + a.clips_per_video * a.clip
    for i, vid in enumerate(vids):
        try:
            allf = resize_frames(load_video_frames(vid["path"], num_frames=nfr), a.max_side)
        except Exception as e:
            print(f"  video sautee {vid['name']}: {e}", flush=True)
            continue
        fps = nfr / vid["dur"]
        bank["windows"].append({"frames": allf[-a.window:].clone(), "fps": fps,
                                "name": vid["name"]})
        for c in range(a.clips_per_video):
            clip = allf[c * a.clip:(c + 1) * a.clip]
            bank["fillers"].append(video_z(model, processor, clip, fps, site, down,
                                           a.device, a.write_stride))
        print(f"  banque : video {i + 1}/{len(vids)}  "
              f"({len(bank['fillers'])} clips, {bank['fillers'][-1].shape[0]} ecritures/clip)",
              flush=True)
    if with_pairs:
        h, w = bank["windows"][0]["frames"].shape[1:3]
        fps = bank["windows"][0]["fps"]
        n = 0
        for key in KEYS:
            for ci in range(NC):
                for v in variants:
                    fr = pair_image(key, ci, a.pair_frames, h, w, v).float()
                    bank["pairs"][(key, ci, v)] = video_z(model, processor, fr, fps, site,
                                                          down, a.device, a.write_stride)
                    n += 1
            print(f"  banque : paires cle {key} ({n})", flush=True)
    return bank


# --------------------------------------------------------------------------- #
# echantillons
# --------------------------------------------------------------------------- #
def assemble(rng, bank, pairs, K, n_fill, variants, gold=None, after=None):
    """Passe ``on`` et ``wrong`` (memes cles et positions), cle interrogee, couleur.

    ``after`` (K = 1, portee) : la paire est placee avec exactement ``after`` clips de
    remplissage APRES elle, donc a ``after * ecritures_par_clip`` ecritures de la fin.
    """
    keys = rng.sample(bank_keys(pairs), K)
    cols = [rng.randrange(NC) for _ in range(K)]
    q = rng.randrange(K)
    if gold is not None:
        cols[q] = gold
    wcols = [rng.randrange(NC) for _ in range(K)]
    wcols[q] = (cols[q] + NC // 2) % NC
    vs = [rng.choice(variants) for _ in range(K)]
    fills = [bank["fillers"][rng.randrange(len(bank["fillers"]))] for _ in range(n_fill)]
    # K paires dans les n_fill + 1 interstices, ordre aleatoire
    gaps = (sorted(rng.randrange(n_fill + 1) for _ in range(K)) if after is None
            else [n_fill - after] * K)

    def seq(cc):
        parts, j = [], 0
        for g in range(n_fill + 1):
            while j < K and gaps[j] == g:
                parts.append(pairs[(keys[j], cc[j], vs[j])])
                j += 1
            if g < n_fill:
                parts.append(fills[g])
        return torch.cat(parts, 0)

    return seq(cols), seq(wcols), keys[q], cols[q]


def write_stream(branch, z, chunk, dense_head=None, delta=1, noise=0.0):
    """Ecrit ``z`` [1, N, d] dans une memoire fraiche, AVEC graphe.

    Avec ``dense_head`` : renvoie aussi la loss dense (prediction de z_{t+delta} depuis la
    lecture apres l'ecriture t) et les portes, sinon ``None``. ``delta`` > 1 : dose
    d'horizon (revue du 12/10), meme loss, cible plus lointaine.
    """
    mem = branch.memory
    k, v, q, al, et, th = mem._project(z)
    state, S = branch.init_carry(z.shape[0])
    # Lecture dense par une requete CONSTANTE apprise (``dense_head.query``), pas par
    # q_proj(z_t) : avec la requete du token courant, la lecture d'une memoire figee
    # est deja une fonction de z_t et predit z_{t+1} sans rien ecrire -- le scout du
    # 12/10 l'a montre (theta -> 0 en 200 pas, memoire eteinte). Ici seul le CONTENU
    # de M_t peut porter l'information : c'est la pression de recence pure.
    qd = (dense_head.query.expand(z.shape[0], z.shape[1], -1)
          if dense_head is not None else None)
    state, S, ys = mem._scan_chunked(k, v, al, et, th, state, S, min(chunk, z.shape[1]),
                                     q=qd,
                                     gamma=mem._gamma(z), x=z)
    ld = None
    if dense_head is not None:
        Y = torch.stack(ys, 1)                                  # [1, N, d]
        if branch.norm_r is not None:
            Y = branch.norm_r(Y)
        if delta > 0:                       # cible future z_{t+delta}
            pred, tgt = dense_head(Y[:, :-delta]), z[:, delta:].detach()
        else:                               # cible passee z_{t-D} (controle "lointain")
            D = -delta
            pred, tgt = dense_head(Y[:, D:]), z[:, :-D].detach()
        if noise > 0:                       # controle de previsibilite (revue du 13/10) :
            # cible bruitee, R^2 atteignable <= 1 / (1 + noise^2)
            tgt = tgt + noise * tgt.std() * torch.randn_like(tgt)
        ld = F.mse_loss(pred, tgt) / tgt.var().clamp_min(1e-6)
    gates = (al.detach(), et.detach(), th.detach())
    return state, S, ld, gates


def pr(w: torch.Tensor) -> float:
    s2 = torch.linalg.svdvals(w.double()) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def take(p, i):
    """Le flux ``i`` d'un carry batche (``[B, ...]`` -> ``[1, ...]``)."""
    return {k: v[i:i + 1] for k, v in p.items()}


def win_prompt_logits(model, processor, win, prompt, letter_ids, device):
    return answer_logits(model, processor, win["frames"], prompt, win["fps"], device,
                         letter_ids)


@torch.no_grad()
def load_eval(model, processor, branch, wrapper, bank, ks, n_per_k, a, letter_ids,
              variants):
    branch.eval()
    rng = random.Random(1234)                     # memes echantillons a chaque eval
    out = {}
    for K in ks:
        hit = {"off": 0, "on": 0, "wrong": 0}
        prs, n = [], 0
        for b0 in range(0, n_per_k, a.batch):
            idx = list(range(b0, min(n_per_k, b0 + a.batch)))
            smp = [assemble(rng, bank, bank["pairs"], K, n_fill_for(K, a), variants,
                            i % NC) for i in idx]
            wins = [bank["windows"][i % len(bank["windows"])] for i in idx]
            for j, (_, _, key, col) in enumerate(smp):
                with disabled(wrapper):
                    hit["off"] += int(win_prompt_logits(model, processor, wins[j],
                                                        pairs_prompt(key), letter_ids,
                                                        a.device).argmax()) == col
            for tag, pos in (("on", 0), ("wrong", 1)):
                z = torch.stack([s_[pos] for s_ in smp]).float().to(a.device)
                state, S, _, _ = write_stream(branch, z, a.chunk)
                if tag == "on" and len(prs) < 8:
                    prs += [pr(state["w1"][j]) for j in range(min(2, len(smp)))]
                for j, (_, _, key, col) in enumerate(smp):
                    wrapper.carry, wrapper.token_offset = (take(state, j), take(S, j)), 0
                    hit[tag] += int(win_prompt_logits(model, processor, wins[j],
                                                      pairs_prompt(key), letter_ids,
                                                      a.device).argmax()) == col
            n += len(smp)
        out[str(K)] = {"n": n, **{c: hit[c] / n for c in hit},
                       "writes": int(z.shape[1]), "pr_w1": sum(prs) / len(prs)}
    branch.train()
    return out


@torch.no_grad()
def reach_eval(model, processor, branch, wrapper, bank, a, letter_ids, variants):
    """Rappel (K = 1) en fonction de la distance : ``--reach-fill`` clips au total, la
    paire suivie de ``after`` clips pour chaque ``after`` de ``--reach-after``."""
    branch.eval()
    rng = random.Random(4321)
    flen = int(bank["fillers"][0].shape[0])
    out = {}
    for after in [int(x) for x in a.reach_after.split(",")]:
        hit, n = 0, 0
        for b0 in range(0, a.reach_n, a.batch):
            idx = list(range(b0, min(a.reach_n, b0 + a.batch)))
            smp = [assemble(rng, bank, bank["pairs"], 1, a.reach_fill, variants, i % NC,
                            after=after) for i in idx]
            z = torch.stack([s_[0] for s_ in smp]).float().to(a.device)
            state, S, _, _ = write_stream(branch, z, a.chunk)
            for j, (_, _, key, col) in enumerate(smp):
                win = bank["windows"][(b0 + j) % len(bank["windows"])]
                wrapper.carry, wrapper.token_offset = (take(state, j), take(S, j)), 0
                hit += int(win_prompt_logits(model, processor, win, pairs_prompt(key),
                                             letter_ids, a.device).argmax()) == col
            n += len(smp)
        out[str(after * flen)] = {"n": n, "on": hit / n, "after_clips": after}
    branch.train()
    return out


def n_fill_for(K, a):
    return a.fillers


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--dur-min", type=float, default=400, dest="dur_min")
    ap.add_argument("--dur-max", type=float, default=800, dest="dur_max")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--bank", required=True, help="cache des morceaux (construit si absent)")
    ap.add_argument("--extend-bank", default=None, dest="extend_bank",
                    help="banque d'origine a etendre a --n-keys cles (si --bank n'existe pas)")
    ap.add_argument("--dense-noise", type=float, default=0.0, dest="dense_noise",
                    help="> 0 : bruit gaussien ajoute a la cible dense (ecart type relatif)")
    ap.add_argument("--max-writes", type=int, default=0, dest="max_writes",
                    help="> 0 : ecritures x flux max par micro-lot (accumulation de gradient)")
    ap.add_argument("--n-keys", type=int, default=90, dest="n_keys")
    ap.add_argument("--build-only", action="store_true", dest="build_only")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--clips-per-video", type=int, default=16, dest="clips_per_video")
    ap.add_argument("--pair-frames", type=int, default=4, dest="pair_frames")
    ap.add_argument("--fillers", type=int, default=8,
                    help="clips de remplissage par passe (les K paires s'intercalent)")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--eval-videos", type=int, default=12, dest="eval_videos")
    ap.add_argument("--max-train-videos", type=int, default=60, dest="max_train_videos")
    ap.add_argument("--train-k", default="1,2,4,8,16,32,64", dest="train_k")
    ap.add_argument("--eval-k", default="1,2,4,8,16,32,64", dest="eval_k")
    ap.add_argument("--eval-n", type=int, default=100, dest="eval_n")
    ap.add_argument("--advance-acc", type=float, default=0.5, dest="advance_acc")
    ap.add_argument("--advance-window", type=int, default=160, dest="advance_window",
                    help="en echantillons")
    ap.add_argument("--min-stage", type=int, default=50, dest="min_stage",
                    help="en pas")
    ap.add_argument("--dense-lambda", type=float, default=0.0, dest="dense_lambda")
    ap.add_argument("--dense-delta", type=int, default=1, dest="dense_delta",
                    help="horizon de la cible dense, en ecritures : predit z_{t+delta} ; negatif = z_{t-D} (rappel)")
    ap.add_argument("--clamp-eta", type=float, default=None, dest="clamp_eta",
                    help="U2 bras C : eta force a cette valeur (ecriture), entrainement compris")
    ap.add_argument("--clamp-alpha", type=float, default=None, dest="clamp_alpha")
    ap.add_argument("--fill-choices", default=None, dest="fill_choices",
                    help="ex. 4,8,16 : nombre de clips de remplissage tire a chaque pas")
    ap.add_argument("--reach-fill", type=int, default=24, dest="reach_fill")
    ap.add_argument("--reach-after", default="0,2,4,8,16,24", dest="reach_after")
    ap.add_argument("--reach-n", type=int, default=0, dest="reach_n",
                    help="> 0 : eval de portee a chaque eval")
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--memory-hidden", type=int, default=64, dest="memory_hidden")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=8, dest="write_stride")
    ap.add_argument("--alpha-init", type=float, default=1.27e-4, dest="alpha_init")
    ap.add_argument("--muon", default="unit")
    ap.add_argument("--muon-scale", type=float, default=None, dest="muon_scale")
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    ap.add_argument("--omega-window", type=int, default=1, dest="omega_window",
                    help="ATLAS Omega : fenetre glissante de c ecritures (1 = Titans)")
    ap.add_argument("--omega-stride", type=int, default=1, dest="omega_stride",
                    help="fenetre Omega : t + les c-1 derniers latents pris 1 sur s")
    ap.add_argument("--feature-degree", type=int, default=1, dest="feature_degree",
                    help="ATLAS : feature map polynomiale des cles (1 = identite)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--batch", type=int, default=8,
                    help="flux ecrits en parallele (meme K) ; 1 pas = 1 maj d'optimiseur")
    ap.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=1000, dest="save_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train_k = sorted(int(x) for x in a.train_k.split(",") if x)
    eval_k = [int(x) for x in a.eval_k.split(",") if x]
    TRAIN_VAR, EVAL_VAR = [0, 1, 2], [3]

    from transformers import AutoProcessor

    from .check_qwen import load_host

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    d_model = int(model.config.text_config.hidden_size)

    if not os.path.exists(a.bank) and a.extend_bank:
        # Charge forte (K > 90) : on reprend clips et fenetres de la banque d'origine et
        # on n'encode que les paires des nouvelles cles 10 .. 10 + n_keys - 1.
        blob = torch.load(a.extend_bank, map_location="cpu", weights_only=False)
        down = MemoryBranch(d_model, a.d_mem, down_pca=pca).down.to(a.device)
        t0 = time.time()
        add_pairs(model, processor, site, down, blob["train"], range(10, 10 + a.n_keys), a,
                  TRAIN_VAR + EVAL_VAR)
        blob["eval"]["pairs"] = blob["train"]["pairs"]
        blob["args_extend"] = vars(a)
        torch.save(blob, a.bank)
        print(f"banque etendue {a.bank} : {len(blob['train']['pairs'])} paires, "
              f"{time.time() - t0:.0f} s", flush=True)
    if not os.path.exists(a.bank):
        pool = video_pool(a.data, "", None, a.eval_videos + a.max_train_videos,
                          "finevideo", (a.dur_min, a.dur_max))
        down = MemoryBranch(d_model, a.d_mem, down_pca=pca).down.to(a.device)
        t0 = time.time()
        tr = build_bank(model, processor, site, down, pool[a.eval_videos:], a,
                        variants=TRAIN_VAR + EVAL_VAR, with_pairs=True)
        ev = build_bank(model, processor, site, down, pool[:a.eval_videos], a,
                        variants=[], with_pairs=False)
        ev["pairs"] = tr["pairs"]
        os.makedirs(os.path.dirname(os.path.abspath(a.bank)), exist_ok=True)
        torch.save({"train": tr, "eval": ev, "args": vars(a)}, a.bank)
        print(f"banque ecrite {a.bank} en {time.time() - t0:.0f} s", flush=True)
    if a.build_only:
        print("BANQUE-FINIE", flush=True)
        return
    blob = torch.load(a.bank, map_location="cpu", weights_only=False)
    bank, ebank = blob["train"], blob["eval"]
    # Longueur commune (les formats d'image varient d'une video a l'autre) : le batch
    # empile des flux de meme K, qui doivent donc avoir exactement la meme longueur.
    from collections import Counter
    flen = Counter(int(f.shape[0]) for f in bank["fillers"]).most_common(1)[0][0]
    plen = min(int(v.shape[0]) for v in bank["pairs"].values())
    for b in (bank, ebank):
        b["pairs"] = {k: v[:plen].to(a.device) for k, v in b["pairs"].items()}
        b["fillers"] = [f[:flen].to(a.device) for f in b["fillers"] if f.shape[0] >= flen]
    print(f"banque : {len(bank['fillers'])} clips train, {len(ebank['fillers'])} eval, "
          f"{len(bank['pairs'])} paires, ~{bank['fillers'][0].shape[0]} ecritures/clip, "
          f"~{next(iter(bank['pairs'].values())).shape[0]} ecritures/paire", flush=True)

    mem_kw = memory_kwargs_from_args(a)
    branch = MemoryBranch(d_model, a.d_mem, chunk_size=a.chunk, write_stride=a.write_stride,
                          down_pca=pca, **mem_kw).to(a.device)
    if a.clamp_eta is not None:
        branch.memory.eta_override = a.clamp_eta
    if a.clamp_alpha is not None:
        branch.memory.alpha_override = a.clamp_alpha
    branch.train()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    dense_head = None
    if a.dense_lambda > 0:
        dense_head = torch.nn.Linear(a.d_mem, a.d_mem).to(a.device)
        dense_head.query = torch.nn.Parameter(torch.randn(a.d_mem, device=a.device)
                                              / a.d_mem ** 0.5)
    params = list(branch.trainable_parameters())
    hparams = list(dense_head.parameters()) if dense_head is not None else []
    opt = torch.optim.AdamW(params + hparams, lr=a.lr, weight_decay=0.01)
    print(f"greffe couche {site} : {sum(p.numel() for p in params)} params, "
          f"muon={mem_kw['orthogonalize_momentum']} lambda={a.dense_lambda} delta={a.dense_delta}", flush=True)

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    letter_ids = torch.tensor(
        [processor.tokenizer.encode(LETTERS[i], add_special_tokens=False)[0]
         for i in range(NC)], device=a.device)

    level, stage_start, recent = 0, 0, []

    def draw_k():
        if level == 0 or rng.random() < 0.5:
            return train_k[level]
        return rng.choice(train_k[:level])

    hist, t0, last_gn = [], time.time(), float("nan")
    run_ce = run_d = 0.0
    gate_log = []
    n_oom = 0
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        K = draw_k()
        nf = (rng.choice([int(x) for x in a.fill_choices.split(",")]) if a.fill_choices
              else n_fill_for(K, a))
        try:
            smp = [assemble(rng, bank, bank["pairs"], K, nf, TRAIN_VAR)
                   for _ in range(a.batch)]
            z_all = torch.stack([s_[0] for s_ in smp]).float()
            # Micro-lots (charge forte) : le graphe de l'ecriture coute ~1 Mo par
            # ecriture et par flux ; au-dela de --max-writes, gradient accumule.
            mb = (max(1, min(a.batch, a.max_writes // z_all.shape[1])) if a.max_writes
                  else a.batch)
            opt.zero_grad(set_to_none=True)
            ce, ld_sum, hits = 0.0, None, []
            for m0 in range(0, a.batch, mb):
                z = z_all[m0:m0 + mb].to(a.device)
                state, S, ld, (al, et, th) = write_stream(branch, z, a.chunk, dense_head,
                                                          a.dense_delta, a.dense_noise)
                ce_m = 0.0
                for j, (_, _, key, col) in enumerate(smp[m0:m0 + mb]):
                    wrapper.carry, wrapper.token_offset = (take(state, j), take(S, j)), 0
                    win = bank["windows"][rng.randrange(len(bank["windows"]))]
                    cls = win_prompt_logits(model, processor, win, pairs_prompt(key),
                                            letter_ids, a.device)
                    ce_m = ce_m + F.cross_entropy(cls.unsqueeze(0),
                                                  torch.tensor([col], device=a.device)) / a.batch
                    hits.append(int(int(cls.argmax()) == col))
                frac = z.shape[0] / a.batch
                loss = ce_m + (a.dense_lambda * ld * frac if ld is not None else 0.0)
                loss.backward()
                ce = ce + float(ce_m)
                if ld is not None:
                    ld_sum = (ld_sum or 0.0) + float(ld) * frac
            ld = ld_sum
            z = z_all
        except torch.OutOfMemoryError:
            n_oom += 1
            print(f"  OOM au pas {step} (K={K}, {z_all.shape[1]} ecritures, micro-lot {mb}) "
                  f"[{n_oom} au total]", flush=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
        last_gn = float(torch.nn.utils.clip_grad_norm_(params + hparams, a.clip_grad))
        opt.step()
        if K == train_k[level]:
            recent = (recent + hits)[-a.advance_window:]
            if (level < len(train_k) - 1 and step - stage_start >= a.min_stage
                    and len(recent) == a.advance_window
                    and sum(recent) / len(recent) >= a.advance_acc):
                level, stage_start, recent = level + 1, step, []
                hist.append({"step": step, "curriculum_K": train_k[level]})
                print(f"  ** palier : K max = {train_k[level]} (pas {step})", flush=True)
        run_ce = 0.95 * run_ce + 0.05 * float(ce) if step else float(ce)
        run_d = (0.95 * run_d + 0.05 * float(ld) if step else float(ld)) if ld is not None else 0.0

        if step % 10 == 0:
            gl = {"step": step, "eta_med": float(et.median()),
                  "eta_frac_off": float((et < 0.5).float().mean()),
                  "alpha_med": float(al.median()), "theta_med": float(th.median()),
                  "pr_w1": pr(state["w1"][0].detach()), "K": K, "writes": int(z.shape[1]),
                  "ce": run_ce, "dense": run_d, "level": train_k[level]}
            gate_log.append(gl)
            print(f"  {step:5d} K {K:2d}/{train_k[level]:2d} ce {run_ce:.3f} dense {run_d:.3f} "
                  f"|g| {last_gn:.2f} eta {gl['eta_med']:.3f} alpha {gl['alpha_med']:.1e} "
                  f"theta {gl['theta_med']:.3f} pr {gl['pr_w1']:.1f} ecr {gl['writes']} "
                  f"{(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)

        last = step + 1 == a.steps
        if a.out and (a.save_every and (step + 1) % a.save_every == 0 or last):
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a),
                        "dense_head": dense_head.state_dict() if dense_head else None},
                       f"{a.out}_at{step + 1}.pt")
        if (step + 1) % a.eval_every == 0 or last:
            ev = load_eval(model, processor, branch, wrapper, ebank, eval_k, a.eval_n, a,
                           letter_ids, EVAL_VAR)
            rch = (reach_eval(model, processor, branch, wrapper, ebank, a, letter_ids,
                              EVAL_VAR) if a.reach_n > 0 else None)
            hist.append({"step": step + 1, "by_K": ev, "reach": rch,
                         "curriculum_K": train_k[level]})
            if rch:
                print("  == portee " + "  ".join(f"{d} ecr: {100 * v['on']:.0f}"
                                                 for d, v in rch.items()), flush=True)
            print(f"  == eval {step + 1}: " + "  ".join(
                f"K={k}: on {100*v['on']:.0f} wr {100*v['wrong']:.0f} off {100*v['off']:.0f}"
                f" pr {v['pr_w1']:.0f}" for k, v in ev.items()), flush=True)
        if a.out and (step % 50 == 0 or last or (step + 1) % a.eval_every == 0):
            with open(a.out + ".json", "w") as f:
                json.dump({"args": vars(a), "history": hist, "gates": gate_log,
                           "elapsed_s": time.time() - t0}, f, indent=1)
    print("TRAIN-LOAD-FINIE", flush=True)


if __name__ == "__main__":
    main()
