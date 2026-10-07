"""Evenement bref dans une video longue : memoire greffee vs Qwen par defaut.

La question que pose un relecteur : "pourquoi ne pas donner a Qwen toute la video,
sous-echantillonnee comme il le fait par defaut ?" Ce banc y repond sur le cas ou ce
defaut est structurellement faible : un evenement BREF (un flash de couleur de
``--event-s`` secondes) dans une video LONGUE (tranche 3600 s de LongVideoBench,
jamais vue a l'entrainement).

L'evenement REMPLACE ``event_s`` secondes de video (la chronologie reste reelle).
Conditions, toutes sur le meme echantillon (video, instant, couleur) :

  off        : la fenetre recente seule (les ``window`` dernieres images du flux)
  memory     : la meme fenetre + la memoire, qui a ecrit TOUT le passe en flux dense
               (``--stream-fps``, 2 par defaut = la cadence par defaut de Qwen)
  uniform_n  : la regle par defaut de Qwen (qwen-vl-utils) : ``n`` images en
               ``linspace`` sur toute la video, dans le contexte, sans memoire.
               On balaye ``n`` (le budget de tokens) ; on note si une image tiree
               tombe dans l'evenement (``hit``).

Toutes les images sont a ``max_side`` (256, la resolution d'entrainement de la
branche), donc le cout en tokens est comparable : environ 23 tokens par image.
La fenetre de ``memory`` coute ~``window * 23`` tokens de contexte, ``uniform_n``
en coute ~``n * 23``.

L'instant de l'evenement est fixe par sa DISTANCE a la fenetre (``--distances`` en
s, plus une gigue aleatoire pour ne pas tomber toujours au meme endroit d'un bloc
hote). ``uniform`` ne depend pas de la distance ; ``memory`` si, via le nombre
d'ecritures posterieures (cf. ``reach.py``).

Cout : les etats couche-L de l'hote sur tout le flux sont calcules UNE fois par
video (blocs de ``host_block`` images) et gardes projetes (``down``, d=128) ; pour
chaque echantillon on ne recalcule que le ou les blocs qui contiennent l'evenement.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 python -m titans.graft.brief_event \\
        --data data/longvideobench --ckpt runs/cf_lvb_lr3e3.pt \\
        --basis cache/qwen_l24_pca128.pt --num-videos 24 \\
        --distances 30,120,600,1800 --uniform 64,256,768 \\
        --out runs/brief/cf_lvb_lr3e3
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .fit_basis import resize_frames
from .needle import COLORS, LETTERS, build_prompt
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_color import answer_logits, host_states, video_pool


def decode(vr, idx, max_side, chunk=64):
    """Images ``idx`` de ``vr``, redimensionnees comme a l'entrainement, par paquets."""
    out = []
    for i in range(0, len(idx), chunk):
        out.append(resize_frames(vr.get_batch(idx[i:i + chunk]), max_side))
    return torch.cat(out, 0)


def paint(frames, mask, rgb):
    """Remplace les images ``mask`` par un aplat ``rgb`` (copie)."""
    f = frames.clone()
    for c in range(3):
        f[mask, ..., c] = rgb[c]
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--duration-group", type=int, default=3600, dest="duration_group")
    ap.add_argument("--skip-videos", type=int, default=0, dest="skip_videos")
    ap.add_argument("--num-videos", type=int, default=24, dest="num_videos")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--event-s", type=float, default=2.0, dest="event_s")
    ap.add_argument("--stream-fps", type=float, default=2.0, dest="stream_fps")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--host-block", type=int, default=112, dest="host_block")
    ap.add_argument("--distances", default="30,120,600,1800",
                    help="secondes entre la fin de l'evenement et le debut de la fenetre")
    ap.add_argument("--jitter-s", type=float, default=20.0, dest="jitter_s")
    ap.add_argument("--uniform", default="64,256,768",
                    help="budgets n (images) de la regle par defaut de Qwen")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard-gb", type=float, default=12.0, dest="guard_gb")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import decord
    from transformers import AutoProcessor

    from .check_qwen import load_host

    decord.bridge.set_bridge("torch")
    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    dists = [float(x) for x in a.distances.split(",") if x]
    budgets = [int(x) for x in a.uniform.split(",") if x]
    rng = random.Random(a.seed)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    t = blob["args"]
    chunk, stride = t["chunk"], t["write_stride"]
    pool = video_pool(a.data, a.split, a.duration_group,
                      a.skip_videos + a.num_videos, "lvb")[a.skip_videos:]
    print(f"{len(pool)} videos ; evenement {a.event_s} s ; flux memoire "
          f"{a.stream_fps} img/s ; distances {dists} ; budgets uniformes {budgets}",
          flush=True)

    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = (t["layer"] if t.get("layer") is not None
            else default_layer(len(find_decoder_layers(model))))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    branch = MemoryBranch(int(model.config.text_config.hidden_size), t["d_mem"],
                          chunk_size=chunk, write_stride=stride, down_pca=pca,
                          **memory_kwargs_from_args(t)).to(a.device)
    branch.load_state_dict(blob["branch"])
    branch.eval()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    prompt = build_prompt()
    letter_ids = torch.tensor(
        [processor.tokenizer.encode(LETTERS[i], add_special_tokens=False)[0]
         for i in range(len(COLORS))], device=a.device)

    @torch.no_grad()
    def pred(frames, fps):
        return int(answer_logits(model, processor, frames, prompt, fps, a.device,
                                 letter_ids).argmax())

    @torch.no_grad()
    def block_z(frames):
        """Etats couche-L d'un bloc, projetes par ``down`` (avant le pas d'ecriture)."""
        with disabled(wrapper):
            st = host_states(model, processor, frames, a.stream_fps, a.device, site)
        return branch.down(st)[0]                              # [n_tok, d_mem]

    rows, s, t0 = [], 0, time.time()
    for vi, vid in enumerate(pool):
        try:
            vr = decord.VideoReader(vid["path"])
            nat = float(vr.get_avg_fps())
            dur = len(vr) / nat
            ts = torch.arange(0.0, dur, 1.0 / a.stream_fps)
            idx = (ts * nat).round().long().clamp(max=len(vr) - 1).tolist()
            stream = decode(vr, idx, a.max_side)               # [N, h, w, 3]
        except Exception as e:
            print(f"  saut {vid['name']}: {type(e).__name__}: {e}", flush=True)
            continue
        N, W, B = stream.shape[0], a.window, a.host_block
        win, past = stream[-W:], stream[:-W]
        P = past.shape[0]
        win_start = float(ts[P])
        # etats de base du passe, par blocs, une fois pour toutes
        blocks = [(i, min(i + B, P)) for i in range(0, P, B)]
        zs = [block_z(past[i:j]) for i, j in blocks]
        off = pred(win, a.stream_fps)
        tv = time.time()

        for d in dists:
            end = win_start - d - rng.uniform(0, a.jitter_s)
            start = end - a.event_s
            if start < 0:
                continue
            ci = s % len(COLORS)
            rgb = COLORS[ci][1]
            # --- memoire : flux dense, seuls les blocs touches sont recalcules
            ev = (ts[:P] >= start) & (ts[:P] < end)
            hit_blocks = {k for k, (i, j) in enumerate(blocks) if ev[i:j].any()}
            zs_ev = list(zs)
            for k in hit_blocks:
                i, j = blocks[k]
                zs_ev[k] = block_z(paint(past[i:j], ev[i:j], rgb))
            z = torch.cat(zs_ev, 0)[::stride].unsqueeze(0)
            with torch.no_grad():
                state, S = branch.init_carry(1)
                state, S = branch.memory.update(z, state, S, min(chunk, z.shape[1]))
            wrapper.carry, wrapper.token_offset = (state, S), 0
            mem = pred(win, a.stream_fps)
            # ecritures posterieures a l'evenement : sa fraction de tokens, au prorata
            # dans son bloc (les blocs font tous B images sauf le dernier)
            last = int(ev.nonzero()[-1]) + 1
            k = (last - 1) // B
            n_tok = [zz.shape[0] for zz in zs_ev]
            i, j = blocks[k]
            tok_upto = sum(n_tok[:k]) + n_tok[k] * (last - i) / (j - i)
            n_writes = z.shape[1]
            after = int(round(n_writes * (1 - tok_upto / sum(n_tok))))
            row = {"video": vid["name"], "dur": dur, "distance_s": d,
                   "event_start": start, "gold": ci, "off": off, "memory": mem,
                   "writes": n_writes, "writes_after": after,
                   "event_frames": int(ev.sum())}
            # --- regle par defaut de Qwen : n images uniformes, tout en contexte
            for n in budgets:
                tu = torch.linspace(0.0, ts[-1].item(), n)
                iu = (tu * nat).round().long().clamp(max=len(vr) - 1).tolist()
                fu = decode(vr, iu, a.max_side)
                hu = (tu >= start) & (tu < end)
                if hu.any():
                    fu = paint(fu, hu, rgb)
                with disabled(wrapper):
                    row[f"uniform_{n}"] = pred(fu, n / dur)
                row[f"hit_{n}"] = bool(hu.any())
            rows.append(row)
            s += 1
        del stream, past, zs
        acc = lambda k: sum(r[k] == r["gold"] for r in rows) / max(1, len(rows))
        print(f"  video {vi + 1}/{len(pool)} ({dur / 60:.0f} min, {N} img)  n={len(rows)}"
              f"  off {100 * acc('off'):.0f}  memoire {100 * acc('memory'):.0f}  "
              + "  ".join(f"unif{n} {100 * acc(f'uniform_{n}'):.0f}" for n in budgets)
              + f"  [{time.time() - tv:.0f} s/video, total {time.time() - t0:.0f} s]",
              flush=True)
        with open(a.out + ".json", "w") as f:
            json.dump({"args": vars(a), "train_args": t, "rows": rows}, f, indent=1)

    # --- resume par distance
    summ = {}
    for d in dists:
        r = [x for x in rows if x["distance_s"] == d]
        if not r:
            continue
        e = {"n": len(r), "writes_after": sum(x["writes_after"] for x in r) / len(r)}
        for k in ["off", "memory"] + [f"uniform_{n}" for n in budgets]:
            e[k] = sum(x[k] == x["gold"] for x in r) / len(r)
        for n in budgets:
            e[f"hit_{n}"] = sum(x[f"hit_{n}"] for x in r) / len(r)
        summ[str(int(d))] = e
        print(f"  d={int(d):>5} s  ecritures apres {e['writes_after']:.0f}  "
              f"off {100 * e['off']:.0f}  memoire {100 * e['memory']:.0f}  "
              + "  ".join(f"unif{n} {100 * e[f'uniform_{n}']:.0f} (vu {100 * e[f'hit_{n}']:.0f})"
                          for n in budgets) + f"  (n={e['n']})", flush=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "train_args": t, "summary": summ, "rows": rows},
                  f, indent=1)
    print(f"-> {a.out}.json\nBRIEF-FINIE", flush=True)


if __name__ == "__main__":
    main()
