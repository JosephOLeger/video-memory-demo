"""Portee : jusqu'ou la memoire entrainee rappelle-t-elle la couleur ?

Evaluation seule, sur une branche deja entrainee par ``train_color``. On fait varier
deux axes que l'entrainement n'a vus que dans une bande etroite :

  * la LONGUEUR du passe ``P`` (images ecrites en memoire), au-dela des 112 images
    d'entrainement : la charge de distracteurs grandit ;
  * la PROFONDEUR de l'aiguille dans ce passe : la distance (en ecritures) entre
    l'aiguille et la fenetre.

La mesure retenue est ``on`` vs ``wrong``/``off`` en fonction du nombre d'ECRITURES
posterieures a l'aiguille -- la distance qui compte pour l'oubli.

L'hote traite le passe par BLOCS de ``--host-block`` images (112 = la longueur vue a
l'entrainement, ou un seul forward suffisait). C'est le regime en flux : l'hote ne voit
jamais plus qu'un bloc, la memoire porte le reste. Pour ``P = host_block`` on retombe
exactement sur la condition d'entrainement.

Les videos viennent par defaut de la tranche 3600 s de LongVideoBench, que
l'entrainement (tranche 600 s, ou FineVideo) n'a jamais vue.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python -m titans.graft.reach \\
        --data data/longvideobench --ckpt runs/cf_lvb_lr3e3.pt \\
        --basis cache/qwen_l24_pca128.pt --duration-group 3600 --num-videos 16 \\
        --past-lengths 112,224,448,896 --depths 0,0.25,0.5,0.75,0.95 \\
        --out runs/reach_cf_lvb_lr3e3
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .needle import COLORS, build_prompt, make_needle
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_color import answer_logits, host_states, video_pool


def blocked_states(model, processor, frames, fps, device, site, block):
    """Etats couche-L du passe, l'hote ne voyant qu'un bloc de ``block`` images a la fois."""
    outs = [host_states(model, processor, frames[i:i + block], fps, device, site)
            for i in range(0, frames.shape[0], block)]
    return torch.cat(outs, 1)


@torch.no_grad()
def write(branch, states, chunk, stride):
    z = branch.down(states)[:, ::stride]
    state, S = branch.init_carry(1)
    state, S = branch.memory.update(z, state, S, min(chunk, z.shape[1]))
    return state, S, int(z.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--dataset", default="lvb", choices=["lvb", "finevideo"])
    ap.add_argument("--duration-group", type=int, default=3600, dest="duration_group")
    ap.add_argument("--skip-videos", type=int, default=0, dest="skip_videos",
                    help="sauter les N premieres (ex. les videos d'eval/entrainement)")
    ap.add_argument("--num-videos", type=int, default=16, dest="num_videos")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--past-lengths", default="112,224,448,896", dest="past_lengths")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,0.95")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--host-block", type=int, default=112, dest="host_block")
    ap.add_argument("--needle-frames", type=int, default=None, dest="needle_frames",
                    help="defaut : celui de l'entrainement")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="prefixe .json")
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

    Ps = [int(x) for x in a.past_lengths.split(",") if x]
    depths = [float(x) for x in a.depths.split(",") if x]
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = blob["args"]
    nf = a.needle_frames or targs.get("needle_frames", 4)
    stride, chunk = targs["write_stride"], targs["chunk"]

    pool = video_pool(a.data, a.split, a.duration_group,
                      a.skip_videos + a.num_videos, a.dataset)[a.skip_videos:]
    print(f"{len(pool)} videos, P={Ps}, profondeurs={depths}, stride {stride}, "
          f"ckpt {a.ckpt} (pas {blob.get('step')}, eval {blob.get('eval')})", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = (targs["layer"] if targs.get("layer") is not None
            else default_layer(len(find_decoder_layers(model))))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    branch = MemoryBranch(int(model.config.text_config.hidden_size), targs["d_mem"],
                          chunk_size=chunk, write_stride=stride, down_pca=pca,
                          **memory_kwargs_from_args(targs)).to(a.device)
    branch.load_state_dict(blob["branch"])
    branch.eval()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    if "lm_head" in blob:
        model.get_output_embeddings().load_state_dict(blob["lm_head"])
        print("tete de langage entrainee rechargee", flush=True)

    prompt = build_prompt()
    letter_ids = torch.tensor(
        [processor.tokenizer.encode("ABCDEFGH"[i], add_special_tokens=False)[0]
         for i in range(len(COLORS))], device=a.device)

    @torch.no_grad()
    def pred(frames, fps):
        return int(answer_logits(model, processor, frames, prompt, fps, a.device,
                                 letter_ids).argmax())

    rows, s, t0 = [], 0, time.time()
    for vi, vid in enumerate(pool):
        for P in Ps:
            T = P + a.window
            try:
                allf = resize_frames(load_video_frames(vid["path"], num_frames=T),
                                     a.max_side)
            except Exception as e:
                print(f"  saut {vid['name']} P={P}: {type(e).__name__}: {e}", flush=True)
                continue
            win, past = allf[-a.window:], allf[:-a.window]
            fps = T / vid["dur"]
            _, h, w, _ = past.shape
            with disabled(wrapper):
                off = pred(win, fps)
            for frac in depths:
                ci = s % len(COLORS)
                wi = (s + len(COLORS) // 2) % len(COLORS)
                pos = min(past.shape[0], int(round(frac * past.shape[0])))
                res = {}
                for tag, c in (("on", ci), ("wrong", wi)):
                    pr = torch.cat([past[:pos], make_needle(nf, h, w, COLORS[c][1]),
                                    past[pos:]], 0)
                    with disabled(wrapper):
                        st = blocked_states(model, processor, pr, fps, a.device, site,
                                            a.host_block)
                    state, S, nw = write(branch, st, chunk, stride)
                    wrapper.carry, wrapper.token_offset = (state, S), 0
                    res[tag] = pred(win, fps)
                # ecritures APRES l'aiguille (distance d'oubli), via la proportion
                after = int(round(nw * (past.shape[0] - pos) / (past.shape[0] + nf)))
                rows.append({"video": vid["name"], "P": P, "depth": frac,
                             "writes": nw, "writes_after": after, "gold": ci,
                             "off": off, "on": res["on"], "wrong": res["wrong"]})
                s += 1
        acc = lambda k: sum(r[k] == r["gold"] for r in rows) / max(1, len(rows))
        print(f"  video {vi + 1}/{len(pool)}  n={len(rows)}  on {100*acc('on'):.0f} "
              f"wrong {100*acc('wrong'):.0f} off {100*acc('off'):.0f}  "
              f"{time.time() - t0:.0f} s", flush=True)

    def table(key):
        out = {}
        for r in rows:
            out.setdefault(r[key], []).append(r)
        return {str(k): {"n": len(v),
                         **{c: sum(x[c] == x["gold"] for x in v) / len(v)
                            for c in ("off", "on", "wrong")},
                         "writes": sum(x["writes"] for x in v) / len(v)}
                for k, v in sorted(out.items())}

    summary = {"by_P": table("P"), "by_depth": table("depth")}
    grid = {}
    for r in rows:
        grid.setdefault(f"P{r['P']}_d{r['depth']}", []).append(r)
    summary["grid"] = {k: {"n": len(v), "writes_after": sum(x["writes_after"]
                                                              for x in v) / len(v),
                           **{c: sum(x[c] == x["gold"] for x in v) / len(v)
                              for c in ("off", "on", "wrong")}}
                       for k, v in grid.items()}
    for k, v in summary["by_P"].items():
        print(f"  P={k:>4}  ecritures {v['writes']:.0f}  on {100*v['on']:.0f}  "
              f"wrong {100*v['wrong']:.0f}  off {100*v['off']:.0f}  (n={v['n']})")
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "train_args": targs, "summary": summary,
                   "rows": rows, "elapsed_s": time.time() - t0}, f, indent=1)
    print(f"-> {a.out}.json\nREACH-FINIE", flush=True)


if __name__ == "__main__":
    main()
