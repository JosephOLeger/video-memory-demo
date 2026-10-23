"""LongVideoBench dans le REGIME DE LA DISTILLATION (Joseph, 14/10).

La video est echantillonnee en ``--frames`` images uniformes (256), coupee en clips de
``--clip`` (16). Qwen voit le DERNIER clip seul ; la memoire est ecrite avec les clips
d'avant, chacun encode seul (comme ``distill_targets``). Trois scores par question :

    base : Qwen sur le dernier clip, sans memoire
    mem  : + memoire des clips precedents de la MEME video
    mis  : + memoire des clips precedents d'une AUTRE video (controle de contenu)

    PYTHONPATH=src python ~/work/iclr/lvb/eval_lvb_last.py --data ... --ckpt ... --basis ...
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoProcessor

import titans.graft.eval_lvb as E
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.check_qwen import load_host
from titans.graft.fit_basis import resize_frames
from titans.graft.qwen import default_layer, disabled, find_decoder_layers, graft
from titans.graft.train_load import video_z
from titans.worldmodel.data import load_video_frames


@torch.no_grad()
def past_memory(branch, model, processor, frames, fps, site, clip, chunk, stride, device):
    """Ecrit les clips 0..n-2 (chacun encode seul) ; renvoie le carry et le nombre d'ecritures."""
    n = frames.shape[0] // clip
    zs = [video_z(model, processor, frames[c * clip:(c + 1) * clip], fps, site,
                  branch.down, device, stride) for c in range(n - 1)]
    z = torch.cat(zs, 0).float().to(device)[None]
    state, S = branch.init_carry(1)
    state, S = branch.memory.update(z, state, S, min(chunk, z.shape[1]))
    return state, S, int(z.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data = json.load(open(os.path.join(a.data, a.split)))
    if a.limit:
        data = data[:a.limit]
    donors = E.donor_indices(data)
    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model)))
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = blob["args"]
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    stride = int(args.get("write_stride", 8))
    branch = MemoryBranch(int(model.config.text_config.hidden_size), args.get("d_mem", 128),
                          chunk_size=a.chunk, down_pca=pca,
                          **memory_kwargs_from_args(args)).to(a.device)
    branch.load_state_dict(blob["branch"])
    branch.eval()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    print(f"{len(data)} questions ; greffe {a.ckpt} (pas {blob.get('step')}) ; contexte = "
          f"dernier clip de {a.clip} images sur {a.frames}", flush=True)

    def sample(path):
        return resize_frames(load_video_frames(path, num_frames=a.frames), a.max_side)

    rows, t0 = [], time.time()
    for i, item in enumerate(data):
        path = os.path.join(a.data, "videos", item["video_path"])
        try:
            dur = float(item["duration"])
            allf = sample(path)
            fps = allf.shape[0] / dur
            ctx = allf[-a.clip:]
            prompt = E.build_prompt(item)
            n_opt = len(item["candidates"])
            with disabled(wrapper):
                base = E.letter_logits(model, processor, ctx, prompt, n_opt, fps, a.device)
            row = {"id": item["id"], "duration_group": item.get("duration_group"),
                   "duration": dur, "gold": item["correct_choice"], "base": int(base.argmax())}
            other = data[donors[i]]
            for key, frames, f in (("mem", allf, fps),
                                   ("mis", None, None)):
                if key == "mis":
                    frames = sample(os.path.join(a.data, "videos", other["video_path"]))
                    f = frames.shape[0] / float(other["duration"])
                with disabled(wrapper):
                    state, S, nw = past_memory(branch, model, processor, frames, f, site,
                                               a.clip, a.chunk, stride, a.device)
                wrapper.carry, wrapper.token_offset = (state, S), 0
                row[key] = int(E.letter_logits(model, processor, ctx, prompt, n_opt,
                                               fps, a.device).argmax())
                row["writes"] = nw
            rows.append(row)
        except Exception as e:
            print(f"  saut {item['id']}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(data):
            n = max(1, len(rows))
            acc = {k: 100 * sum(r[k] == r["gold"] for r in rows) / n for k in ("base", "mem", "mis")}
            print(f"  [{i + 1}/{len(data)}] base {acc['base']:.1f} | memoire {acc['mem']:.1f} | "
                  f"autre video {acc['mis']:.1f}  ({(time.time() - t0) / (i + 1):.1f} s/question)",
                  flush=True)
    rep_m = E.summarise([{**r} for r in rows])
    rep_x = E.summarise([{**r, "mem": r["mis"]} for r in rows])
    print("\nmemoire de la meme video\n" + E.format_report(rep_m))
    print("\nmemoire d'une autre video\n" + E.format_report(rep_x))
    with open(a.out, "w") as fo:
        json.dump({"args": vars(a), "report_mem": rep_m, "report_mis": rep_x, "rows": rows}, fo,
                  indent=1)
    print("json:", a.out)


if __name__ == "__main__":
    main()
