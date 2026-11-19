"""ctx_targets.py (17/11, Josep's formulation): the memory stands in for the AMPUTATED context.

Teacher = frozen Qwen2.5-VL with the whole video up to the window end (frames[:hi]); amputated model = the same
Qwen with the window only (last clip, 16 frames); the student will be amputated Qwen + the memory written from the
amputated frames. Targets = the teacher's own greedy answer y to a query, its top-k log-probas, and the same positions
under (a) the window only (lp_short, gap w) and (b) ANOTHER video's past + this window (lp_wrong, gap ww).
Queries are drawn broadly so that no single probe is privileged: anchored scene questions, generic past questions,
a free description or a list of facts about the past, a FineVideo QA, and a question on the CURRENT scene (where the
memory must add nothing). Several amputation lengths per video (window end in clips).

Output: <out>_<shard>.pt, same layout as distill_targets (videos: clips/windows/fps/dur/eval ; samples) so that
train_distill can read it; samples carry the extra field ``ww``.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m titans.graft.ctx_targets --data data/finevideo \\
        --basis cache/qwen_l24_pca128.pt --shard 0 --nshards 4 --out runs/ctx/targets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

import torch

from .branch import MemoryBranch
from .distill_targets import (GENERIC_Q, SCENE_Q, forced_logits, gather_lp, is_eval, kl_buckets,
                              questions, teacher, topk_lp)
from .qwen import default_layer, find_decoder_layers
from .train_load import video_z

DESC_Q = [
    "Describe briefly and synthetically what happened earlier in this video, before the current moment. Focus on "
    "the important elements: people, objects, places, actions and the order of events.",
    "List as short bullet points the important facts about what happened earlier in this video, before the current "
    "moment: who or what appeared, where, and what was done. Only facts you are sure of.",
    "Summarise in three sentences what this video showed before the current moment.",
]
NOW_Q = [
    "Describe briefly what is shown in the current scene.",
    "What is happening right now in the video? Answer in one sentence.",
    "Which people or objects are visible in the current scene? Answer briefly.",
]
ENDS = [3, 4, 5, 6, 8, 10, 12, 14, 16]


def same_shape(x, ref):
    """frames of another video brought to the frame shape of ``ref`` (uint8, channel dim detected)."""
    if tuple(x.shape[1:]) == tuple(ref.shape[1:]):
        return x
    cl = 1 if x.shape[1] == 3 else 3
    xf = x.float() if cl == 1 else x.permute(0, 3, 1, 2).float()
    tgt = ref.shape[2:] if cl == 1 else ref.shape[1:3]
    y = torch.nn.functional.interpolate(xf, size=tuple(tgt), mode="bilinear", align_corners=False)
    y = y if cl == 1 else y.permute(0, 2, 3, 1)
    return y.round().clamp(0, 255).to(x.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--n-ends", type=int, default=4, dest="n_ends", help="window ends drawn per video among ENDS")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--n-scene", type=int, default=2, dest="n_scene")
    ap.add_argument("--n-generic", type=int, default=1, dest="n_generic")
    ap.add_argument("--max-new", type=int, default=48, dest="max_new")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--eval-frac", type=float, default=0.05, dest="eval_frac")
    ap.add_argument("--write-stride", type=int, default=8, dest="write_stride")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, help="prefix ; writes <out>_<shard>.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from ..worldmodel.data import load_video_frames
    from .check_qwen import load_host
    from .fit_basis import resize_frames

    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    down = MemoryBranch(int(model.config.text_config.hidden_size), 128, down_pca=pca).down.to(a.device)

    rows = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))]
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    if a.limit:
        rows = rows[:a.limit]
    out_path = f"{a.out}_{a.shard}.pt"
    done = {"videos": {}, "samples": []}
    if os.path.exists(out_path):                     # resume
        done = torch.load(out_path, weights_only=False)
    t0, n_new, prev = time.time(), 0, None
    kinds = {}
    for vi, r in enumerate(rows):
        vid = r["id"]
        if vid in done["videos"]:
            continue
        path = os.path.join(a.data, "videos", vid + ".mp4")
        dur = float(r.get("duration") or 0)
        if not os.path.exists(path) or dur <= 0:
            continue
        try:
            allf = resize_frames(load_video_frames(path, num_frames=a.frames), a.max_side)
        except Exception as e:
            print(f"  {vid} skipped : {e}", flush=True)
            continue
        fps = a.frames / dur
        nclip = a.frames // a.clip
        clips = [video_z(model, processor, allf[c * a.clip:(c + 1) * a.clip], fps, site, down, a.device, a.write_stride)
                 for c in range(nclip)]
        rng = random.Random(int(hashlib.md5(vid.encode()).hexdigest()[:8], 16))
        wrongf = same_shape(prev, allf) if prev is not None else None
        wins = {}
        for e in sorted(rng.sample(ENDS, min(a.n_ends, len(ENDS)))):
            lo, hi = (e - 1) * a.clip, e * a.clip
            win = allf[lo:hi]
            wins[e] = win.clone()
            t_win = lo * dur / a.frames
            qs = questions(r["meta"], t_win, rng, a.n_scene, a.n_generic)
            qs.append((rng.choice(DESC_Q), "desc", None))
            qs.append((rng.choice(NOW_Q), "now", None))
            for q, kind, dist in qs:
                try:
                    y, lg_full = teacher(model, processor, allf[:hi], q, fps, a)
                    if y.numel() == 0:
                        continue
                    idx, lp_full = topk_lp(lg_full, a.topk)
                    lp_short = gather_lp(forced_logits(model, processor, win, q, y, fps, a.device), idx)
                    lp_wrong = None
                    if wrongf is not None:
                        lp_wrong = gather_lp(forced_logits(model, processor, torch.cat([wrongf[:lo], win], 0), q, y,
                                                           fps, a.device), idx)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM {vid} end {e} {kind}", flush=True)
                    continue
                w = kl_buckets(lp_full, lp_short)
                ww = kl_buckets(lp_full, lp_wrong) if lp_wrong is not None else None
                done["samples"].append({
                    "vid": vid, "end": e, "q": q, "kind": kind, "dist_s": dist,
                    "y": y.to(torch.int32), "idx": idx.cpu().to(torch.int32),
                    "lp_full": lp_full.cpu().half(), "lp_short": lp_short.cpu().half(),
                    "w": w.cpu(), "ww": ww.cpu() if ww is not None else None, "eval": is_eval(vid, a.eval_frac)})
                kinds[kind] = kinds.get(kind, 0) + 1
                n_new += 1
        done["videos"][vid] = {"clips": clips, "windows": wins, "fps": fps, "dur": dur, "eval": is_eval(vid, a.eval_frac)}
        prev = allf
        el = time.time() - t0
        print(f"  [{a.shard}] video {vi + 1}/{len(rows)} {vid}  samples {len(done['samples'])}  "
              f"({el / max(1, n_new):.2f} s/sample)  kinds {kinds}", flush=True)
        if (vi + 1) % 10 == 0:
            torch.save(done, out_path)
    torch.save(done, out_path)
    S = done["samples"]
    for k in sorted(kinds):
        v = [s for s in S if s["kind"] == k]
        sw = sorted(float(s["w"].sum()) for s in v); sww = sorted(float(s["ww"].sum()) for s in v if s["ww"] is not None)
        med = lambda xs: xs[len(xs) // 2] if xs else float("nan")
        print(f"  {k:8s} n {len(v):5d}  sum w med {med(sw):.2f}  sum ww med {med(sww):.2f}", flush=True)
    print("TARGETS-FINIE", flush=True)


if __name__ == "__main__":
    main()
