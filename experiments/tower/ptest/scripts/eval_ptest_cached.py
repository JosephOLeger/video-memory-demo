"""Qwen2.5-VL-3B on Perception Test mcQA (valid), 3-way, by option-letter logits.

Conditions per question:
  full   : whole clip (up to the official cut frame), 2 fps, frames resized to fit 640x360
  blind  : the text only
  window : only the last WINDOW_S seconds before the cut, same fps/resolution

Resumable: rows already in --out are skipped.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--questions", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--videos", default=os.path.expanduser("~/work/iclr/ptest/data/videos"))
ap.add_argument("--cut", default=os.path.expanduser("~/work/iclr/ptest/dl/cut_frame_mapping_valid.json"))
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
ap.add_argument("--fps", type=float, default=2.0)
ap.add_argument("--max-w", type=int, default=640)
ap.add_argument("--max-h", type=int, default=360)
ap.add_argument("--window", type=float, default=4.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conds", default="full,blind,window")
ap.add_argument("--check", action="store_true", help="also run uncached and compare")
a = ap.parse_args()

conds = a.conds.split(",")
qs = [json.loads(l) for l in open(a.questions)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["qid"] for l in open(a.out)}
qs = [q for q in qs if q["qid"] not in done]
if a.limit:
    qs = qs[:a.limit]
by_vid = {}
for q in qs:
    by_vid.setdefault(q["video_id"], []).append(q)
cut = json.load(open(a.cut))
print(f"{len(qs)} questions to do on {len(by_vid)} videos ({len(done)} already done)", flush=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

dev = "cuda:0"
proc = AutoProcessor.from_pretrained(a.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
tok = proc.tokenizer
letter_ids = [tok.convert_tokens_to_ids(c) for c in "ABC"]
assert all(isinstance(i, int) and i >= 0 for i in letter_ids), letter_ids
print("letter ids", letter_ids, flush=True)
VEND = tok.convert_tokens_to_ids("<|vision_end|>")


def decode(vid):
    """2-fps frames of the clip up to the cut, resized to fit max_w x max_h."""
    import decord
    path = os.path.join(a.videos, vid + ".mp4")
    vr0 = decord.VideoReader(path, num_threads=2)
    h, w = vr0[0].shape[:2]
    s = min(a.max_w / max(w, h) * (1 if w >= h else 1), 1.0)
    # fit landscape into max_w x max_h, portrait into max_h x max_w
    if w >= h:
        s = min(a.max_w / w, a.max_h / h, 1.0)
    else:
        s = min(a.max_h / w, a.max_w / h, 1.0)
    nw, nh = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    vr = decord.VideoReader(path, width=nw, height=nh, num_threads=4)
    n = len(vr)
    fr = vr.get_avg_fps()
    c = cut.get(vid, -1)
    end = n if c is None or c < 0 else min(int(c), n)
    dur = end / fr
    t = np.arange(0.0, dur, 1.0 / a.fps)
    idx = np.minimum((t * fr).round().astype(int), end - 1)
    frames = vr.get_batch(list(idx)).asnumpy()            # T,H,W,3 uint8
    tw = t >= dur - a.window
    return frames, tw, {"dur": dur, "cut": c, "n_frames_full": len(idx), "hw": [nh, nw]}


def even(frames):
    return frames[len(frames) % 2:]   # temporal patch is 2: drop the earliest if odd


def prompt(q):
    opts = "\n".join(f"{L}. {o}" for L, o in zip("ABC", q["options"]))
    return ("Select the best answer to the following multiple-choice question based on "
            "the video. Respond with only the letter (A, B, or C) of the correct option.\n"
            f"{q['question']}\n{opts}\nThe best answer is:")


@torch.no_grad()
def score(text, frames):
    content = ([{"type": "video"}] if frames is not None else []) + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if frames is not None:
        inp = proc(text=[chat], videos=[frames], return_tensors="pt",
                   video_metadata=[{"fps": a.fps, "total_num_frames": len(frames)}],
                   do_sample_frames=False, do_resize=True,
                   max_pixels=a.max_w * a.max_h * 2, min_pixels=28 * 28 * 4)
    else:
        inp = proc(text=[chat], return_tensors="pt")
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    if frames is not None:
        sp = inp["second_per_grid_ts"]
        assert abs(float(sp[0]) - 2.0 / a.fps) < 1e-4, sp   # mrope time must be real time
    out = model(**inp, logits_to_keep=1)
    lg = out.logits[0, -1].float()
    full_top = int(lg.argmax())
    sel = lg[letter_ids]
    return sel.tolist(), int(sel.argmax()), int(inp["input_ids"].shape[1]), full_top in letter_ids


@torch.no_grad()
def video_prefix(frames):
    """Run [system, user-start, video] once; questions of the same clip reuse its KV cache."""
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "X"}]}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[chat], videos=[frames], return_tensors="pt",
               video_metadata=[{"fps": a.fps, "total_num_frames": len(frames)}],
               do_sample_frames=False, do_resize=True,
               max_pixels=a.max_w * a.max_h * 2, min_pixels=28 * 28 * 4)
    sp = inp["second_per_grid_ts"]
    assert abs(float(sp[0]) - 2.0 / a.fps) < 1e-4, sp
    ids = inp["input_ids"][0]
    k = int((ids == VEND).nonzero()[-1]) + 1          # up to and including <|vision_end|>
    pre = {key: (v.to(dev) if hasattr(v, "to") else v) for key, v in inp.items()}
    pre["input_ids"] = pre["input_ids"][:, :k]
    pre["attention_mask"] = pre["attention_mask"][:, :k]
    out = model(**pre, use_cache=True, logits_to_keep=1)
    return {"cache": out.past_key_values, "k": k, "rope_deltas": model.model.rope_deltas.clone()}


@torch.no_grad()
def score_cached(text, st):
    suf = tok(text + "<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt",
              add_special_tokens=False)["input_ids"].to(dev)
    k = st["k"]
    model.model.rope_deltas = st["rope_deltas"]
    n = suf.shape[1]
    am = torch.ones(1, k + n, dtype=torch.long, device=dev)
    out = model(input_ids=suf, attention_mask=am, past_key_values=st["cache"], use_cache=True,
                cache_position=torch.arange(k, k + n, device=dev), logits_to_keep=1)
    st["cache"].crop(k)
    lg = out.logits[0, -1].float()
    sel = lg[letter_ids]
    return sel.tolist(), int(sel.argmax()), k + n, int(lg.argmax()) in letter_ids


# decode in a background thread pool so the GPU never waits on decord
qd = queue.Queue(maxsize=16)
vids = list(by_vid)


def producer(ids):
    for v in ids:
        try:
            qd.put((v, decode(v), None))
        except Exception as e:  # noqa: BLE001
            qd.put((v, None, repr(e)))


NW = 4
ths = [threading.Thread(target=producer, args=(vids[i::NW],), daemon=True) for i in range(NW)]
for th in ths:
    th.start()

t0 = time.time()
nq = 0
first_dump = True
with open(a.out, "a") as f:
    for _ in range(len(vids)):
        v, dec, err = qd.get()
        if err:
            print("DECODE FAIL", v, err, flush=True)
            for q in by_vid[v]:
                f.write(json.dumps({"qid": q["qid"], "error": err}) + "\n")
            continue
        frames, tw, meta = dec
        fr_full = even(frames)
        fr_win = even(frames[tw])
        if len(fr_win) < 2:
            fr_win = frames[-2:]
        states = {}
        for c in conds:
            if c != "blind":
                states[c] = video_prefix({"full": fr_full, "window": fr_win}[c])
        for q in by_vid[v]:
            row = {"qid": q["qid"], "answer_id": q["answer_id"], **meta,
                   "n_frames_win": len(fr_win)}
            p = prompt(q)
            for c in conds:
                frs = {"full": fr_full, "blind": None, "window": fr_win}[c]
                if c == "blind":
                    lg, pred, ntok, top_is_letter = score(p, None)
                else:
                    lg, pred, ntok, top_is_letter = score_cached(p, states[c])
                    if a.check:
                        lg2, pred2, ntok2, _ = score(p, frs)
                        print(f"CHECK {q['qid']} {c} ntok {ntok}/{ntok2} pred {pred}/{pred2} "
                              f"maxdiff {max(abs(x-y) for x, y in zip(lg, lg2)):.3f}", flush=True)
                row[c] = {"logits": lg, "pred": pred, "correct": int(pred == q["answer_id"]),
                          "ntok": ntok, "top_is_letter": top_is_letter}
            f.write(json.dumps(row) + "\n")
            nq += 1
            if first_dump:
                print("first row", json.dumps(row), flush=True)
                first_dump = False
        del states
        f.flush()
        if nq and (nq % 100 < len(by_vid[v])):
            el = time.time() - t0
            print(f"{nq}/{len(qs)} q  {el/60:.1f} min  {el/nq:.2f} s/q", flush=True)
print(f"DONE {nq} q in {(time.time()-t0)/60:.1f} min", flush=True)
