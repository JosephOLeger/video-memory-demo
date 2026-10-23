"""22/10 : impact of the M-RoPE fps bug on the stream colour-needle targets (tgc).

For K held-out (eval) videos of ~/work/iclr/stream/tgc, recompute with the LEGACY input path
(processor(fps=...) -> second_per_grid_ts 1/12 s) and the FIXED path (fit_basis.video_inputs:
video_metadata + do_sample_frames=False -> 2/fps s):
  * window letters (Qwen on the window clip only),
  * teacher letters + answer states ([evidence clip | window], answer forced),
  * the memory encoding of the evidence clip (layer-24 states, 1 token in 8).
Also checks the legacy recompute reproduces the cached lt_window / lt_teacher.
Writes a FIXED mini target set (all clips re-encoded) for the same videos, plus symlinks to
the old files, so train_stream can evaluate a trained graft on both.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/work/titans-mac/src python fps_impact.py --k 25
"""
import argparse
import glob
import json
import os
import time

import torch

import titans.graft.fit_basis as FB
from titans.graft.check_qwen import load_host
from titans.graft.distill import forced_inputs
from titans.graft.qwen import default_layer, find_decoder_layers
from titans.graft.stream_color_targets import fill
from titans.graft.stream_targets import LET, read_fps

NEW = FB.video_inputs


def LEGACY(processor, frames, prompt, fps):
    """The pre-22/10 path, verbatim: fps= is ignored by transformers 5.14."""
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    arr = frames.round().clamp(0, 255).to(torch.uint8).numpy()
    return processor(text=[text], videos=[arr], fps=fps, return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgc", default=os.path.expanduser("~/work/iclr/stream/tgc"))
    ap.add_argument("--data", default=os.path.expanduser("~/work/titans-mac/data/finevideo"))
    ap.add_argument("--out", default=os.path.expanduser("~/work/iclr/graft_v3/fps_impact"))
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--needle", type=int, default=4)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import AutoProcessor
    os.makedirs(f"{a.out}/tgc_fixed", exist_ok=True)
    os.makedirs(f"{a.out}/tgc_old", exist_ok=True)
    model = load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    layers = find_decoder_layers(model)
    got = []
    layers[default_layer(len(layers))].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    vtok = model.config.video_token_id
    lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]

    def use(path):
        FB.video_inputs = NEW if path == "new" else LEGACY

    @torch.no_grad()
    def encode(fr, fps):
        got.clear()
        inp = FB.video_inputs(proc, fr.float(), "", fps)
        spg = float(inp["second_per_grid_ts"][0])
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        model(**inp, use_cache=False)
        return got[0][0][inp["input_ids"][0] == vtok][::a.stride].half().cpu(), spg

    @torch.no_grad()
    def letters(fr, prompt, fps):
        inp = FB.video_inputs(proc, fr.float(), prompt, fps)
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        return model(**inp, use_cache=False).logits[0, -1, lid].float().cpu()

    @torch.no_grad()
    def forced(fr, prompt, y, fps):
        inp, n0 = forced_inputs(proc, fr.float(), prompt, y, fps, a.device)
        out = model(**inp, use_cache=False, output_hidden_states=True)
        T = y.shape[0]
        return (out.hidden_states[-1][0, n0 - 1:n0 - 1 + T].half().cpu(),
                out.logits[0, n0 - 1, lid].float().cpu())

    files = sorted(glob.glob(f"{a.tgc}/*.pt"))
    rows, nv, t0 = [], 0, time.time()
    clip_rel = []
    for f in files:
        if nv >= a.k:
            break
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not d["eval"]:
            continue
        nv += 1
        vid, fps = d["vid"], d["fps"]
        frames, _ = read_fps(os.path.join(a.data, "videos", vid + ".mp4"), fps, 256)
        clip = lambda c: frames[c * a.clip:(c + 1) * a.clip]          # noqa: E731
        nclip = len(d["clips"])
        # memory encodings, all clips, fixed path; compare with the cached (legacy) ones
        use("new")
        clips_new = []
        for c in range(nclip):
            z, spg = encode(clip(c), fps)
            assert abs(spg - 2.0 / fps) < 1e-4
            old = d["clips"][c].float()
            if old.shape == z.shape:
                clip_rel.append(float((z.float() - old).norm() / old.norm()))
            clips_new.append(z)
        qs_new = []
        for i, q in enumerate(d["qs"]):
            m = q["meta"]
            ev = fill(clip(q["p"]), m["color"], m["f0"], m["f0"] + a.needle)
            win = d["windows"][q["c"]]
            r = {"vid": vid, "i": i, "c": q["c"], "p": q["p"], "answer": q["answer"],
                 "cached_win": int(q["lt_window"].argmax()), "cached_tea": int(q["lt_teacher"].argmax())}
            res = {}
            for path in ("old", "new"):
                use(path)
                lw = letters(win, q["prompt"], fps)
                hT, lT = forced(torch.cat([ev, win]), q["prompt"], q["y"], fps)
                zp, spg = encode(ev, fps)
                res[path] = (lw, hT, lT, zp)
                r[f"win_{path}"] = int(lw.argmax())
                r[f"tea_{path}"] = int(lT.argmax())
                r[f"spg_{path}"] = spg
            (lw0, h0, l0, z0), (lw1, h1, l1, z1) = res["old"], res["new"]
            r["cached_lw_maxdiff"] = float((lw0 - q["lt_window"]).abs().max())
            r["cached_lt_maxdiff"] = float((l0 - q["lt_teacher"]).abs().max())
            r["h_teacher_rel"] = float((h1.float() - h0.float()).norm() / h0.float().norm())
            r["h_teacher_cached_rel"] = float((q["h_teacher"].float() - h0.float()).norm() / h0.float().norm())
            r["patch_rel"] = float((z1.float() - z0.float()).norm() / z0.float().norm()) if z0.shape == z1.shape else None
            r["patch_cos"] = float(torch.nn.functional.cosine_similarity(z1.float(), z0.float(), dim=-1).mean()) if z0.shape == z1.shape else None
            rows.append(r)
            qs_new.append(dict(q, h_teacher=h1, lt_teacher=l1, lt_window=lw1, patch=z1))
        torch.save({**d, "clips": clips_new, "qs": qs_new, "fps_fix": True}, f"{a.out}/tgc_fixed/{vid}.pt")
        dst = f"{a.out}/tgc_old/{vid}.pt"
        if not os.path.exists(dst):
            os.symlink(f, dst)
        print(f"  [{nv}/{a.k}] {vid} {len(d['qs'])} q, {nclip} clips, {(time.time() - t0) / nv:.0f} s/video", flush=True)
    use("new")
    n = len(rows)
    acc = lambda k: 100 * sum(r[k] == r["answer"] for r in rows) / n          # noqa: E731
    agree = lambda x, y: 100 * sum(r[x] == r[y] for r in rows) / n           # noqa: E731
    mean = lambda k: sum(r[k] for r in rows if r[k] is not None) / max(1, sum(r[k] is not None for r in rows))  # noqa: E731
    summ = {"n_q": n, "n_videos": nv,
            "acc_window_old": acc("win_old"), "acc_window_new": acc("win_new"),
            "acc_teacher_old": acc("tea_old"), "acc_teacher_new": acc("tea_new"),
            "acc_window_cached": acc("cached_win"), "acc_teacher_cached": acc("cached_tea"),
            "agree_window_old_new": agree("win_old", "win_new"),
            "agree_teacher_old_new": agree("tea_old", "tea_new"),
            "legacy_reproduces_cache_window_%": agree("win_old", "cached_win"),
            "legacy_reproduces_cache_teacher_%": agree("tea_old", "cached_tea"),
            "cached_lw_maxdiff_mean": mean("cached_lw_maxdiff"),
            "cached_lt_maxdiff_mean": mean("cached_lt_maxdiff"),
            "h_teacher_rel_mean": mean("h_teacher_rel"),
            "h_teacher_legacy_vs_cache_rel_mean": mean("h_teacher_cached_rel"),
            "patch_rel_mean": mean("patch_rel"), "patch_cos_mean": mean("patch_cos"),
            "clip_rel_mean": sum(clip_rel) / max(1, len(clip_rel)), "n_clips": len(clip_rel),
            "spg_old": rows[0]["spg_old"], "spg_new": rows[0]["spg_new"],
            "elapsed_s": time.time() - t0}
    json.dump({"summary": summ, "rows": rows}, open(f"{a.out}/fps_impact.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    print("FPS-IMPACT-FINI", flush=True)


if __name__ == "__main__":
    main()
