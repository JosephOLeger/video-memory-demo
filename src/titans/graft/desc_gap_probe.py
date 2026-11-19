"""desc_gap_probe.py (17/11): measure the teacher gap for GLOBAL DESCRIPTIONS of the past.
Teacher = Qwen2.5-VL-3B with the whole video up to the window end; same model with the window (last clip) only.
Gap per token w_t = KL(p_full || p_short) on the teacher's own greedy description (soft top-k + rest bucket),
as in distill_targets. Three prompts: free description of the past (P_desc), list of facts (P_facts),
description of the CURRENT scene (P_now, control: the gap should be small).
Second control at the TEACHER level: p_wrong = teacher with ANOTHER video's past (same frame count) + this window.
w_wrong_t = KL(p_full || p_wrong) keeps only what depends on the SPECIFIC past; KL(p_full || p_short) also contains the
effect of simply having many frames in context.
Writes <out>.jsonl (one row per sample) and prints a summary table and annotated examples."""
import argparse, hashlib, json, os, random, time
import torch
from titans.graft.distill_targets import teacher, forced_logits, topk_lp, gather_lp, kl_buckets
from titans.graft.qwen import default_layer, find_decoder_layers

PROMPTS = {
    "desc": "Describe briefly and synthetically what happened earlier in this video, before the current moment. "
            "Focus on the important elements: people, objects, places, actions and the order of events.",
    "facts": "List as short bullet points the important facts about what happened earlier in this video, before the "
             "current moment: who or what appeared, where, and what was done. Only facts you are sure of.",
    "now": "Describe briefly what is shown in the current scene.",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--ends", default="8,12,16")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--max-new", type=int, default=96, dest="max_new")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N manifest rows (pick other videos)")
    ap.add_argument("--prompts", default="desc,facts,now")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import AutoProcessor
    from titans.worldmodel.data import load_video_frames
    from titans.graft.check_qwen import load_host
    from titans.graft.fit_basis import resize_frames
    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    tok = processor.tokenizer
    rows = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))][a.skip:a.skip + a.limit]
    ends = [int(x) for x in a.ends.split(",")]
    fo = open(a.out + ".jsonl", "w"); res = []; t0 = time.time(); n = 0
    loaded = []
    for r in rows:
        vid = r["id"]; path = os.path.join(a.data, "videos", vid + ".mp4"); dur = float(r.get("duration") or 0)
        if not os.path.exists(path) or dur <= 0:
            continue
        try:
            loaded.append((vid, dur, resize_frames(load_video_frames(path, num_frames=a.frames), a.max_side)))
        except Exception as e:
            print(f"  {vid} saute : {e}", flush=True)
    print(f"{len(loaded)} videos chargees", flush=True)

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

    for vi, (vid, dur, allf) in enumerate(loaded):
        wrongf = same_shape(loaded[(vi + 1) % len(loaded)][2], allf) if len(loaded) > 1 else None
        fps = a.frames / dur
        for e in ends:
            lo, hi = (e - 1) * a.clip, e * a.clip
            win = allf[lo:hi]
            for pk in a.prompts.split(","):
                q = PROMPTS[pk]
                try:
                    y, lg_full = teacher(model, processor, allf[:hi], q, fps, a)
                    if y.numel() == 0:
                        continue
                    idx, lp_full = topk_lp(lg_full, a.topk)
                    lg_short = forced_logits(model, processor, win, q, y, fps, a.device)
                    lp_short = gather_lp(lg_short, idx)
                    lp_wrong = None
                    if wrongf is not None:
                        lg_wrong = forced_logits(model, processor, torch.cat([wrongf[:lo], win], 0), q, y, fps, a.device)
                        lp_wrong = gather_lp(lg_wrong, idx)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); print("  OOM", vid, e, pk, flush=True); continue
                w = kl_buckets(lp_full, lp_short).cpu()
                ww = kl_buckets(lp_full, lp_wrong).cpu() if lp_wrong is not None else None
                toks = [tok.decode([int(t)]) for t in y]
                ann = "".join(f"[{t}|{float(x):.1f}/{float(z):.1f}]" if max(float(x), float(z)) > 0.3 else t
                              for t, x, z in zip(toks, w, ww if ww is not None else w))
                row = {"vid": vid, "dur": dur, "end": e, "t_win_s": lo * dur / a.frames, "prompt": pk, "T": int(w.numel()),
                       "sum_w": float(w.sum()), "mean_w": float(w.mean()), "frac_w_gt_0.1": float((w > 0.1).float().mean()),
                       "frac_w_gt_0.5": float((w > 0.5).float().mean()), "max_w": float(w.max()),
                       "sum_ww": float(ww.sum()) if ww is not None else None, "mean_ww": float(ww.mean()) if ww is not None else None,
                       "frac_ww_gt_0.1": float((ww > 0.1).float().mean()) if ww is not None else None,
                       "text": tok.decode(y), "annotated": ann, "toks": toks,
                       "w": [round(float(x), 4) for x in w], "ww": [round(float(z), 4) for z in ww] if ww is not None else None}
                res.append(row); fo.write(json.dumps(row) + "\n"); fo.flush(); n += 1
        print(f"  video {vi + 1}/{len(rows)} {vid} dur {dur:.0f}s  echantillons {n}  ({(time.time() - t0) / max(1, n):.1f} s/echantillon)", flush=True)
    fo.close()
    def q_(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
    print("\n| invite | n | somme w (fenetre seule) med (q10, q90) | w moyen | frac w>0.1 | somme w (mauvais passe) med | ww moyen | frac ww>0.1 | T med |")
    print("|---|---|---|---|---|---|---|---|---|")
    for pk in a.prompts.split(","):
        v = [x for x in res if x["prompt"] == pk]
        if not v: continue
        sw = [x["sum_w"] for x in v]; vw = [x for x in v if x["sum_ww"] is not None]
        print(f"| {pk} | {len(v)} | {q_(sw,.5):.2f} ({q_(sw,.1):.2f}, {q_(sw,.9):.2f}) | {q_([x['mean_w'] for x in v],.5):.3f} | "
              f"{q_([x['frac_w_gt_0.1'] for x in v],.5):.2f} | {q_([x['sum_ww'] for x in vw],.5):.2f} | {q_([x['mean_ww'] for x in vw],.5):.3f} | "
              f"{q_([x['frac_ww_gt_0.1'] for x in vw],.5):.2f} | {q_([x['T'] for x in v],.5):.0f} |")
    for pk in a.prompts.split(","):
        for e in ends:
            v = [x for x in res if x["prompt"] == pk and x["end"] == e]
            if v: print(f"  {pk} end {e}: somme w med {q_([x['sum_w'] for x in v],.5):.2f}, somme ww med {q_([x['sum_ww'] for x in v if x['sum_ww'] is not None],.5):.2f} (n {len(v)})")
    rng = random.Random(0)
    for pk in a.prompts.split(","):
        v = [x for x in res if x["prompt"] == pk]
        for x in rng.sample(v, min(3, len(v))):
            print(f"\n== {pk} {x['vid']} end {x['end']} (fenetre a {x['t_win_s']:.0f}s / {x['dur']:.0f}s) somme w {x['sum_w']:.1f} somme ww {x['sum_ww']}\n  [token|w fenetre/w mauvais passe]  {x['annotated']}")
    print("DESC-GAP-FINIE", flush=True)

if __name__ == "__main__":
    main()
