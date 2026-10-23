"""Ceiling of Qwen2.5-VL-3B on the procedural scenes, per question family.

Three conditions per question:
  full  : the whole episode in context
  blind : the text only (no video)
  last  : only the last LAST_S seconds (the still end; start-of-episode evidence gone)

Scoring = argmax of the option-letter logits at the first answer position (as in
titans.graft.eval_lvb.letter_logits). Groups are counterfactual (same text, same
options, each option correct once), so `blind` is exactly chance by construction
and `grp` = fraction of groups with all variants right.

    CUDA_VISIBLE_DEVICES=1 python ceiling.py --groups 25 --out results/v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import torch

import gen_scenes as G


def build_inputs(proc, frames, prompt, fps):
    content = ([{"type": "video"}] if frames is not None else []) + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if frames is None:
        return proc(text=[text], return_tensors="pt")
    # transformers 5.14 ignores fps= and assumes 24 fps (audit #4): pass the real
    # metadata, forbid resampling, and check the M-RoPE time step below.
    inp = proc(text=[text], videos=[frames], return_tensors="pt",
               video_metadata=[{"fps": fps, "total_num_frames": len(frames)}],
               do_sample_frames=False)
    spg = float(inp["second_per_grid_ts"][0])
    assert abs(spg - 2.0 / fps) < 1e-4, f"M-RoPE time step {spg} != {2.0 / fps}"
    return inp


@torch.no_grad()
def letter_logits(model, proc, frames, prompt, n_opt, fps, device, lid):
    inp = build_inputs(proc, frames, prompt, fps)
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    lg = model(**inp, use_cache=False).logits[0, -1].float()
    return lg[lid[:n_opt]].cpu(), int(inp["input_ids"].shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--families", default=",".join(G.FAMILIES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--groups", type=int, default=25, help="groups of 4 per family")
    ap.add_argument("--gstart", type=int, default=0)
    ap.add_argument("--last-s", type=float, default=4.0, dest="last_s")
    ap.add_argument("--conds", default="full,blind,last")
    ap.add_argument("--style", default="{}", help="JSON overrides of gen_scenes.Style")
    ap.add_argument("--intro", default=None, help="override gen_scenes.INTRO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16).to(a.device).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    lid = torch.tensor([proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in G.LET])
    style = G.Style(**json.loads(a.style))
    if a.intro is not None:
        G.INTRO = a.intro
    print("generator md5", __import__("hashlib").md5(open(G.__file__, "rb").read()).hexdigest()[:8],
          flush=True)
    conds = a.conds.split(",")
    nlast = int(round(a.last_s * G.FPS))

    rows, t0 = [], time.time()
    for fam in a.families.split(","):
        for g in range(a.gstart, a.gstart + a.groups):
            for it in G.make_group(fam, g, style, split=a.split):
                frames = G.render(it["episode"])
                n_opt = len(it["options"])
                r = {k: it[k] for k in ("family", "group", "variant", "answer", "options")}
                r["facts"] = it["facts"]
                r["T"] = int(frames.shape[0])
                for c in conds:
                    fr = {"full": frames, "blind": None, "last": frames[-nlast:]}[c]
                    lg, ntok = letter_logits(model, proc, fr, it["prompt"], n_opt, G.FPS,
                                             a.device, lid)
                    r[c] = int(lg.argmax())
                    r[c + "_lg"] = [round(x, 3) for x in lg.tolist()]
                    if c == "full":
                        r["ntok"] = ntok
                rows.append(r)
        sel = [r for r in rows if r["family"] == fam]
        print(f"{fam:12s} n={len(sel)} " + " ".join(
            f"{c} {100 * sum(r[c] == r['answer'] for r in sel) / len(sel):.0f}" for c in conds)
            + f"  ntok~{sel[0].get('ntok')}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("\n" + table(rows, conds))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"))
    print("json:", a.out, "\nCEILING-FINI", flush=True)


def table(rows, conds):
    """Per family: item accuracy, GROUP accuracy (all variants right; the certifying
    number, audit #3) and a debiased item accuracy (letter logits minus their mean
    over the group's variants, which removes any constant letter preference)."""
    fams = list(dict.fromkeys(r["family"] for r in rows))
    hdr = []
    for c in conds:
        hdr += [c, c + " grp"] + ([c + " deb"] if c != "blind" else [])
    lines = ["| family | n | chance item / grp | " + " | ".join(hdr) + " |",
             "|---|---|---|" + "---|" * len(hdr)]
    for f in fams:
        sel = [r for r in rows if r["family"] == f]
        k = len(sel[0]["options"])
        cells = []
        for c in conds:
            groups = defaultdict(list)
            for r in sel:
                groups[r["group"]].append(r)
            acc = 100 * sum(r[c] == r["answer"] for r in sel) / len(sel)
            grp = 100 * sum(all(r[c] == r["answer"] for r in g) for g in groups.values()) / len(groups)
            cells += [f"{acc:.0f}", f"{grp:.0f}"]
            if c != "blind":
                ok = 0
                for g in groups.values():
                    m = torch.tensor([r[c + "_lg"] for r in g]).mean(0)
                    ok += sum(int((torch.tensor(r[c + "_lg"]) - m).argmax()) == r["answer"] for r in g)
                cells.append(f"{100 * ok / len(sel):.0f}")
        lines.append(f"| {f} | {len(sel)} | {100 / k:.0f} / 0 | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
