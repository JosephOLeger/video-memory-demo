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
    return proc(text=[text], videos=[frames], fps=fps, return_tensors="pt")


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
    conds = a.conds.split(",")
    nlast = int(round(a.last_s * G.FPS))

    rows, t0 = [], time.time()
    for fam in a.families.split(","):
        for g in range(a.gstart, a.gstart + a.groups):
            for it in G.make_group(fam, g, style):
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
    fams = list(dict.fromkeys(r["family"] for r in rows))
    lines = ["| family | n | chance | " + " | ".join(conds) + " | grp(full) |",
             "|---|---|---|" + "---|" * len(conds) + "---|"]
    for f in fams:
        sel = [r for r in rows if r["family"] == f]
        ch = 100 / len(sel[0]["options"])
        accs = [100 * sum(r[c] == r["answer"] for r in sel) / len(sel) for c in conds]
        groups = defaultdict(list)
        for r in sel:
            groups[r["group"]].append(r["full"] == r["answer"] if "full" in r else False)
        grp = 100 * sum(all(v) for v in groups.values()) / len(groups)
        lines.append(f"| {f} | {len(sel)} | {ch:.0f} | " + " | ".join(f"{x:.0f}" for x in accs)
                     + f" | {grp:.0f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
