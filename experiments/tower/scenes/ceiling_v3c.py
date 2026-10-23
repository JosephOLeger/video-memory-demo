"""Ceiling of Qwen2.5-VL on the v3c tableau benchmark (copy of ceiling_v3.py; generator gen_tableau_v3c).

Conditions (per item):
  full   : ONE video = [pre filler 4 s][tableau 6 s][post filler 8 s][window filler 8 s]
           (the episode in its natural position, never in the last 8 s)
  oracle : ladder format: "Memory of earlier parts..." video = tableau ; "The most recent
           seconds..." video = window filler   (core.py Host.build layout)
  other  : ladder format, memory = tableau of ANOTHER variant of the same group
  window : ladder format, window filler only   (identical across a group -> chance)
  blind  : text only                           (identical across a group -> chance)
  solo   : the tableau alone                   (generator ceiling)

M-RoPE time: real metadata, no resampling, assert second_per_grid_ts == 1.0 s (audit #4).
Certification metric = GROUP accuracy (all variants right), plus debiased item acc.

    CUDA_VISIBLE_DEVICES=1 python ceiling_v3.py --groups 25 --out results/v3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

import gen_tableau_v3c as T
from ceiling import table

MEM_TXT = "Memory of earlier parts of the video stream:"
WIN_TXT = "The most recent seconds of the video stream:"


class Scorer:
    def __init__(self, model_id, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.lid = torch.tensor([self.proc.tokenizer.encode(c, add_special_tokens=False)[0]
                                 for c in T.LET])
        self.device = device

    @torch.no_grad()
    def score(self, pieces, prompt, n_opt):
        """pieces: list of (label_text or None, frames)."""
        content, vids = [], []
        for lab, fr in pieces:
            if lab:
                content.append({"type": "text", "text": lab})
            content.append({"type": "video"})
            vids.append(fr)
        content.append({"type": "text", "text": prompt})
        text = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        if vids:
            inp = self.proc(text=[text], videos=vids, return_tensors="pt",
                            video_metadata=[{"fps": T.FPS, "total_num_frames": len(v)} for v in vids],
                            do_sample_frames=False)
            spg = inp["second_per_grid_ts"]
            assert torch.allclose(torch.as_tensor(spg, dtype=torch.float32),
                                  torch.full((len(vids),), 2.0 / T.FPS)), spg
        else:
            inp = self.proc(text=[text], return_tensors="pt")
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        lg = self.model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        return lg[self.lid[:n_opt].to(lg.device)].cpu(), int(inp["input_ids"].shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--families", default=",".join(T.FAMILIES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--groups", type=int, default=25)
    ap.add_argument("--gstart", type=int, default=0)
    ap.add_argument("--conds", default="full,oracle,other,window,blind,solo")
    ap.add_argument("--style", default="{}")
    ap.add_argument("--check-at", type=int, default=15, dest="check_at")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md5 = hashlib.md5(open(T.__file__, "rb").read() + open(T.T.__file__, "rb").read() + open(T.G.__file__, "rb").read()).hexdigest()[:8]
    print("generator md5", md5, "model", a.model, flush=True)
    sc = Scorer(a.model)
    style = T.TStyle(**json.loads(a.style))
    conds = a.conds.split(",")
    rows, t0 = [], time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    live = open(a.out.replace(".json", ".jsonl"), "a")        # partial results, row by row
    for fam in a.families.split(","):
        for gi, g in enumerate(range(a.gstart, a.gstart + a.groups)):
            if gi == a.check_at and "full" in conds:          # EARLY STOP (lead rule 22/10)
                sel = [r for r in rows if r["family"] == fam]
                gc = lambda c: 100 * sum(all(r[c] == r["answer"] for r in sel if r["group"] == h)   # noqa: E731
                                         for h in {r["group"] for r in sel}) / max(1, len({r["group"] for r in sel}))
                ceil = gc("full")
                fl = max(gc(c) for c in ("blind", "window", "other") if c in conds)
                if ceil < 40 or fl >= ceil - 10:
                    print(f"EARLY-STOP {fam} after {gi} groups: ceiling grp {ceil:.0f}, floor grp {fl:.0f}",
                          flush=True)
                    break
            items = T.make_group(fam, g, style, split=a.split)
            streams = [T.stream(it) for it in items]
            for v, it in enumerate(items):
                s = streams[v]
                # other-variant memory: same group (decor, filler, options), the next
                # variant's tableau, whose answer is a DIFFERENT option by construction
                o = streams[(v + 1) % len(items)]
                r = {k: it[k] for k in ("family", "group", "variant", "answer", "options")}
                r["facts"] = it["facts"]
                for c in conds:
                    pieces = {"full": [(None, s["full"])],
                              "oracle": [(MEM_TXT, s["tab"]), (WIN_TXT, s["win"])],
                              "other": [(MEM_TXT, o["tab"]), (WIN_TXT, s["win"])],
                              "window": [(WIN_TXT, s["win"])],
                              "blind": [],
                              "solo": [(None, s["tab"])]}[c]
                    lg, ntok = sc.score(pieces, it["prompt"], len(it["options"]))
                    r[c] = int(lg.argmax())
                    r[c + "_lg"] = [round(x, 3) for x in lg.tolist()]
                    r[c + "_ntok"] = ntok
                rows.append(r)
                live.write(json.dumps(r) + "\n")
                live.flush()
        sel = [r for r in rows if r["family"] == fam]
        print(f"{fam:12s} n={len(sel)} " + " ".join(
            f"{c} {100 * sum(r[c] == r['answer'] for r in sel) / len(sel):.0f}" for c in conds)
            + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("\n" + table(rows, conds))
    json.dump({"args": vars(a), "md5": md5, "rows": rows}, open(a.out, "w"))
    print("json:", a.out, "\nCEILING-FINI", flush=True)


if __name__ == "__main__":
    main()
