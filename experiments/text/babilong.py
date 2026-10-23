"""BABILong qa1-qa3 at 4k (100 samples each, data/<task>/4k.json of RMT-team/babilong) on the frozen host: full context vs window only.

Official prompts (DEFAULT_PROMPTS: instruction + examples + post_prompt) and official metric
(compare_answers), from babilong_ext/ (raw files of github.com/booydar/babilong, reviewed: plain data
and string functions). Greedy, 15 new tokens, chat template.
    window: the last W tokens of the context only (the sliding window of the graft, no memory)
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "babilong_ext"))
sys.path.insert(0, HERE)
from prompts import DEFAULT_PROMPTS, get_formatted_input  # noqa: E402
from metrics import compare_answers, TASK_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="qa1,qa2,qa3")
    ap.add_argument("--length", default="4k")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    fo = open(a.out, "a")
    for t in a.tasks.split(","):
        ds = json.load(open(os.path.join(HERE, "babilong_data", f"{t}_{a.length}.json")))
        P = DEFAULT_PROMPTS[t]
        acc = {"full": [], "window": []}
        for i, ex in enumerate(ds):
            if i >= a.n:
                break
            for cond in ("full", "window"):
                ctx = ex["input"]
                if cond == "window":
                    ids = tok.encode(ctx, add_special_tokens=False)[-a.window:]
                    ctx = tok.decode(ids)
                txt = get_formatted_input(ctx, ex["question"], P["examples"], P["instruction"], P["post_prompt"])
                msgs = [{"role": "user", "content": txt}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
                x = ids["input_ids"].cuda()
                with torch.no_grad():
                    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=15,
                                         do_sample=False)
                o = tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)
                ok = bool(compare_answers(ex["target"], o, ex["question"], TASK_LABELS[t]))
                acc[cond].append(ok)
                fo.write(json.dumps({"task": t, "length": a.length, "i": i, "cond": cond, "target": ex["target"],
                                     "output": o, "correct": int(ok), "ntok": int(x.shape[1])}) + "\n")
            fo.flush()
        print(t, a.length, {k: round(100 * sum(v) / len(v), 1) for k, v in acc.items()}, flush=True)
    print("BABILONG-FINI", flush=True)


if __name__ == "__main__":
    main()
