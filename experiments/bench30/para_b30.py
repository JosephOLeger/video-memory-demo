"""05/11: paraphrased copies of the bench30 dev sets. Same stories, same answers, only the question is reworded
(Qwen2.5-VL-7B-Instruct, text only, greedy then sampled retries). A rewrite is kept only if it ends with '?', keeps
every capitalised word of the original (names, places) and differs from it; otherwise para_ok = False and the
original question is kept (such records are left out of the analysis).
    python para_b30.py IN.jsonl OUT.jsonl
"""
import json
import re
import sys

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

MID = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTR = ("Rewrite the question below so that it asks for exactly the same information, using clearly different "
         "words and sentence structure. Keep every name, object and place unchanged, and keep the same person "
         "(I, my, me stay as they are). Output only the new question.\n\nQuestion: ")

tok = AutoTokenizer.from_pretrained(MID)
tok.padding_side = "left"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to("cuda").eval()


def caps(q):
    w = re.findall(r"[A-Za-z']+", q)
    return {x for x in w[1:] if x[0].isupper() and x != "I"}


def ok(q, p):
    return p.endswith("?") and caps(q) <= set(re.findall(r"[A-Za-z']+", p)) and \
        p.lower().strip() != q.lower().strip() and len(p) < 4 * len(q) + 40


@torch.no_grad()
def rewrite(qs, sample):
    txt = [tok.apply_chat_template([{"role": "user", "content": INSTR + q}], add_generation_prompt=True,
                                   tokenize=False) for q in qs]
    x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    kw = dict(do_sample=True, temperature=0.9, top_p=0.95) if sample else dict(do_sample=False)
    go = model.generate(**x, max_new_tokens=48, **kw)
    return [tok.decode(g[x["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0] for g in go]


recs = [json.loads(l) for l in open(sys.argv[1])]
qs = [r["question"] for r in recs]
para = [None] * len(qs)
for attempt in range(4):
    todo = [i for i in range(len(qs)) if para[i] is None]
    for b in range(0, len(todo), 32):
        idx = todo[b:b + 32]
        for i, p in zip(idx, rewrite([qs[i] for i in idx], attempt > 0)):
            if ok(qs[i], p):
                para[i] = p
    print(f"attempt {attempt}: {sum(p is not None for p in para)}/{len(qs)} kept", flush=True)
with open(sys.argv[2], "w") as f:
    for r, p in zip(recs, para):
        r = dict(r)
        r["question_orig"], r["para_ok"] = r["question"], p is not None
        r["question"] = p or r["question"]
        f.write(json.dumps(r) + "\n")
