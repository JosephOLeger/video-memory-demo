"""05/11: evaluation sets for the effect of the graft on the host OUTSIDE its task (train_b30.py --host-eval).

One jsonl per benchmark, small fixed subsets (seed 0), raw outputs scored afterwards by score_host.py.
Record: {id, bench, kind (mcq|num|short|yn|relaxed|nll), prompt, answer, accept?, image?}
    python prep_host_eval.py OUTDIR
"""
import io
import json
import os
import random
import sys

from datasets import load_dataset

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
L = "ABCDEFGHIJ"


def mcq(q, opts, tail="Answer with the letter of the correct option only."):
    return q.strip() + "\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + "\n" + tail


def dump(name, recs):
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(name, len(recs), flush=True)


def save_img(img, name):
    p = os.path.join("img", name + ".jpg")
    img.convert("RGB").save(os.path.join(OUT, p), quality=92)
    return p


def stream_take(ds, n, seed=0, buf=5000):
    return list(ds.shuffle(seed=seed, buffer_size=buf).take(n))


# ---------------------------------------------------------------- text
DONE = {f[:-6] for f in os.listdir(OUT) if f.endswith(".jsonl")}
if "mmlu" not in DONE:
  mm = load_dataset("cais/mmlu", "all", split="test")
by = {}
for i, r in enumerate(mm):
    by.setdefault(r["subject"], []).append(i)
rng = random.Random(0)
recs = []
for s in sorted(by):
    for i in rng.sample(by[s], min(20, len(by[s]))):
        r = mm[i]
        recs.append({"id": f"mmlu-{i}", "bench": "mmlu", "kind": "mcq", "prompt": mcq(r["question"], r["choices"]),
                     "answer": L[r["answer"]], "subject": s})
dump("mmlu", recs)

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=0).select(range(500))
dump("arc", [{"id": f"arc-{r['id']}", "bench": "arc", "kind": "mcq", "prompt": mcq(r["question"], r["choices"]["text"]),
              "answer": L[r["choices"]["label"].index(r["answerKey"])] if r["answerKey"] in r["choices"]["label"]
              else L[int(r["answerKey"]) - 1]} for r in arc])

gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(250))
dump("gsm8k", [{"id": f"gsm-{i}", "bench": "gsm8k", "kind": "num",
                "prompt": r["question"] + "\nThink step by step, then finish with 'The answer is N.'",
                "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(gsm)])

tq = stream_take(load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True), 500)
dump("triviaqa", [{"id": f"tq-{r['question_id']}", "bench": "triviaqa", "kind": "short",
                   "prompt": r["question"] + "\nAnswer with a few words only.", "answer": r["answer"]["value"],
                   "accept": r["answer"]["normalized_aliases"] + [r["answer"]["normalized_value"]]} for r in tq])

# plain continuation of books never used in training (PG19 TEST split; bench30 filler is PG19 validation)
pg = stream_take(load_dataset("emozilla/pg19", split="test", streaming=True), 100, buf=100)
dump("pg19test", [{"id": f"pg-{i}", "bench": "pg19test", "kind": "nll",
                   "input": r["text"][len(r["text"]) // 3: len(r["text"]) // 3 + 40000]} for i, r in enumerate(pg)])

# ---------------------------------------------------------------- vision
mb = load_dataset("lmms-lab/MMBench", "en", split="dev").shuffle(seed=0).select(range(400))
recs = []
for r in mb:
    opts = [r[c] for c in "ABCD" if r.get(c) not in (None, "", "nan")]
    q = (r["hint"] + "\n" if r.get("hint") not in (None, "", "nan") else "") + r["question"]
    recs.append({"id": f"mmb-{r['index']}", "bench": "mmbench", "kind": "mcq", "prompt": mcq(q, opts),
                 "answer": r["answer"], "image": save_img(r["image"], f"mmb_{r['index']}")})
dump("mmbench", recs)

po = load_dataset("lmms-lab/POPE", split="test").shuffle(seed=0).select(range(500))
dump("pope", [{"id": f"pope-{r['question_id']}", "bench": "pope", "kind": "yn",
               "prompt": r["question"] + "\nAnswer yes or no.", "answer": r["answer"].strip().lower(),
               "category": r["category"], "image": save_img(r["image"], f"pope_{r['question_id']}")} for r in po])

cq = load_dataset("HuggingFaceM4/ChartQA", split="test").shuffle(seed=0).select(range(300))
dump("chartqa", [{"id": f"cq-{i}", "bench": "chartqa", "kind": "relaxed",
                  "prompt": r["query"] + "\nAnswer with a single word or number.",
                  "answer": r["label"][0] if isinstance(r["label"], list) else r["label"],
                  "image": save_img(r["image"], f"cq_{i}")} for i, r in enumerate(cq)])

tv = stream_take(load_dataset("lmms-lab/textvqa", split="validation", streaming=True), 300, buf=1000)
dump("textvqa", [{"id": f"tv-{r['question_id']}", "bench": "textvqa", "kind": "short",
                  "prompt": r["question"] + "\nAnswer with a single word or phrase.", "answer": r["answers"][0],
                  "accept": r["answers"], "image": save_img(r["image"], f"tv_{r['question_id']}")} for r in tv])
