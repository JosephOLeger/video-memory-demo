"""build_preserve_host.py (17/11): 'silence' pool for --preserve-host. Host-task prompts from TRAINING splits
(TriviaQA rc.nocontext train, MMLU auxiliary_train, ARC-Challenge/Easy train, GSM8K train), formatted exactly like
evalsets/host/*.jsonl, with every eval item excluded (by TriviaQA question_id, ARC id, and normalised question text).
Usage: python build_preserve_host.py RAWDIR OUT.jsonl   (parquet files already downloaded into RAWDIR)"""
import json, os, random, re, sys
import pyarrow.parquet as pq
raw, out = sys.argv[1], sys.argv[2]
EV = os.path.join(os.path.dirname(os.path.abspath(out)), "..", "host")
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
ex_ids, ex_txt = set(), set()
for b in ("triviaqa", "mmlu", "arc", "gsm8k"):
    for l in open(os.path.join(EV, b + ".jsonl")):
        r = json.loads(l); ex_ids.add(r["id"].split("-", 1)[1]); ex_txt.add(norm(r["prompt"].split("\n")[0]))
rng = random.Random(0); pool = []; stats = {}
def add(bench, prompt, key):
    if key in ex_ids or norm(prompt.split("\n")[0]) in ex_txt:
        stats[bench + "_excluded"] = stats.get(bench + "_excluded", 0) + 1; return
    pool.append({"bench": bench, "prompt": prompt})
# TriviaQA
t = pq.read_table(os.path.join(raw, "triviaqa_train.parquet"), columns=["question", "question_id"]).to_pylist()
rng.shuffle(t)
for r in t[:4500]:
    add("triviaqa", f"{r['question'].strip()}\nAnswer with a few words only.", r["question_id"])
# MMLU auxiliary_train
t = pq.read_table(os.path.join(raw, "mmlu_aux_train.parquet"))
t = [r["train"] for r in t.to_pylist()] if "train" in t.column_names else t.to_pylist()   # HF wraps the rows in a struct
rng.shuffle(t)
for r in t[:3500]:
    ch = r["choices"]
    if len(ch) != 4: continue
    opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", ch))
    add("mmlu", f"{r['question'].strip()}\n{opts}\nAnswer with the letter of the correct option only.", "")
# ARC
for f in ("arc_challenge_train.parquet", "arc_easy_train.parquet"):
    t = pq.read_table(os.path.join(raw, f), columns=["id", "question", "choices"]).to_pylist()
    for r in t:
        ch = r["choices"]["text"]
        if len(ch) != 4: continue
        opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", ch))
        add("arc", f"{r['question'].strip()}\n{opts}\nAnswer with the letter of the correct option only.", r["id"])
# GSM8K
t = pq.read_table(os.path.join(raw, "gsm8k_train.parquet"), columns=["question"]).to_pylist()
rng.shuffle(t)
for r in t[:1000]:
    add("gsm8k", f"{r['question'].strip()}\nThink step by step, then finish with 'The answer is' followed by the number.", "")
rng.shuffle(pool)
with open(out, "w") as fo:
    for r in pool: fo.write(json.dumps(r) + "\n")
from collections import Counter
print("pool", len(pool), dict(Counter(r["bench"] for r in pool)), "exclusions", stats)
