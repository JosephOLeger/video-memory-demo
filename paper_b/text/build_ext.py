"""bench30 (29/10): held-out external text benchmarks -> jsonl records for train_b30.py --jsonl-eval.

LoCoMo (snap-research/locomo, data/locomo10.json): the whole conversation (sessions in order, with their date line,
'Speaker: text', photo captions in brackets) is the context; categories 1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop (5 adversarial skipped: no gold 'answer' field). n per category, fixed seed, all 10 conversations.
LongMemEval ORACLE (xiaowu0162/longmemeval-cleaned, longmemeval_oracle.json): only the evidence sessions, in order,
'Session date: ...' then 'User: / Assistant:' turns; the question date is appended to the question. n per type.
Official metrics need an LLM judge (LongMemEval) or token F1 (LoCoMo): we store outputs and score F1 / containment
offline (score_ext.py). Neither benchmark is ever used in training.
"""
import argparse
import json
import random
import collections

CAT = {1: "locomo_multihop", 2: "locomo_temporal", 3: "locomo_open", 4: "locomo_single"}


def locomo_text(conv):
    out = []
    k = 1
    while f"session_{k}" in conv:
        out.append(f"\n[{conv.get(f'session_{k}_date_time', '')}]")
        for t in conv[f"session_{k}"]:
            cap = f" [shares a photo: {t['blip_caption']}]" if t.get("blip_caption") else ""
            out.append(f"{t['speaker']}: {t['text']}{cap}")
        k += 1
    return "\n".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="ext/locomo/locomo10.json")
    ap.add_argument("--lme", default="ext/lme/longmemeval_oracle.json")
    ap.add_argument("--n-locomo", type=int, default=50, dest="n_locomo")
    ap.add_argument("--n-lme", type=int, default=30, dest="n_lme")
    a = ap.parse_args()
    r = random.Random(29102026)
    d = json.load(open(a.locomo))
    by = collections.defaultdict(list)
    for s in d:
        txt = locomo_text(s["conversation"])
        for q in s["qa"]:
            if q.get("category") in CAT and "answer" in q:
                by[CAT[q["category"]]].append(dict(type=CAT[q["category"]], sample=s["sample_id"], question=q["question"],
                                                   answer=str(q["answer"]), evidence=q.get("evidence"), input=txt))
    recs = []
    for c, lst in sorted(by.items()):
        r.shuffle(lst)
        recs += lst[:a.n_locomo]
    r.shuffle(recs)
    with open("evalsets/locomo.jsonl", "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    print("locomo", collections.Counter(x["type"] for x in recs), "chars", sum(len(x["input"]) for x in recs) // len(recs))
    l = json.load(open(a.lme))
    by = collections.defaultdict(list)
    for x in l:
        parts = []
        for dt, sess in zip(x["haystack_dates"], x["haystack_sessions"]):
            parts.append(f"\nSession date: {dt}")
            for t in sess:
                parts.append(f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}")
        t_ = "lme_" + x["question_type"].replace("-", "_") + ("_abs" if x["question_id"].endswith("_abs") else "")
        by[t_].append(dict(type=t_, qid=x["question_id"], question=f"{x['question']} (Today is {x['question_date']}.)",
                           answer=str(x["answer"]), input="\n".join(parts).strip()))
    recs = []
    for c, lst in sorted(by.items()):
        r.shuffle(lst)
        recs += lst[:a.n_lme]
    r.shuffle(recs)
    with open("evalsets/lme_oracle.jsonl", "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    print("lme", collections.Counter(x["type"] for x in recs), "chars", sum(len(x["input"]) for x in recs) // len(recs))


if __name__ == "__main__":
    main()
