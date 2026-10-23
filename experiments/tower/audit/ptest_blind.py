import json, collections
R="/home/josephleger/work/iclr/ptest/"
Q={json.loads(l)["qid"]:json.loads(l) for l in open(R+"questions_all.jsonl")}
T=json.load(open(R+"data/ann/mc_question_train.json"))
key=lambda q,o:(q.strip().lower(),tuple(x.strip().lower() for x in o))
tr=collections.defaultdict(collections.Counter)
for v in T.values():
    for q in v["mc_question"]: tr[key(q["question"],q["options"])][q["answer_id"]]+=1
rows=[json.loads(l) for l in open(R+"res_sub3000.jsonl")]; rows=[r for r in rows if "error" not in r]
print("n scored", len(rows))
S=collections.defaultdict(collections.Counter)
for r in rows:
    q=Q[r["qid"]]; c=tr.get(key(q["question"],q["options"]))
    prior = c.most_common(1)[0][0] if c else None
    for g in ["ALL"]+q["classes"]:
        s=S[g]; s["n"]+=1
        for cond in ("full","blind","window"): s[cond]+=r[cond]["correct"]
        if prior is not None:
            s["cov"]+=1; s["prior_ok"]+=prior==q["answer_id"]; s["blind_eq_prior"]+=r["blind"]["pred"]==prior
            s["blind_cov_ok"]+=r["blind"]["correct"]
    s=S["ALL"]
for g,s in S.items():
    n=s["n"]; cv=s["cov"] or 1
    print(f"{g:7s} n={n} full {100*s['full']/n:.0f} blind {100*s['blind']/n:.0f} window {100*s['window']/n:.0f} | on template-covered ({cv}): train-prior {100*s['prior_ok']/cv:.0f}  blind {100*s['blind_cov_ok']/cv:.0f}  blind==prior {100*s['blind_eq_prior']/cv:.0f}")
print("blind pred letters", collections.Counter("ABC"[r["blind"]["pred"]] for r in rows), "full", collections.Counter("ABC"[r["full"]["pred"]] for r in rows))
