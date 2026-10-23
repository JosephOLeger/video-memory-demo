"""Text-only priors on Perception Test valid: answer-id histogram, train-fitted and
leave-one-out template majority, 'cannot judge / I don't know' options, per class/tag."""
import json, collections, re
R = "/home/josephleger/work/iclr/ptest/"
V = [json.loads(l) for l in open(R + "questions_all.jsonl")]
T = json.load(open(R + "data/ann/mc_question_train.json"))
key = lambda q, o: (q.strip().lower(), tuple(x.strip().lower() for x in o))
tr = collections.defaultdict(collections.Counter)
trq = collections.defaultdict(collections.Counter)
for vid, v in T.items():
    for q in v["mc_question"]:
        tr[key(q["question"], q["options"])][q["answer_id"]] += 1
        trq[q["question"].strip().lower()][q["options"][q["answer_id"]].strip().lower()] += 1
va = collections.defaultdict(collections.Counter)
for q in V:
    va[key(q["question"], q["options"])][q["answer_id"]] += 1
print("answer_id histogram (valid):", collections.Counter(q["answer_id"] for q in V))
idk = re.compile(r"(don't know|cannot (judge|tell|be determined)|can't tell|not sure|none of)", re.I)
n_idk = [q for q in V if any(idk.search(o) for o in q["options"])]
print(f"questions with an IDK-type option: {len(n_idk)} ; IDK is the answer: "
      f"{sum(bool(idk.search(q['options'][q['answer_id']])) for q in n_idk)}")
stats = collections.defaultdict(lambda: collections.Counter())
for q in V:
    k = key(q["question"], q["options"])
    c = tr.get(k)
    trm = (c.most_common(1)[0][0] == q["answer_id"]) if c else None
    loo = va[k].copy(); loo[q["answer_id"]] -= 1
    loo = +loo
    loom = (loo.most_common(1)[0][0] == q["answer_id"]) if loo else None
    # answer-text prior on question text only (options shuffled across videos)
    ct = trq.get(q["question"].strip().lower())
    best = None
    if ct:
        sc = [ct.get(o.strip().lower(), 0) for o in q["options"]]
        best = (max(range(3), key=lambda i: sc[i]) == q["answer_id"]) if max(sc) > 0 else None
    noidk = None
    if any(idk.search(o) for o in q["options"]):
        noidk = 1 / sum(not idk.search(o) for o in q["options"]) if not idk.search(q["options"][q["answer_id"]]) else 0
    for g in ["ALL"] + ["cls:" + c for c in q["classes"]] + ["tag:" + t for t in q["tags"]]:
        s = stats[g]; s["n"] += 1
        if trm is not None: s["tr_cov"] += 1; s["tr_ok"] += trm
        if loom is not None: s["loo_cov"] += 1; s["loo_ok"] += loom
        if best is not None: s["txt_cov"] += 1; s["txt_ok"] += best
        if noidk is not None: s["idk_n"] += 1; s["idk_elim"] += noidk
def pr(g):
    s = stats[g]; f = lambda a, b: f"{100*s[a]/s[b]:.0f}% (cov {100*s[b]/s['n']:.0f}%)" if s[b] else "-"
    print(f"{g:32s} n={s['n']:5d} train-template-majority {f('tr_ok','tr_cov'):18s} LOO-valid-majority {f('loo_ok','loo_cov'):18s} train-answer-text {f('txt_ok','txt_cov')}  idk-elim {f('idk_elim','idk_n')}")
for g in sorted(stats, key=lambda g: (not g.startswith(("ALL", "cls")), g)):
    pr(g)
