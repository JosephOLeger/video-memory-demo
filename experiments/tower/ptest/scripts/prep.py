"""Perception Test mcQA (valid): gist/detail mapping, stratified subsample,
counterfactual-variant groups.  Writes into ~/work/iclr/ptest/."""
import collections
import json
import os
import random

ROOT = os.path.expanduser("~/work/iclr/ptest")
ann = json.load(open(f"{ROOT}/data/ann/mc_question_valid.json"))

# --- 1. official tags -> gist / detail / other --------------------------------
# Tags are the paper's skills (Patraucean et al. 2023, Table/Fig. skills per area).
MAPPING = {
    # gist: what a memory that keeps the scene's outline must retain
    "gist": {
        "change detection": "memory: did something change between start and end",
        "sequencing": "memory: order of actions/events",
        "event recall": "memory: did an event happen",
        "object counting": "abstraction: how many objects",
        "action counting": "abstraction: how many times an action happened",
        "event counting": "abstraction: how many events",
        "object permanence": "physics: cups game, where is the hidden object",
    },
    # detail: fine-grained appearance, what a gist memory is allowed to drop
    "detail": {
        "object attributes": "semantics: colour/shape/material of a given object",
        "feature matching": "abstraction: which objects share a feature",
        "distractor object": "semantics: detecting an unusual object",
        "colour recognition": "semantics: colour of an object",
        "visual discrimination": "memory: was this object seen (fine discrimination)",
        "part recognition": "semantics: parts of objects",
        "material": "physics: material of objects",
    },
}
tag2cls = {t: c for c, d in MAPPING.items() for t in d}

qs = []
for vid, v in ann.items():
    for q in v["mc_question"]:
        cls = sorted({tag2cls.get(t, "other") for t in q["tag"]})
        qs.append({"qid": f"{vid}#{q['id']}", "video_id": vid, "question": q["question"],
                   "options": q["options"], "answer_id": q["answer_id"], "area": q["area"],
                   "reasoning": q["reasoning"], "tags": q["tag"], "classes": cls,
                   "is_cup_game": v["metadata"]["is_cup_game"]})
all_tags = sorted({t for q in qs for t in q["tags"]})
json.dump({"mapping": MAPPING,
           "other": [t for t in all_tags if t not in tag2cls],
           "note": "tags are the official per-question skill labels; a question with "
                   "several tags counts in each tag's row"},
          open(f"{ROOT}/mapping_gist_detail.json", "w"), indent=1, ensure_ascii=False)
with open(f"{ROOT}/questions_all.jsonl", "w") as f:
    for q in qs:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

# --- 2. stratified subsample ~3000, balanced across tags ----------------------
rng = random.Random(0)
by_tag = collections.defaultdict(list)
for i, q in enumerate(qs):
    for t in q["tags"]:
        by_tag[t].append(i)
for t in by_tag:
    rng.shuffle(by_tag[t])
TARGET = 3000
quota = 60
while True:
    chosen = set()
    for t in sorted(by_tag, key=lambda t: len(by_tag[t])):   # rare tags first
        have = sum(1 for i in chosen if t in qs[i]["tags"])
        for i in by_tag[t]:
            if have >= quota:
                break
            if i not in chosen:
                chosen.add(i)
                have += 1
    if len(chosen) >= TARGET:
        break
    quota += 5
sub = sorted(chosen, key=lambda i: (qs[i]["video_id"], i))
with open(f"{ROOT}/questions_sub3000.jsonl", "w") as f:
    for i in sub:
        f.write(json.dumps(qs[i], ensure_ascii=False) + "\n")
cnt = collections.Counter(t for i in sub for t in qs[i]["tags"])
print(f"subsample: {len(sub)} questions, quota/tag={quota}, "
      f"{len({qs[i]['video_id'] for i in sub})} videos")
print(" min tag count", min(cnt.values()), cnt.most_common()[-5:])

# --- 4. counterfactual variants: same question text+options, different answers ---
grp = collections.defaultdict(list)
for q in qs:
    grp[(q["question"].strip().lower(), tuple(o.strip().lower() for o in q["options"]))].append(q)
multi = {k: v for k, v in grp.items() if len(v) > 1}
varying = {k: v for k, v in multi.items() if len({q["answer_id"] for q in v}) > 1}
q_in_varying = sum(len(v) for v in varying.values())
# majority-answer ceiling: a text-only prior that knows the per-group answer histogram
maj = sum(collections.Counter(q["answer_id"] for q in v).most_common(1)[0][1]
          for v in grp.values())
per_tag = collections.defaultdict(lambda: [0, 0, 0])   # n, in varying group, majority-correct
for k, v in grp.items():
    top = collections.Counter(q["answer_id"] for q in v).most_common(1)[0][0]
    var = k in varying
    for q in v:
        for t in q["tags"]:
            per_tag[t][0] += 1
            per_tag[t][1] += var
            per_tag[t][2] += (q["answer_id"] == top)
cf = {"n_questions": len(qs), "n_groups_text_options": len(grp),
      "n_groups_size_gt1": len(multi), "n_groups_answer_varies": len(varying),
      "n_questions_in_varying_groups": q_in_varying,
      "majority_answer_per_group_acc": maj / len(qs),
      "per_tag": {t: {"n": a[0], "frac_in_varying_group": a[1] / a[0],
                      "group_majority_acc": a[2] / a[0]} for t, a in sorted(per_tag.items())},
      "largest_varying_groups": [
          {"question": k[0], "options": list(k[1]), "n": len(v),
           "answers": dict(collections.Counter(q["answer_id"] for q in v))}
          for k, v in sorted(varying.items(), key=lambda kv: -len(kv[1]))[:15]]}
json.dump(cf, open(f"{ROOT}/counterfactual_groups.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in cf.items() if k not in ("per_tag", "largest_varying_groups")})
