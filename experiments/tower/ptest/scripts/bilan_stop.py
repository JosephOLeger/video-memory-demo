import collections, json
R = "/home/josephleger/work/iclr/ptest/"
Q = [json.loads(l) for l in open(R + "questions_all.jsonl")]
M = json.load(open(R + "mapping_gist_detail.json"))["mapping"]
t2c = {t: c for c, d in M.items() for t in d}
key = lambda q: (q["question"].strip().lower(), tuple(o.strip().lower() for o in q["options"]))
G = collections.defaultdict(collections.Counter)
for q in Q: G[key(q)][q["answer_id"]] += 1
def prior(q):  # leave-one-out majority answer of the same template (question+options)
    c = G[key(q)].copy(); c[q["answer_id"]] -= 1
    tot = sum(c.values())
    if tot == 0: return 1/3
    m = max(c.values()); top = [a for a, v in c.items() if v == m]
    return (q["answer_id"] in top) / len(top)
qd = {q["qid"]: q for q in Q}
rows = [json.loads(l) for l in open(R + "res_sub3000.jsonl")]
rows = [r for r in rows if "error" not in r]
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for r in rows:
    q = qd[r["qid"]]; p = prior(q)
    for t in q["tags"] + ["__ALL__"]:
        a = agg[t]; a["n"] += 1; a["prior"] += p
        for c in ("full", "blind", "window"): a[c] += r[c]["correct"]
out = {}
for t, a in agg.items():
    n = a["n"]; d = {k: a[k] / n for k in ("full", "blind", "window", "prior")}
    d["n"] = int(n); d["class"] = t2c.get(t, "other")
    d["margin"] = d["full"] - max(d["blind"], d["prior"])
    out[t] = d
json.dump({"n_scored": len(rows), "prior": "leave-one-out majority answer of same question+options template over all 19140 valid questions",
           "by_tag": out}, open(R + "bilan.json", "w"), indent=1)
L = ["# Perception Test (valid) x Qwen2.5-VL-3B : bilan (arrêté)", "",
     f"Évaluation arrêtée sur décision du lead : {len(rows)} questions notées sur 3018 (échantillon stratifié par tag). "
     "Choix 3 voies par logits des lettres. full = clip jusqu'au cut officiel, 2 fps, 640x360 max. "
     "window = 4 dernières secondes. blind = texte seul. prior = réponse majoritaire (leave-one-out) des autres questions de même gabarit (texte+options) dans valid.", "",
     "Marge = full − max(blind, prior). Hasard 33.", "",
     "| tag | classe | n | full | blind | window | prior | marge |", "|---|---|---|---|---|---|---|---|"]
for t, d in sorted(out.items(), key=lambda kv: -kv[1]["margin"]):
    nm = "**tout**" if t == "__ALL__" else t; cl = "" if t == "__ALL__" else d["class"]
    L.append(f"| {nm} | {cl} | {d['n']} | {100*d['full']:.0f} | {100*d['blind']:.0f} | {100*d['window']:.0f} | {100*d['prior']:.0f} | {100*d['margin']:+.0f} |")
ok = [t for t, d in out.items() if t != "__ALL__" and d["margin"] >= 0.30]
L += ["", f"Tags avec marge ≥ 30 points : {', '.join(ok) if ok else 'AUCUN'}.",
      "", "Variantes contrefactuelles : 5248 gabarits (texte+options), 1242 répétés, 948 dont la réponse varie selon la vidéo (14380 questions sur 19140). "
      "La réponse majoritaire par gabarit donne 70 % sur valid (in-sample) : les priors textuels sont forts."]
open(R + "bilan.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
