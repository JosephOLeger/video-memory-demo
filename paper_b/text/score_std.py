"""Official-style metrics for the standard benchmarks run through train_b30.py --jsonl-eval.
Reads runs/<name>_<set>.jsonl (one record per question, keys <cond> = output text), joins the eval set (all_answers,
metric) and prints per type and per condition: RULER = every answer string contained (official), LongBench qa = token
F1 (official normalisation), retrieval / classification = exact normalised match of the gold, containment = any answer.
    python score_std.py EVALSET.jsonl RUN_OUTPUT.jsonl [RUN_OUTPUT2.jsonl ...]"""
import json, re, string, sys, collections

def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())

def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g); n = sum(common.values())
    if n == 0: return 0.0
    pr, rc = n / len(p), n / len(g); return 2 * pr * rc / (pr + rc)

def score(metric, out, answers, all_classes=None):
    o = norm(out)
    if metric == "ruler_all": return float(all(norm(a) in o for a in answers))
    if metric == "qa_f1": return max(f1(out, a) for a in answers)
    if metric == "retrieval": return float(any(norm(a) == o or norm(a) in o for a in answers))
    if metric == "classification":
        hits = [c for c in (all_classes or answers) if norm(c) in o]
        return float(any(norm(a) in o for a in answers)) if hits else 0.0
    return float(any(norm(a) in o for a in answers))


def main():
    ev = {}
    for l in open(sys.argv[1]):
        d = json.loads(l); ev[d["id"]] = d
    for path in sys.argv[2:]:
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for l in open(path):
            r = json.loads(l); d = ev.get(r.get("id")) or ev.get(r.get("sample"))
            if d is None:
                continue
            for k, v in r.items():
                if k in ("win", "full", "mem", "omem", "summ", "bm25", "tlora", "orc", "sup") and isinstance(v, str):
                    acc[d["type"]][k].append(score(d.get("metric", "containment"), v, d["all_answers"], d.get("all_classes")))
        print("==", path)
        conds = sorted({c for t in acc for c in acc[t]})
        print("type".ljust(28), " ".join(c.rjust(6) for c in conds), "   n")
        tot = collections.defaultdict(list)
        for t in sorted(acc):
            print(t.ljust(28), " ".join(f"{100*sum(acc[t][c])/len(acc[t][c]):6.1f}" if acc[t][c] else "     -" for c in conds), f"  {len(acc[t][conds[0]])}")
            for c in conds: tot[c] += acc[t][c]
        print("ALL".ljust(28), " ".join(f"{100*sum(tot[c])/len(tot[c]):6.1f}" if tot[c] else "     -" for c in conds))


if __name__ == "__main__":
    main()
