"""bench30 (29/10): tables from the eval outputs.
  python score_b30.py runs/ev_b30_kv_b30_dev_4k.jsonl [runs/ev_base_kv_b30_dev_4k.jsonl ...]
For each file: per type, per condition, judged accuracy (the *_ok fields, gen_bench30.judge) and SQuAD-style token F1
(lowercase, punctuation and articles removed; LoCoMo's official metric is this F1, with stemming we do not apply).
Conditions found in several files (win, full) are reported once. Output: markdown on stdout.
"""
import collections
import json
import re
import string
import sys


def ntoks(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def f1(pred, gold):
    p, g = ntoks(pred), ntoks(gold)
    if not p or not g:
        return float(p == g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def load(path):
    rows = [json.loads(x) for x in open(path)]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for c in ("win", "full", "tea", "sup", "orc", "mem", "omem"):
            if c in r:
                acc[r["type"]][c + "_ok"].append(r[c + "_ok"])
                acc[r["type"]][c + "_f1"].append(f1(r[c], r["answer"]))
    return acc


def main():
    files = sys.argv[1:]
    for path in files:
        acc = load(path)
        print(f"\n#### {path}")
        conds = [c for c in ("win", "full", "tea", "sup", "orc", "mem", "omem") if any(c + "_ok" in d for d in acc.values())]
        print("| type | n | " + " | ".join(f"{c} acc / F1" for c in conds) + " |")
        print("|---|---|" + "---|" * len(conds))
        tot = collections.defaultdict(list)
        for t in sorted(acc):
            d = acc[t]
            n = max(len(v) for v in d.values())
            cells = []
            for c in conds:
                if c + "_ok" in d:
                    a = 100 * sum(d[c + "_ok"]) / len(d[c + "_ok"])
                    ff = 100 * sum(d[c + "_f1"]) / len(d[c + "_f1"])
                    cells.append(f"{a:.0f} / {ff:.0f}")
                    tot[c + "_ok"] += d[c + "_ok"]
                    tot[c + "_f1"] += d[c + "_f1"]
                else:
                    cells.append("")
            print(f"| {t} | {n} | " + " | ".join(cells) + " |")
        cells = []
        for c in conds:
            a = 100 * sum(tot[c + "_ok"]) / max(1, len(tot[c + "_ok"]))
            ff = 100 * sum(tot[c + "_f1"]) / max(1, len(tot[c + "_f1"]))
            cells.append(f"{a:.1f} / {ff:.1f}")
        print(f"| **all** | {len(tot[conds[0] + '_ok'])} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
