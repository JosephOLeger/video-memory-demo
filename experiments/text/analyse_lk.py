"""Ladder v2 tables with the A/B metric: option log-likelihood in the free-answer format ('lk'):
per-question accuracy and group accuracy by best assignment (permutation of the 4 answers over the 4
variants, chance 1/24); letter-MCQ contrastive kept as secondary. Usage: analyse_lk.py runs2/main.jsonl [...]"""
import itertools
import json
import sys
from collections import defaultdict

import numpy as np

PERMS = list(itertools.permutations(range(4)))
FAMS = ["count_start", "order", "main_object", "place", "presence", "colour", "number", "object_name"]
ORDER = ["oracle", "sel_best", "ep", "full", "fifo", "evfifo", "sel_rand", "key", "mean", "lex", "summ", "other",
         "empty", "blind"]
by = defaultdict(dict)
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if "lk" in r:
            by[(r["cond"], r["family"], r["group"])][r["variant"]] = r


def se(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    return 100 * np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(400)])


acc = defaultdict(lambda: defaultdict(list))
for (c, f, g), vs in by.items():
    if len(vs) < 4:
        continue
    L = np.array([vs[v]["lk"] for v in range(4)])
    ans = [vs[v]["answer"] for v in range(4)]
    best = max(PERMS, key=lambda p: sum(L[v, p[v]] for v in range(4)))
    acc[(c, f)]["perm"].append(int(all(best[v] == ans[v] for v in range(4))))
    acc[(c, f)]["q"].append(np.mean([int(L[v].argmax() == ans[v]) for v in range(4)]))
    G = np.array([vs[v]["lg"] for v in range(4)])
    C = G - G.mean(0, keepdims=True)
    acc[(c, f)]["letter_con"].append(int(all(C[v].argmax() == ans[v] for v in range(4))))
conds = sorted(set(k[0] for k in acc), key=lambda c: (ORDER.index(c.split("@")[0]) if c.split("@")[0] in ORDER else 99, c))
fams = [f for f in FAMS if any(k[1] == f for k in acc)]
print("cell = per-question % / group-by-assignment % ± SE (letter contrastive %) n")
print("cond | " + " | ".join(fams))
for c in conds:
    cells = []
    for f in fams:
        d = acc.get((c, f))
        if not d:
            cells.append("")
            continue
        cells.append(f"{100*np.mean(d['q']):.0f} / {100*np.mean(d['perm']):.0f}±{se(d['perm']):.0f} "
                     f"({100*np.mean(d['letter_con']):.0f}) n{len(d['q'])}")
    print(f"{c} | " + " | ".join(cells))
