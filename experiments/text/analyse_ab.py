"""Aggregate A/B eval files: contrastive group accuracy (own memory, other variant's memory, window), donor,
free answer, per family x D, with bootstrap SE over groups. Usage: analyse_ab.py runs_ab/*_eval1000.jsonl"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

FAMS = ["count_start", "order", "main_object", "place", "presence", "colour", "number", "object_name"]
KIND = {"count_start": "gist", "order": "gist", "main_object": "gist", "place": "held-out", "presence": "uncert.",
        "colour": "detail", "number": "detail", "object_name": "detail"}


def load(path):
    by = defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        by[(r["D"], r["family"], r["group"])][r["variant"]] = r
    out = defaultdict(lambda: defaultdict(list))
    for (D, f, g), vs in by.items():
        if len(vs) < 4:
            continue
        for key in ("lg_mem", "lg_other", "lg_win"):
            L = np.array([vs[v][key] for v in range(4)])
            C = L - L.mean(0, keepdims=True)
            out[(D, f)][key].append(int(all(C[v].argmax() == vs[v]["answer"] for v in range(4))))
        out[(D, f)]["free"].append(np.mean([vs[v]["free_ok"] for v in range(4)]))
        out[(D, f)]["donor"].append(np.mean([int(np.argmax(vs[v]["lg_other"]) == vs[v]["donor_answer"]) for v in range(4)]))
    return out


def se(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    return 100 * np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(400)])


for p in sys.argv[1:]:
    d = load(p)
    print(f"== {os.path.basename(p)}   (contrastive group %: mem±SE / other / window | donor | free)")
    for D in sorted(set(k[0] for k in d)):
        for f in FAMS:
            if (D, f) not in d:
                continue
            x = d[(D, f)]
            print(f"  D{D:5d} {f:12s} {KIND[f]:9s} {100*np.mean(x['lg_mem']):5.1f}±{se(x['lg_mem']):3.1f} / "
                  f"{100*np.mean(x['lg_other']):5.1f} / {100*np.mean(x['lg_win']):5.1f} | {100*np.mean(x['donor']):5.1f}"
                  f" | {100*np.mean(x['free']):5.1f}  n{len(x['lg_mem'])}")
