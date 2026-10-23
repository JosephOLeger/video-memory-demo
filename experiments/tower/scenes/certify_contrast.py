"""Certification at GROUP level, raw AND contrastive (logits minus their mean over the 4
variants of the group, then argmax), for ceiling_v3*.py outputs (.json or live .jsonl).
CERTIFIED (contrastive) if ceiling(full) >= 80, floor = max(blind, window, other) ~ 0, margin >= 50.
    python certify_contrast.py results/v3c.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(p):
    if p.endswith(".jsonl"):
        return [json.loads(l) for l in open(p) if l.strip()]
    return json.load(open(p))["rows"]


def scores(sel, c):
    g = defaultdict(list)
    for r in sel:
        g[r["group"]].append(r)
    item, grp, cgrp, citem = [], [], [], []
    for rs in g.values():
        if len(rs) != 4:
            continue
        L = np.array([r[c + "_lg"] for r in rs])
        ok = [int(np.argmax(l)) == r["answer"] for l, r in zip(L, rs)]
        Lc = L - L.mean(0, keepdims=True)
        okc = [int(np.argmax(l)) == r["answer"] for l, r in zip(Lc, rs)]
        item += ok
        citem += okc
        grp.append(all(ok))
        cgrp.append(all(okc))
    return 100 * np.mean(item), 100 * np.mean(grp), 100 * np.mean(citem), 100 * np.mean(cgrp), len(grp)


for p in sys.argv[1:]:
    rows = load(p)
    print(f"\n## {p}")
    print("| family | n grp | cond | raw item | raw grp | contr item | contr grp |")
    print("|---|---|---|---|---|---|---|")
    for fam in dict.fromkeys(r["family"] for r in rows):
        sel = [r for r in rows if r["family"] == fam]
        conds = [k[:-3] for k in sel[0] if k.endswith("_lg")]
        res = {}
        for c in conds:
            res[c] = scores(sel, c)
            i, g, ci, cg, n = res[c]
            print(f"| {fam} | {n} | {c} | {i:.0f} | {g:.0f} | {ci:.0f} | {cg:.0f} |")
        if "full" in res:
            fl = [x for x in ("blind", "window", "other") if x in res]
            for k, name in ((1, "raw"), (3, "contrastive")):
                ceil = res["full"][k]
                floor = max(res[x][k] for x in fl)
                ok = ceil >= 80 and ceil - floor >= 50
                print(f"  {fam} {name}: ceiling {ceil:.0f}, floor {floor:.0f}, margin {ceil - floor:.0f} -> "
                      f"{'CERTIFIED' if ok else 'not certified'}")
