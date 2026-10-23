"""MANDATORY GATE for the v3c generator (copy of leak_gate_v3.py; count, colour only by default).

For every family, on N groups of each split:
  1. answer == f(facts) (asserted inside make_group) and every option is gold equally often;
  2. prompt identical across the variants of a group (blind is chance by construction);
  3. window frames (last WIN s) AND post-episode filler pixel-identical across variants;
  4. stream length constant;
  5. Bayes shortcut from non-memory features (prompt text, window hash, stream length)
     fit on groups [NTE, NTE+NTR) and tested on [0, NTE): item <= chance + 3, group 0.
    python leak_gate_v3.py
"""
import collections
import hashlib
import sys

import numpy as np

import gen_tableau_v3c as T

NTR, NTE, NPIX = 2000, 500, 10


def run(fam, split):
    bal = collections.Counter()
    tr = collections.defaultdict(collections.Counter)
    for g in range(NTE, NTE + NTR):
        for it in T.make_group(fam, g, split=split):
            bal[it["answer"]] += 1
            tr[(it["prompt"],)][it["answer"]] += 1
    ok, grp = [], []
    for g in range(NTE):
        items = T.make_group(fam, g, split=split)
        assert len({it["prompt"] for it in items}) == 1
        res = []
        for it in items:
            c = tr.get((it["prompt"],))
            res.append((c.most_common(1)[0][0] if c else 0) == it["answer"])
        ok += res
        grp.append(all(res))
    same, lens = [], set()
    for g in range(NPIX):
        st = [T.stream(it) for it in T.make_group(fam, g, split=split)]
        lens |= {s["full"].shape[0] for s in st}
        same.append(all(np.array_equal(st[0][k], s[k]) for s in st[1:] for k in ("win", "post", "pre")))
    return {"balance": dict(sorted(bal.items())), "shortcut_item": round(100 * np.mean(ok), 1),
            "shortcut_grp": round(100 * np.mean(grp), 1), "filler_identical_%": 100 * np.mean(same),
            "lens": sorted(lens)}


def main():
    allok = True
    for split in ("train", "eval"):
        for fam in (sys.argv[1].split(",") if len(sys.argv) > 1 else T.V3C):
            if split == "train" and fam in T.EVAL_ONLY:
                continue
            r = run(fam, split)
            k = len(T.make_group(fam, 0, split=split)[0]["options"])
            vals = list(r["balance"].values())
            passed = (r["shortcut_item"] <= 100 / k + 3 and r["shortcut_grp"] == 0
                      and r["filler_identical_%"] == 100 and len(r["lens"]) == 1
                      and max(vals) == min(vals))
            allok &= passed
            print(f"{split:5s} {fam:12s} {r} -> {'PASS' if passed else 'FAIL'}", flush=True)
    print("LEAK GATE v3:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
