"""MANDATORY GATE for every generator revision (audit #1).

Bayes-optimal shortcut accuracy from features that need no memory of the episode:
the final state (= what the end hold shows), the episode length T, the option text.
Fit on eval-split groups [NTE, NTE+NTR), test on [0, NTE). Per item and per group.
Also checks, by rendering, that the last LAST frames are pixel-identical across the
variants of a group for every memory family.

PASS = every memory family has shortcut item acc <= chance + 3 pts and group acc 0.
    python leak_gate.py
"""
import collections
import json
import sys

import numpy as np

import gen_scenes as G

NTR, NTE, NPIX, LAST = 3000, 1000, 15, 8


def feats(it):
    f, ep, opts = it["facts"], it["episode"], tuple(it["options"])
    fin = (tuple(f["final"]), f["door1"])
    return {"text": (opts,), "T": (opts, ep.n_frames), "final": (opts, fin),
            "final+T": (opts, fin, ep.n_frames),
            "final_count+door1": (opts, len(f["final"]), f["door1"])}


def run(fam):
    tr = collections.defaultdict(collections.Counter)
    for g in range(NTE, NTE + NTR):
        for it in G.make_group(fam, g):
            for k, v in feats(it).items():
                tr[(k, v)][it["answer"]] += 1
    acc, grp = collections.defaultdict(list), collections.defaultdict(list)
    for g in range(NTE):
        ok = collections.defaultdict(list)
        for it in G.make_group(fam, g):
            for k, v in feats(it).items():
                c = tr.get((k, v))
                ok[k].append((c.most_common(1)[0][0] if c else 0) == it["answer"])
        for k, v in ok.items():
            acc[k] += v
            grp[k].append(all(v))
    res = {k: (round(100 * np.mean(v), 1), round(100 * np.mean(grp[k]), 1)) for k, v in acc.items()}
    # pixel check of the end hold
    same = []
    for g in range(NPIX):
        tails = [G.render(it["episode"])[-LAST:] for it in G.make_group(fam, g)]
        same.append(all(np.array_equal(tails[0], t) for t in tails[1:]))
    res["tail_identical_%"] = round(100 * np.mean(same), 1)
    res["T_values"] = sorted({it["episode"].n_frames for g in range(50) for it in G.make_group(fam, g)})
    return res


def main():
    out, ok = {}, True
    for fam in G.FAMILIES:
        r = out[fam] = run(fam)
        chance = 100 / len(G.make_group(fam, 0)[0]["options"])
        worst = max(r[k][0] for k in ("text", "T", "final", "final+T", "final_count+door1"))
        wgrp = max(r[k][1] for k in ("text", "T", "final", "final+T", "final_count+door1"))
        passed = (worst <= chance + 3 and wgrp == 0 and r["tail_identical_%"] == 100
                  and len(r["T_values"]) == 1)
        if fam in G.MEMORY:
            ok &= passed
        tag = "PASS" if passed else ("control (expected leak)" if fam in G.CONTROL else "FAIL")
        print(f"{fam:12s} chance {chance:.0f}  shortcut worst item {worst:.1f} grp {wgrp:.1f}  "
              f"tail identical {r['tail_identical_%']}%  T={r['T_values']}  -> {tag}", flush=True)
        print("   ", {k: v for k, v in r.items() if k not in ("T_values",)}, flush=True)
    json.dump(out, open("results/leak_gate.json", "w"), indent=1)
    print("LEAK GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
