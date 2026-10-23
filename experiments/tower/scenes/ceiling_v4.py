"""Ceiling of Qwen2.5-VL on the v4 composited benchmark (+ load L=2, L=5).

Conditions per item (M-RoPE real time, asserted in ceiling_v3.Scorer):
  full          ONE video: [pre 4 s][episode 6 s][post 8 s][window 8 s], all the same real clip
  oracle        ladder format: memory = episode ; window = last 8 s of the clip
  other         same, memory = episode of ANOTHER variant of the same group
  window        window only ; blind  text only
  oracle_all|L  load: memory = target episode WITH the turquoise border (ladder cue) followed by
                L distractor episodes = other variants of the same group (cycling, no border,
                as ladder3.py), prompt names the turquoise clip. L in {2, 5}.
Early stop (lead rule): after --check-at groups, stop a family if its full group ceiling < 40
or the floor >= ceiling - 10.

    CUDA_VISIBLE_DEVICES=0 python ceiling_v4.py --groups 25 --out results/v4.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

import gen_composite_v4 as V
from ceiling import table
from ceiling_v3 import MEM_TXT, WIN_TXT, Scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--families", default=",".join(V.FAMILIES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--groups", type=int, default=25)
    ap.add_argument("--gstart", type=int, default=0)
    ap.add_argument("--conds", default="full,oracle,other,window,blind,oracle_all|L2,oracle_all|L5")
    ap.add_argument("--check-at", type=int, default=15, dest="check_at")
    ap.add_argument("--load-order", default="first", choices=["first", "last"], dest="load_order",
                    help="target episode first (ladder) or last (just before the window)")
    ap.add_argument("--load-src", default="variants", choices=["variants", "groups"], dest="load_src",
                    help="distractors = other variants of the group (ladder) or episodes of other groups")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cue", default="border", choices=["border", "banner"])
    ap.add_argument("--v4c", action="store_true", help="dark halo, bigger object, count {2,4,6,8}")
    ap.add_argument("--count-even", action="store_true", dest="count_even",
                    help="v4b: count options always {2,4,6,8}")
    a = ap.parse_args()
    if a.cue == "banner":
        V.CUE_STYLE = "banner"
        V.CUE_INTRO = V.CUE_INTRO.replace("has a turquoise border. The question is about the clip with the "
                                          "turquoise border. ", "has a turquoise border and the label TARGET. "
                                          "The question is about that TARGET clip only. ")
    if a.v4c:
        V.HALO, V.OBJ_SCALE, V.COUNT_SETS = (30, 30, 30), 0.9, [[2, 4, 6, 8]]
    if a.count_even:
        V.COUNT_SETS = [[2, 4, 6, 8]]
    md5 = hashlib.md5(open(V.__file__, "rb").read() + open(V.G.__file__, "rb").read()).hexdigest()[:8]
    print("generator md5", md5, "model", a.model, flush=True)
    sc = Scorer(a.model)
    conds = a.conds.split(",")
    rows, t0 = [], time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    live = open(a.out.replace(".json", ".jsonl"), "a")
    for fam in a.families.split(","):
        for gi, g in enumerate(range(a.gstart, a.gstart + a.groups)):
            if gi == a.check_at:
                sel = [r for r in rows if r["family"] == fam]
                groups = {r["group"] for r in sel}

                def gc(c):
                    return 100 * sum(all(r[c] == r["answer"] for r in sel if r["group"] == h)
                                     for h in groups) / max(1, len(groups))
                ceil = gc("full")
                fl = max(gc(c) for c in ("blind", "window", "other") if c in conds)
                if ceil < 40 or fl >= ceil - 10:
                    print(f"EARLY-STOP {fam} after {gi} groups: ceiling grp {ceil:.0f}, floor grp {fl:.0f}",
                          flush=True)
                    break
            items = V.make_group(fam, g, a.split)
            streams = [V.stream(it) for it in items]
            cued = ([V.composite(s["ep_bg"], it["overlay"], V.CUE) for s, it in zip(streams, items)]
                    if any(c.startswith("oracle_all") for c in conds) else None)
            for v, it in enumerate(items):
                s, o = streams[v], streams[(v + 1) % len(items)]
                r = {k: it[k] for k in ("family", "group", "variant", "answer", "options", "vid", "start")}
                r["facts"] = it["facts"]
                for c in conds:
                    prompt = it["prompt"]
                    if c == "full":
                        pieces = [(None, s["full"])]
                    elif c == "oracle":
                        pieces = [(MEM_TXT, s["ep"]), (WIN_TXT, s["win"])]
                    elif c == "other":
                        pieces = [(MEM_TXT, o["ep"]), (WIN_TXT, s["win"])]
                    elif c == "window":
                        pieces = [(WIN_TXT, s["win"])]
                    elif c == "blind":
                        pieces = []
                    elif c.startswith("oracle_all|L"):
                        L = int(c.split("L")[-1])
                        if a.load_src == "variants":
                            dist = [streams[(v + 1 + (j % 3)) % 4]["ep"] for j in range(L)]
                        else:      # same family, other groups (other clips), random variant
                            dist = []
                            for j in range(L):
                                dg = V.make_group(fam, 900000 + 10 * g + j, a.split)
                                dist.append(V.stream(dg[(v + j + 1) % 4])["ep"])
                        seq = [cued[v]] + dist if a.load_order == "first" else dist + [cued[v]]
                        pieces = [(MEM_TXT, np.concatenate(seq)), (WIN_TXT, s["win"])]
                        prompt = it["prompt_cue"]
                    else:
                        raise ValueError(c)
                    lg, ntok = sc.score(pieces, prompt, len(it["options"]))
                    r[c] = int(lg.argmax())
                    r[c + "_lg"] = [round(x, 3) for x in lg.tolist()]
                    r[c + "_ntok"] = ntok
                rows.append(r)
                live.write(json.dumps(r) + "\n")
                live.flush()
        sel = [r for r in rows if r["family"] == fam]
        print(f"{fam:12s} n={len(sel)} " + " ".join(
            f"{c} {100 * sum(r[c] == r['answer'] for r in sel) / len(sel):.0f}" for c in conds)
            + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("\n" + table(rows, conds))
    json.dump({"args": vars(a), "md5": md5, "rows": rows}, open(a.out, "w"))
    print("json:", a.out, "\nCEILING-FINI", flush=True)


if __name__ == "__main__":
    main()
