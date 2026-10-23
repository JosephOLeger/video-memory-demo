"""MANDATORY GATE for the v4 composited generator.
  1. answer == f(facts) asserted in make_group; every option gold equally often (N groups/split);
  2. prompt (plain and cue) identical across a group's variants -> text-only Bayes = chance, grp 0;
  3. pre / post / window footage pixel-identical across variants (they are the same clip);
  4. inside the episode, variants differ ONLY inside the union of their overlay masks;
  5. stream length constant; train/eval background videos disjoint, and disjoint from the
     ladder's 120 filler videos; eval backgrounds disjoint from the graft_v3 train pool rule.
    python leak_gate_v4.py
"""
import collections
import sys

import numpy as np

import gen_composite_v4 as V

N, NPIX = 2000, 6


def main():
    ok = True
    tr, ev = {v for v, _ in V.pool("train")}, {v for v, _ in V.pool("eval")}
    lad = V._ladder_ids()
    c5 = not (tr & ev) and not (tr & lad) and not (ev & lad) and all(V.is_eval(v) for v in ev)
    print(f"pools train {len(tr)} eval {len(ev)} | disjoint train/eval/ladder120, eval all is_eval: {c5}")
    ok &= c5
    for split in ("train", "eval"):
        for fam in V.FAMILIES:
            bal = collections.Counter()
            for g in range(N):
                its = V.make_group(fam, g, split)
                assert len({i["prompt"] for i in its}) == 1 and len({i["prompt_cue"] for i in its}) == 1
                assert len({(i["vid"], i["start"]) for i in its}) == 1
                for i in its:
                    bal[i["answer"]] += 1
            same, inmask, lens = [], [], set()
            for g in range(NPIX):
                its = V.make_group(fam, g, split)
                st = [V.stream(i) for i in its]
                lens |= {s["full"].shape[0] for s in st}
                same.append(all(np.array_equal(st[0][k], s[k]) for s in st[1:] for k in ("pre", "post", "win")))
                mask = np.zeros((V.E * V.FPS, V.SIZE, V.SIZE), bool)
                for i in its:
                    for f in range(V.E * V.FPS):
                        mask[f] |= np.asarray(V.overlay_rgba(i["overlay"], f))[..., 3] > 0
                diff = np.zeros_like(mask)
                for s in st[1:]:
                    diff |= (s["ep"] != st[0]["ep"]).any(-1)
                inmask.append(not (diff & ~mask).any())
            vals = list(bal.values())
            p = max(vals) == min(vals) and all(same) and all(inmask) and len(lens) == 1
            ok &= p
            print(f"{split:5s} {fam:12s} balance {dict(sorted(bal.items()))} surroundings identical {all(same)} "
                  f"differences only inside overlays {all(inmask)} lens {sorted(lens)} -> {'PASS' if p else 'FAIL'}",
                  flush=True)
    print("text-only Bayes: prompt constant within every group -> item = chance, group = 0 by construction")
    print("LEAK GATE v4:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
