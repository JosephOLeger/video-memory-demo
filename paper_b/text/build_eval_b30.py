"""bench30 (29/10): fixed DEV eval files (eval split: salted story seeds, PG19 validation background) + layout checks.

Each record = one story of 2 episodes (the target type + one distractor episode of any type), one variant, past-only
placement (every fact segment >= D tokens before the window, D uniform in [64, 1500]), P past tokens + W window.
Records are interleaved across types, so in the eval the 'omem' memory (previous record) is another story.
    python build_eval_b30.py --P 4096 --n 30 --out evalsets/b30_dev_4k.jsonl
"""
import argparse
import collections
import json
import random
import re

import gen_bench30 as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-ep", type=int, default=2, dest="n_ep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=7_000_000)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    recs = collections.defaultdict(list)
    chk = collections.Counter()
    for ti, typ in enumerate(G.FAMILIES):
        for k in range(a.n):
            g = a.seed0 + ti * 1000 + k
            r = random.Random(g)
            other = r.choice(G.FAMILIES)
            st = G.make_story(g, "eval", types=[typ, other], n_ep=a.n_ep)
            D = r.randrange(64, 1501)
            L = G.layout(st, tok, P=a.P, W=a.W, D=D)
            v = k % 4
            fam = f"ep#{typ}#0"
            it = st["items"][fam][v]
            lay = L[v]
            ids = lay["past"] + lay["window"]
            assert len(lay["past"]) == a.P and len(lay["window"]) == a.W
            fb = G.facts_of(lay, "ep")
            # checks: every segment is in the past and decodes to the rendered fact; support present; no fact in window
            chk["stories"] += 1
            if len(fb) != len(st["facts"][v]):
                chk["n_segments_mismatch"] += 1
            for (s_, e_), f in zip(fb, st["facts"][v]):
                if tok.decode(lay["past"][s_:e_]).strip() != f.strip():
                    chk["segment_decode_mismatch"] += 1
            wtxt = tok.decode(lay["window"]).lower()
            if any(f.lower()[:40] in wtxt for f in st["facts"][v]):
                chk["fact_in_window"] += 1
            if it["answer_text"].lower() in wtxt and not it["yn"]:
                chk["answer_string_in_window_by_chance"] += 1
            chk["D_min_ok"] += int(lay["D"] >= min(D, 64))
            recs[typ].append(dict(id=f"{typ}-{k}", type=typ, held=it["held"], form=it["form"], other_type=other,
                                  question=it["question"], answer=it["answer_text"], accept=it["accept"], yn=it["yn"],
                                  allkeys=it["allkeys"], universe=it["universe"], options=it["options"],
                                  support=it["support"], D=lay["D"], P=a.P, ids=ids,
                                  fact_spans=[list(x) for x in fb], sup_spans=[list(fb[i]) for i in it["support"] if i < len(fb)]))
    out = []
    for k in range(a.n):
        for typ in G.FAMILIES:
            out.append(recs[typ][k])
    with open(a.out, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print("CHECKS", dict(chk))
    print("wrote", len(out), "records ->", a.out)


if __name__ == "__main__":
    main()
