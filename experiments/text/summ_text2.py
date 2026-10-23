"""v2 'summary' memory: the frozen host keeps running notes over a joint story (one summary serves the
8 questions). Same neutral instruction as summ_text.py. Chunks of C tokens, notes <= B tokens, the 4
variants generated as one batch. Writes ladder rows with cond summ@D.

    CUDA_VISIBLE_DEVICES=3 python summ_text2.py --groups 20 --out runs2/summ.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_story2 as G  # noqa: E402
from ladder_text2 import Host2, prompt_of, GEN_MD5, SYS, END  # noqa: E402
from summ_text import INSTR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gstart", type=int, default=10_000)
    ap.add_argument("--past", type=int, default=8192)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--dists", default="0,6000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = Host2()
    tok = h.tok
    pad = tok.pad_token_id
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    done = set()
    if os.path.exists(a.out):
        done = {(r["group"], r["cond"]) for r in map(json.loads, open(a.out))}
    fo = open(a.out, "a")
    words = int(a.budget * 0.6)
    t0 = time.time()
    for g in range(a.gstart, a.gstart + a.groups):
        st = G.make_story(g, "eval")
        for D in [int(x) for x in a.dists.split(",")]:
            cond = f"summ@{D}"
            if (g, cond) in done:
                continue
            lays = G.layout(st, tok, P=a.past, W=a.window, D=D)
            notes = [[] for _ in range(4)]
            for c0 in range(0, a.past, a.chunk):
                batch = []
                for v in range(4):
                    u = (h._t(INSTR.format(w=words) + "\n\nOld notes:\n") + (notes[v] or h._t("(none)"))
                         + h._t("\n\nNew piece of the story:\n") + lays[v]["past"][c0:c0 + a.chunk]
                         + h._t("\n\nNew notes:"))
                    batch.append(h._t(SYS) + u + h._t(END))
                m = max(len(x) for x in batch)
                x = torch.tensor([[pad] * (m - len(b)) + b for b in batch], device="cuda")
                att = torch.tensor([[0] * (m - len(b)) + [1] * len(b) for b in batch], device="cuda")
                with torch.no_grad():
                    out = h.model.generate(input_ids=x, attention_mask=att, max_new_tokens=a.budget, do_sample=False)
                for v in range(4):
                    gen = [t for t in out[v, m:].tolist() if t != pad]
                    if im_end in gen:
                        gen = gen[:gen.index(im_end)]
                    notes[v] = [t for t in gen if t != tok.eos_token_id][:a.budget]
            win = lays[0]["window"]
            for v in range(4):
                its = [st["items"][f][v] for f in G.FAMILIES]
                lgs, n = h.score_many(h.ctx_ids(notes[v], win), [prompt_of(it) for it in its])
                for it, lg in zip(its, lgs):
                    pred = int(lg.argmax())
                    fo.write(json.dumps({"family": it["family"], "kind": it["kind"], "group": g, "variant": v,
                                         "answer": it["answer"], "cond": cond, "pred": pred,
                                         "correct": int(pred == it["answer"]), "lg": [round(x, 3) for x in lg.tolist()],
                                         "ntok": n, "mem_tok": len(notes[v]), "gen": GEN_MD5,
                                         "notes": tok.decode(notes[v]), "D": lays[v]["D"]}) + "\n")
            fo.flush()
            print(f"g{g} D{D} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    print("SUMM2-FINI", flush=True)


if __name__ == "__main__":
    main()
