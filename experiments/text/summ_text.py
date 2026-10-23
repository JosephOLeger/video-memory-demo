"""Untrained 'summary' memory: the frozen host keeps running notes (ReadAgent-like gist memory).

The past is read in chunks of C tokens; after each chunk the host rewrites its notes (greedy, at
most B new tokens) from (old notes, chunk). The 4 variants of a group are generated as one batch.
The instruction is neutral (no hint of the question families). The final notes are the memory,
scored exactly like the other ladder memories (cond summ@D). Writes the same jsonl rows.

    CUDA_VISIBLE_DEVICES=3 python summ_text.py --groups 2 --out runs/summ_smoke.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_story as G  # noqa: E402
from ladder_text import Host, prompt_of, GEN_MD5  # noqa: E402

INSTR = ("You are reading a long story in pieces and keeping short notes, because you will not be able to "
         "read the earlier pieces again. Rewrite your notes so that they include everything important from "
         "the old notes and from the new piece. Use at most {w} words. Write only the notes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(G.FAMILIES))
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gstart", type=int, default=10_000)
    ap.add_argument("--past", type=int, default=3072)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--dist", type=int, default=1536)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = Host()
    tok = h.tok
    tok.padding_side = "left"
    fo = open(a.out, "a")
    done = set()
    if os.path.exists(a.out):
        done = {(r["family"], r["group"]) for r in map(json.loads, open(a.out))}
    words = int(a.budget * 0.6)
    t0 = time.time()
    for g in range(a.gstart, a.gstart + a.groups):
        for fam in a.families.split(","):
            if (fam, g) in done:
                continue
            items = G.make_group(fam, g, "eval")
            lays = G.layout(items, tok, P=a.past, W=a.window, D=a.dist)
            notes = [[] for _ in items]
            for c0 in range(0, a.past, a.chunk):
                batch = []
                for v in range(4):
                    piece = lays[v]["past"][c0:c0 + a.chunk]
                    u = (h._t(INSTR.format(w=words) + "\n\nOld notes:\n") + (notes[v] or h._t("(none)"))
                         + h._t("\n\nNew piece of the story:\n") + piece + h._t("\n\nNew notes:"))
                    ids = (h._t("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
                           + u + h._t("<|im_end|>\n<|im_start|>assistant\n"))
                    batch.append(ids)
                m = max(len(x) for x in batch)
                pad = tok.pad_token_id
                x = torch.tensor([[pad] * (m - len(b)) + b for b in batch], device="cuda")
                att = torch.tensor([[0] * (m - len(b)) + [1] * len(b) for b in batch], device="cuda")
                with torch.no_grad():
                    out = h.model.generate(input_ids=x, attention_mask=att, max_new_tokens=a.budget,
                                           do_sample=False)
                for v in range(4):
                    gen = out[v, m:].tolist()
                    gen = [t for t in gen if t not in (pad, tok.eos_token_id)]
                    im_end = tok.convert_tokens_to_ids("<|im_end|>")
                    if im_end in gen:
                        gen = gen[:gen.index(im_end)]
                    notes[v] = gen[:a.budget]
            win = lays[0]["window"]
            for v, it in enumerate(items):
                lg, nt = h.score(h.build(notes[v], win, prompt_of(it)))
                pred = int(lg.argmax())
                r = {"family": fam, "kind": it["kind"], "group": g, "variant": v, "answer": it["answer"],
                     "cond": f"summ@{a.dist}", "pred": pred, "correct": int(pred == it["answer"]),
                     "lg": [round(x, 3) for x in lg.tolist()], "ntok": nt, "mem_tok": len(notes[v]),
                     "gen": GEN_MD5, "notes": tok.decode(notes[v]), "D": lays[v]["D"]}
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(f"{fam} g{g} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    print("SUMM-FINI", flush=True)


if __name__ == "__main__":
    main()
