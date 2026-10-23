"""Design sweep: several Style configs in one process (model loaded once).

    CUDA_VISIBLE_DEVICES=1 python sweep.py --configs '[{"qset":2},{"qset":2,"clock":true}]' \
        --groups 10 --out results/sweep1.json
"""
import argparse
import json
import time

import torch

import gen_scenes as G
from ceiling import letter_logits, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--families", default=",".join(G.GIST))
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--gstart", type=int, default=100, help="sweep on held-out seeds")
    ap.add_argument("--conds", default="full,last")
    ap.add_argument("--last-s", type=float, default=4.0, dest="last_s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    lid = torch.tensor([proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in G.LET])
    conds = a.conds.split(",")
    nlast = int(round(a.last_s * G.FPS))
    allres = []
    for cfg in json.loads(a.configs):
        intro = cfg.pop("intro", None)
        if intro is not None:
            G.INTRO = intro
        st = G.Style(**cfg)
        rows, t0 = [], time.time()
        for fam in a.families.split(","):
            for g in range(a.gstart, a.gstart + a.groups):
                for it in G.make_group(fam, g, st):
                    fr = G.render(it["episode"])
                    r = {k: it[k] for k in ("family", "group", "variant", "answer", "options")}
                    for c in conds:
                        x = {"full": fr, "blind": None, "last": fr[-nlast:]}[c]
                        lg, _ = letter_logits(model, proc, x, it["prompt"], len(it["options"]),
                                              G.FPS, "cuda", lid)
                        r[c] = int(lg.argmax())
                    rows.append(r)
        print(f"\n### config {cfg} intro={intro!r}  ({(time.time() - t0) / 60:.1f} min)")
        print(table(rows, conds), flush=True)
        allres.append({"cfg": cfg, "intro": intro, "rows": rows})
    json.dump(allres, open(a.out, "w"))
    print("SWEEP-FINI", flush=True)


if __name__ == "__main__":
    main()
