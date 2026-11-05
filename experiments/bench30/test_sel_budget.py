"""03/11 CPU: recall of support / fact tokens vs fixed budget N, token and sentence units (tokenizer only)."""
import json, os, sys, importlib
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/work/iclr/bench30"))
from transformers import AutoTokenizer
import sel_write as SW
G = importlib.import_module("gen_bench30")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
W = 256
recs = [json.loads(x) for x in open(os.path.expanduser("~/work/iclr/bench30/evalsets/b30_dev_4k_1ep_sp.jsonl"))]
words = SW.Words(tok); stop = G.STOP | SW.EXTRA_STOP
def orc_pos(bb, past):
    pos0 = [p for b0, b1 in bb for p in range(b0, b1)]
    if len(pos0) > 1:
        m = G.content_mask(tok, [past[p] for p in pos0])
        return [p for p, k in zip(pos0, m) if k] or pos0[-1:]
    return pos0
agg = defaultdict(lambda: defaultdict(list))
for rec in recs:
    ctx = rec["ids"]; past = ctx[:-W]
    sup, fact = set(orc_pos(rec["sup_spans"], past)), set(orc_pos(rec["fact_spans"], past))
    cands = SW.candidates(len(past), G.content_mask(tok, past))
    for N in ("sup", 64, 256, 512):
        k = len(sup) if N == "sup" else N
        key = (0, rec["id"])
        r_tok, _, cnt = SW.sel_reref(words, ctx, cands, k, stop)
        sels = {"reref/tok": r_tok, "rand/tok": SW.sel_rand(cands, k, key),
                "reref/sent": SW.sel_sent(words, past, cands, k, cnt), "rand/sent": SW.sel_sent(words, past, cands, k, None, key)}
        for n, s in sels.items():
            assert len(s) == min(k, len(cands)), (n, len(s), k)
            s = set(s)
            agg[(N, n)]["r_sup"].append(len(s & sup) / len(sup)); agg[(N, n)]["r_fact"].append(len(s & fact) / len(fact))
        agg[(N, "chance")]["r_sup"].append(min(1, k / len(cands))); agg[(N, "chance")]["r_fact"].append(min(1, k / len(cands)))
    agg[("sup", "_k")]["k"].append(len(sup))
print("mean sup budget:", sum(agg[("sup", "_k")]["k"]) / 600)
for (N, n), d in agg.items():
    if n != "_k":
        print(f"N={N!s:<4} {n:<11} " + "  ".join(f"{m}={100 * sum(v) / len(v):5.1f}" for m, v in d.items()))
