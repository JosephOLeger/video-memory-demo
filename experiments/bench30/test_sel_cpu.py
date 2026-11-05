"""03/11 CPU check of sel_write on dev records (tokenizer only, no model weights, no GPU)."""
import json
import os
import sys
import importlib
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.expanduser("~/work/iclr/bench30"))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import sel_write as SW  # noqa: E402

G = importlib.import_module("gen_bench30")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
W = 256
recs = [json.loads(x) for x in open(os.path.expanduser("~/work/iclr/bench30/evalsets/b30_dev_4k_1ep_sp.jsonl"))]
# spread over types: the first record of each type, then the second, ... (N records)
byt = defaultdict(list)
for r in recs:
    byt[r["type"]].append(r)
recs = [byt[t][j] for j in range(20) for t in sorted(byt) if j < len(byt[t])][:N]
words = SW.Words(tok)


def orc_pos(bb, past):        # train_b30 orc_pos for deep + --b30-content, no oracle flags
    pos0 = [p for b0, b1 in bb for p in range(b0, b1)]
    if len(pos0) > 1:
        m = G.content_mask(tok, [past[p] for p in pos0])
        return [p for p, k in zip(pos0, m) if k] or pos0[-1:]
    return pos0


# host_nll index logic, with a fake "model" whose NLL output at row j is the target's absolute position
def nll_index_check(P, Wc, mode):
    out = torch.full((P,), float("nan"))
    ctxs = {}
    blk = ([(max(0, s0 - Wc) if mode == "slide" else s0, s0, min(P, s0 + Wc)) for s0 in range(0, P, Wc)]
           if Wc and P > Wc else [(0, 0, P)])
    for c0, s0, e0 in blk:
        nl = torch.arange(c0 + 1, e0, dtype=torch.float)     # row j predicts x[c0 + 1 + j]
        lo = max(s0, c0 + 1)
        out[lo:e0] = nl[lo - c0 - 1:]
        for p in range(lo, e0):
            ctxs[p] = c0
    ok = all(int(out[p]) == p for p in range(P) if p in ctxs)
    cov = sorted(ctxs) == ([p for p in range(1, P)] if mode == "slide" or not Wc else
                           [p for p in range(1, P) if p % Wc != 0])
    ctx_ok = all(ctxs[p] == (max(0, (p // Wc) * Wc - Wc) if mode == "slide" else (p // Wc) * Wc) for p in ctxs) if Wc else True
    return ok and cov and ctx_ok and torch.isnan(out[0])


print("host_nll index check slide/chunk/none:", nll_index_check(4096, 256, "slide"), nll_index_check(4096, 256, "chunk"),
      nll_index_check(300, 0, "slide"))

# unigram rarity over the 600 dev pasts: a CPU stand-in for surprise (NOT the host NLL)
allr = [json.loads(x) for x in open(os.path.expanduser("~/work/iclr/bench30/evalsets/b30_dev_4k_1ep_sp.jsonl"))]
cnt = defaultdict(int)
tot = 0
for r in allr:
    for t in r["ids"][:-W]:
        cnt[t] += 1; tot += 1
import math  # noqa: E402

agg = defaultdict(lambda: defaultdict(list))
for i, rec in enumerate(recs):
    ctx = rec["ids"]; past = ctx[:-W]
    sup = orc_pos(rec["sup_spans"], past)
    fact = orc_pos(rec["fact_spans"], past)
    k = len(sup)
    cands = SW.candidates(len(past), G.content_mask(tok, past))
    assert set(sup) <= set(cands) | {0}, "oracle tokens outside the content pool"
    uni = [float("nan")] + [-math.log(cnt[t] / tot) for t in past[1:]]
    sels = {"reref": SW.sel_reref(words, ctx, cands, k, G.STOP | SW.EXTRA_STOP)[0],
            "rand": SW.sel_rand(cands, k, (0, rec["id"])),
            "unigram_proxy": SW.sel_nll(uni, cands, k),
            "unigram_proxy_sent": SW.sel_nllsent(words, past, uni, cands, k),
            "reref_tie_uni": SW.sel_reref(words, ctx, cands, k, G.STOP | SW.EXTRA_STOP, uni)[0]}
    assert SW.sel_rand(cands, k, (0, rec["id"])) == sels["rand"]           # fixed seed per record
    assert all(len(v) == k for v in sels.values()), {n: len(v) for n, v in sels.items()}
    line = [f"{rec['id']:<24} k={k:3d} cand={len(cands)} chance={len(set(sup)) / len(cands):.3f}"]
    for n, s in sels.items():
        o, of = SW.overlap(s, sup), SW.overlap(s, fact)
        agg[n]["p_sup"].append(o["p"]); agg[n]["r_sup"].append(o["r"]); agg[n]["p_fact"].append(of["p"])
        line.append(f"{n} {o['p']:.2f}/{of['p']:.2f}")
    agg["chance"]["p_sup"].append(len(set(sup)) / len(cands))
    print("  ".join(line))
    if i < 3:
        print("   sup words  :", [words.norm(past[p]) for p in sup])
        print("   reref words:", [words.norm(past[p]) for p in sels["reref"]])
        print("   rand words :", [words.norm(past[p]) for p in sels["rand"]])
print(f"\nmean over {len(recs)} records (precision vs support tokens / recall vs support / precision vs fact tokens):")
for n, d in agg.items():
    print(f"  {n:<20} " + "  ".join(f"{m}={100 * sum(v) / len(v):5.1f}" for m, v in d.items()))
