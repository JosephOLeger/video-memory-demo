"""03/11 CPU: reref variants (proposals only, not in train_b30_sel.py): window-only mentions, one token per word type."""
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
    m = G.content_mask(tok, [past[p] for p in pos0])
    return [p for p, k in zip(pos0, m) if k] or pos0[-1:]
agg = defaultdict(list)
for rec in recs:
    ctx = rec["ids"]; past = ctx[:-W]; sup = set(orc_pos(rec["sup_spans"], past)); k = len(sup)
    cands = SW.candidates(len(past), G.content_mask(tok, past))
    good = lambda w: len(w) >= 2 and any(c.isalnum() for c in w) and w not in stop
    wv = set(words.norm(t) for t in ctx[-W:])
    cnt = SW.reref_counts(words, ctx, cands, stop)
    # window-only: later mentions restricted to the window; ties by count over the rest of the past
    winc = defaultdict(int)
    for t in ctx[-W:]:
        winc[words.norm(t)] += 1
    v_win = sorted(cands, key=lambda p: (-(winc[words.norm(past[p])] if good(words.norm(past[p])) else 0), -cnt[p], -p))[:k]
    # one token per word type: the LAST occurrence that is still re-mentioned, types ranked by later mentions
    seen, uniq = set(), []
    for p in sorted(cands, key=lambda p: (-cnt[p], -p)):
        w = words.norm(past[p])
        if w in seen or not good(w):
            continue
        seen.add(w); uniq.append(p)
    v_uniq = uniq[:k]
    for n, s in (("reref_window", v_win), ("reref_uniq", v_uniq)):
        agg[n].append(len(set(s) & sup) / max(1, len(s)))
for n, v in agg.items():
    print(f"{n:<14} p_sup={100 * sum(v) / len(v):.1f} (n={len(v)})")
