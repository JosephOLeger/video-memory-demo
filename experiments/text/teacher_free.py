"""Teacher free-answer accuracy per family (normalized), facts visible in context, train split."""
import sys, os, random, json, torch
sys.argv = ["x", "--arm", "A", "--out", "/dev/null"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "pylib"))
import train_text_ab as T, gen_story2 as G
from collections import defaultdict
host = T.TextHost("Qwen/Qwen2.5-VL-3B-Instruct", "cuda"); tok = host.tok
acc, ex = defaultdict(list), defaultdict(list)
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60):
    st = G.make_story(2_000_000 + i, "train"); r = random.Random(i)
    lays = G.layout(st, tok, P=8192, W=256, D=r.randrange(0, 6001)); v = r.randrange(4); win = lays[0]["window"]
    seqs, its = [], []
    for f in G.FAMILIES:
        it = st["items"][f][v]; facts = [x for s, e in G.facts_of(lays[v], f) for x in lays[v]["past"][s:e]]
        seqs.append(T.build_teacher(host, facts, win, it)[0]); its.append(it)
    tok.padding_side = "left"
    m = max(len(s) for s in seqs)
    x = torch.tensor([[host.pad] * (m - len(s)) + s for s in seqs], device="cuda")
    att = torch.tensor([[0] * (m - len(s)) + [1] * len(s) for s in seqs], device="cuda")
    with torch.no_grad():
        out = host.model.generate(input_ids=x, attention_mask=att, max_new_tokens=12, do_sample=False)
    for j, it in enumerate(its):
        o = tok.decode(out[j, m:], skip_special_tokens=True)
        ok = T.ans_ok(it["answer_text"], o); acc[it["family"]].append(ok)
        if not ok and len(ex[it["family"]]) < 3: ex[it["family"]].append((it["answer_text"], o))
for f in G.FAMILIES:
    print(f"{f:12s} teacher free-answer {100*sum(acc[f])/len(acc[f]):5.1f}%  n={len(acc[f])}  errors: {ex[f]}")
