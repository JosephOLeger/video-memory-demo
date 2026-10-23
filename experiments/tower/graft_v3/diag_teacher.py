"""Teacher diagnosis (23/10): is the free-form oracle's 50 % on gist a FORMAT artefact?
Per question of fresh rich training scenes: greedy free answer (strict exact / lenient normalised
match), teacher-forced p(gold first token), and the same question lettered (bench MCQ style).
    AB_V3C_DIR=ab_gen_v3cb python diag_teacher.py --groups 60
"""
import argparse, json, random, sys, os
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_ab as PA, gen_ab as GA

ap = argparse.ArgumentParser()
ap.add_argument("--groups", type=int, default=60)
ap.add_argument("--gstart", type=int, default=20_000_000)
ap.add_argument("--out", default="logs_ab/diag_teacher.jsonl")
a = ap.parse_args()
h = PA.Host(None)
print("gen", GA.GEN_MD5, "v3c design", GA.V3C_DESIGN, flush=True)
lid = [h.tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
rows = []
fo = open(a.out, "w")
for g in range(a.gstart, a.gstart + a.groups):
    items = GA.make_group_ab(g)
    win = PA.T.render_filler(f"diag-{g}", 8, 0)
    wv = h.vis(win)
    for v, it in enumerate(items[:2]):
        tab = PA.render_tab(it)
        tv = h.vis(tab)
        pieces = [(PA.MEM_TXT, tab), (PA.WIN_TXT, win)]
        qs = GA.questions_for(it, random.Random(f"diag-{g}-{v}"))
        tt = PA.teacher_topk(h, pieces, qs, vis=[tv, wv])
        for q, t in zip(qs, tt):
            pred = PA.greedy(h, pieces, q["prompt"])
            r = {"qfam": q["qfam"], "kind": q["kind"], "gold": q["answer"], "pred": pred,
                 "strict": pred.strip() == q["answer"], "lenient": GA.free_match(pred, q["answer"]),
                 "p_gold0": float(t["gold_lp"][0].exp()), "forced_ok": t["argmax_ok"], "n": it["facts"]["n"]}
            m = GA.mcq_for(q, it["facts"], random.Random(f"m-{g}-{v}-{q['qfam']}"), q["intro"])
            if m:
                lg = PA.mcq_letters(h, pieces, m[0], len(m[1]))
                r["mcq_ok"] = int(lg.argmax()) == m[2]
                r["mcq_p"] = float(torch.softmax(lg, -1)[m[2]])
            rows.append(r)
            fo.write(json.dumps(r) + "\n")
    fo.flush()
    if (g - a.gstart) % 10 == 9:
        print(f"  {g - a.gstart + 1} groups", flush=True)
by = defaultdict(list)
for r in rows:
    by[r["qfam"]].append(r)
print(f"{'family':14s} {'n':>4s} {'strict':>7s} {'lenient':>8s} {'forced':>7s} {'p_gold':>7s} {'mcq':>6s} {'mcq_p':>6s}")
for k, rs in sorted(by.items(), key=lambda kv: (kv[1][0]["kind"], kv[0])):
    mc = [r for r in rs if "mcq_ok" in r]
    print(f"{k:14s} {len(rs):4d} {100*np.mean([r['strict'] for r in rs]):7.1f} {100*np.mean([r['lenient'] for r in rs]):8.1f} "
          f"{100*np.mean([r['forced_ok'] for r in rs]):7.1f} {np.mean([r['p_gold0'] for r in rs]):7.2f} "
          f"{100*np.mean([r['mcq_ok'] for r in mc]):6.1f} {np.mean([r['mcq_p'] for r in mc]):6.2f}")
cnt = defaultdict(list)
for r in by["count"]:
    cnt[r["n"]].append(r["lenient"])
print("count lenient by n:", {k: f"{100*np.mean(v):.0f}% ({len(v)})" for k, v in sorted(cnt.items())})
print("DIAG-FINI")
