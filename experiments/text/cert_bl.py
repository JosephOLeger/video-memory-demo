"""Certification of the BABILong-like training stories (gen_babilong), host Qwen2.5-VL-3B, no graft (26/10).

Per task, on the DEV split (bAbI train stories id % 10 == 0, PG19 validation books), 4 counterfactual
variants per group, official prompt and metric:
    teacher      notes = all bAbI sentences of the variant, then the window          (KL target source)
    teacher_sup  notes = the supporting sentences only
    notes_other  notes of ANOTHER variant (v+1): must follow the notes, not the question (content control)
    full         the whole past + window as context                                  (full-context ceiling)
    window       window only                                                         (floor: no fact in it)
    nofact       background only: the past WITHOUT the fact sentences, as full context (answer not guessable)
Free answer (greedy 16, official compare_answers) per question, and option likelihood over the 4 official
answer sentences: per question and per group (best assignment of the 4 variants, chance 1/24).
Shortcut audit (no model): per group, are the token lengths of the 4 variants' fact sentences and their
positions identical? (mass / position shortcuts impossible when they are).
"""
import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_babilong as G  # noqa: E402

PERMS = list(itertools.permutations(range(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--gstart", type=int, default=10000)
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--D", type=int, default=300)
    ap.add_argument("--conds", default="teacher,teacher_sup,notes_other,full,window,nofact")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    conds = a.conds.split(",")
    fo = open(a.out, "w")
    audit = defaultdict(lambda: defaultdict(int))

    @torch.no_grad()
    def gen(ids):
        x = torch.tensor([ids], device="cuda")
        o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=16, do_sample=False)
        return tok.decode(o[0, x.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def opt_lk(ids, opts):
        seqs = [ids + tok.encode(o, add_special_tokens=False) for o in opts]
        m = max(len(s) for s in seqs)
        x = torch.tensor([s + [tok.pad_token_id] * (m - len(s)) for s in seqs], device="cuda")
        att = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs], device="cuda")
        lg = model(input_ids=x, attention_mask=att, use_cache=False).logits
        out = []
        for i, s in enumerate(seqs):
            lp = F.log_softmax(lg[i, len(ids) - 1:len(s) - 1].float(), -1)
            tgt = torch.tensor(s[len(ids):], device="cuda")
            out.append(float(lp.gather(1, tgt[:, None]).sum()))
        return out

    rows = []
    for g in range(a.gstart, a.gstart + a.groups):
        st = G.make_story(g, "eval")
        t = st["task"]
        its = st["items"][t]
        try:
            L = G.layout(st, tok, P=a.P, W=a.W, D=a.D)
        except ValueError as e:
            print("skip", g, e, flush=True)
            continue
        # shortcut audit
        fl = [[b - s for s, b in G.facts_of(l, t)] for l in L]
        fp = [[s for s, b in G.facts_of(l, t)] for l in L]
        audit[t]["groups"] += 1
        audit[t]["same_fact_lengths"] += int(all(x == fl[0] for x in fl))
        audit[t]["same_fact_positions"] += int(all(x == fp[0] for x in fp))
        audit[t]["fact_tokens"] += sum(fl[0])
        audit[t]["n_facts"] += len(fl[0])
        audit[t]["min_dist_last_fact_to_window"] += L[0]["D"]
        for v in range(4):
            it = its[v]
            Lv = L[v]
            notes = [x for s, b in G.facts_of(Lv, t) for x in Lv["past"][s:b]]
            fb = G.facts_of(Lv, t)
            sup = [x for i in it["support"] if i < len(fb) for x in Lv["past"][fb[i][0]:fb[i][1]]]
            Lo = L[(v + 1) % 4]
            notes_o = [x for s, b in G.facts_of(Lo, t) for x in Lo["past"][s:b]]
            nofact = [x for s, b, k in Lv["bounds"] if k == "scen" for x in Lv["past"][s:b]]
            ctx = {"teacher": G.teacher_ids(tok, notes, Lv["window"], it),
                   "teacher_sup": G.teacher_ids(tok, sup, Lv["window"], it),
                   "notes_other": G.teacher_ids(tok, notes_o, Lv["window"], it),
                   "full": G.student_ids(tok, Lv["past"] + Lv["window"], it),
                   "window": G.student_ids(tok, Lv["window"], it),
                   "nofact": G.student_ids(tok, nofact + Lv["window"], it)}
            r = {"group": g, "task": t, "variant": v, "answer": it["answer"], "answer_text": it["answer_text"],
                 "donor_answer": its[(v + 1) % 4]["answer"], "n_facts": it["n_facts"]}
            for c in conds:
                o = gen(ctx[c])
                r[c] = o
                r[c + "_ok"] = int(G.judge(it, o))
                r[c + "_lk"] = opt_lk(ctx[c], it["option_texts"])
            rows.append(r)
            fo.write(json.dumps(r) + "\n")
            fo.flush()
        if (g - a.gstart) % 10 == 9:
            print("done", g - a.gstart + 1, flush=True)
    # summary
    by = defaultdict(dict)
    for r in rows:
        by[(r["task"], r["group"])][r["variant"]] = r
    res = defaultdict(lambda: defaultdict(list))
    for (t, g), vs in by.items():
        if len(vs) < 4:
            continue
        for c in conds:
            res[t][c + "_free"].append(sum(vs[v][c + "_ok"] for v in range(4)) / 4)
            Lk = torch.tensor([vs[v][c + "_lk"] for v in range(4)])
            res[t][c + "_lk_q"].append(sum(int(Lk[v].argmax()) == vs[v]["answer"] for v in range(4)) / 4)
            best = max(PERMS, key=lambda p: sum(float(Lk[v, p[v]]) for v in range(4)))
            tie = bool((Lk - Lk[0:1]).abs().max() < 1e-6)         # identical rows (window only): no information
            res[t][c + "_group"].append(float(all(best[v] == vs[v]["answer"] for v in range(4)) and not tie))
            if c == "notes_other":
                res[t][c + "_follows_donor"].append(sum(int(Lk[v].argmax()) == vs[v]["donor_answer"] for v in range(4)) / 4)
    summ = {}
    for t, d in res.items():
        summ[t] = {k: round(100 * sum(v) / len(v), 1) for k, v in d.items()} | {"n_groups": len(d[conds[0] + "_free"])}
        au = audit[t]
        summ[t]["audit"] = {"groups": au["groups"], "same_fact_lengths": au["same_fact_lengths"],
                            "same_fact_positions": au["same_fact_positions"],
                            "mean_facts": round(au["n_facts"] / max(1, au["groups"]), 1),
                            "mean_fact_tokens": round(au["fact_tokens"] / max(1, au["groups"]), 1),
                            "mean_D": round(au["min_dist_last_fact_to_window"] / max(1, au["groups"]), 1)}
        print("CERT " + t + " " + json.dumps(summ[t]), flush=True)
    json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    print("CERT-BL-FINI", flush=True)


if __name__ == "__main__":
    main()
