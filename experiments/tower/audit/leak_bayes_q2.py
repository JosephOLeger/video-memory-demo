"""Bayes-optimal 'shortcut' accuracy per family from features that do NOT require
remembering the episode: final state (what a window on the end hold shows),
episode duration T, option text. Fit on groups [0,NTR), test on [NTR,NTR+NTE).
Scored per item and per counterfactual group (all 4 right)."""
import sys, collections, json, random
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "snap1")
import gen_scenes as G
NTR, NTE = 3000, 1000

def feats(it):
    ep, f = it["episode"], it["facts"]
    fam, opts = it["family"], tuple(it["options"])
    final_cols = {a.color for i, a in enumerate(ep.agents)
                  if (a.initial and a.color not in f["left"]) or (not a.initial)}
    present = tuple(int(any(c in o for c in final_cols)) if fam in ("first_left", "hat_color", "order") else 0 for o in opts)
    fin = (f["n_final"], present, f["door1"])
    return {"text": (opts,), "final": (opts, fin), "T": (opts, ep.n_frames),
            "final+T": (opts, fin, ep.n_frames),
            "start": (opts, f["n0"], f["door0"]),
            "final_mask": (present, f["n_final"]), "final_mask+T": (present, f["n_final"], ep.n_frames)}

def run(fam):
    tr = collections.defaultdict(collections.Counter)
    for g in range(NTR):
        for it in G.make_group(fam, g, G.Style(qset=2)):
            for k, v in feats(it).items():
                tr[(k, v)][it["answer"]] += 1
    acc = collections.defaultdict(list); grp = collections.defaultdict(list)
    for g in range(NTR, NTR + NTE):
        ok = collections.defaultdict(list)
        for it in G.make_group(fam, g, G.Style(qset=2)):
            for k, v in feats(it).items():
                c = tr.get((k, v))
                pred = c.most_common(1)[0][0] if c else 0
                ok[k].append(pred == it["answer"])
        for k, v in ok.items():
            acc[k] += v; grp[k].append(all(v))
    return {k: (round(100 * sum(v) / len(v), 1), round(100 * sum(grp[k]) / len(grp[k]), 1)) for k, v in acc.items()}

out = {}
for fam in (sys.argv[2].split(",") if len(sys.argv)>2 else G.FAMILIES):
    out[fam] = run(fam)
    print(fam, out[fam], flush=True)
json.dump(out, open("leak_bayes_qset2.json", "w"), indent=1)
