"""Table by tag / area / class from res_*.jsonl -> bilan.md + bilan.json."""
import collections
import json
import math
import os
import sys

ROOT = os.path.expanduser("~/work/iclr/ptest")
res_files = sys.argv[1:] or [f"{ROOT}/res_sub3000.jsonl"]
Q = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{ROOT}/questions_all.jsonl")}
M = json.load(open(f"{ROOT}/mapping_gist_detail.json"))
CF = json.load(open(f"{ROOT}/counterfactual_groups.json"))
tag2cls = {t: c for c, d in M["mapping"].items() for t in d}

rows = {}
for fn in res_files:
    for l in open(fn):
        r = json.loads(l)
        if "error" in r:
            continue
        rows[r["qid"]] = r
C = ("full", "blind", "window")


def agg(qids):
    n = len(qids)
    out = {"n": n}
    for c in C:
        k = sum(rows[q][c]["correct"] for q in qids)
        p = k / n if n else float("nan")
        out[c] = p
        out[c + "_se"] = math.sqrt(p * (1 - p) / n) if n else float("nan")
    out["full_minus_blind"] = out["full"] - out["blind"]
    out["full_minus_window"] = out["full"] - out["window"]
    return out


def group(keyf):
    g = collections.defaultdict(list)
    for q in rows:
        for k in keyf(Q[q]):
            g[k].append(q)
    return {k: agg(v) for k, v in g.items()}


by_tag = group(lambda q: q["tags"])
by_area = group(lambda q: [q["area"]])
by_cls = group(lambda q: q["classes"])
overall = agg(list(rows))
for t, d in by_tag.items():
    d["class"] = tag2cls.get(t, "other")
    d["passes"] = bool(d["full"] >= 0.80 and d["full_minus_blind"] >= 0.25)
    cf = CF["per_tag"].get(t, {})
    d["frac_counterfactual_group"] = cf.get("frac_in_varying_group")

passing = sorted([t for t, d in by_tag.items() if d["passes"]], key=lambda t: -by_tag[t]["full"])
near = sorted([t for t, d in by_tag.items() if not d["passes"] and d["full"] >= 0.70
               and d["full_minus_blind"] >= 0.15], key=lambda t: -by_tag[t]["full"])
json.dump({"files": res_files, "n_scored": len(rows), "overall": overall, "by_tag": by_tag,
           "by_area": by_area, "by_class": by_cls, "passing": passing, "near_miss": near,
           "criterion": "full >= 0.80 and full - blind >= 0.25"},
          open(f"{ROOT}/bilan.json", "w"), indent=1, ensure_ascii=False)


def pct(x):
    return f"{100*x:.0f}"


def line(name, d, extra=""):
    return (f"| {name} | {extra}{d['n']} | {pct(d['full'])} | {pct(d['blind'])} | "
            f"{pct(d['window'])} | 33 | {100*d['full_minus_blind']:+.0f} | "
            f"{100*d['full_minus_window']:+.0f} |")


hdr = ("| famille | n | full | blind | window | hasard | full−blind | full−window |\n"
       "|---|---|---|---|---|---|---|---|")
out = []
out.append(f"Questions évaluées : {len(rows)}. Fichiers : {', '.join(os.path.basename(f) for f in res_files)}.\n")
out.append("## Par tag (skill officiel), groupé par classe\n")
for cls in ("gist", "detail", "other"):
    out.append(f"### {cls}\n")
    out.append(hdr)
    for t in sorted([t for t in by_tag if by_tag[t]["class"] == cls], key=lambda t: -by_tag[t]["full"]):
        d = by_tag[t]
        mark = " **✓**" if d["passes"] else ""
        out.append(line(t + mark, d))
    out.append("")
out.append("## Par area\n")
out.append(hdr)
for k, d in sorted(by_area.items()):
    out.append(line(k, d))
out.append(line("**tout**", overall))
out.append("\n## Par classe (une question à plusieurs tags compte dans chaque classe)\n")
out.append(hdr)
for k, d in sorted(by_cls.items()):
    out.append(line(k, d))
out.append(f"\n**Passent** (full ≥ 80 et full − blind ≥ 25) : {', '.join(passing) or 'aucune'}\n")
out.append(f"Proches (full ≥ 70 et full − blind ≥ 15) : {', '.join(near) or 'aucune'}\n")
open(f"{ROOT}/bilan_table.md", "w").write("\n".join(out))
print("\n".join(out))
