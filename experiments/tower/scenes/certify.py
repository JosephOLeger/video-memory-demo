"""Certification table (lead criterion, 22/10), at GROUP level (all variants right):
ceiling = full (episode in its natural stream position), also oracle;
floor   = max(blind, window, other-variant memory);
margin  = ceiling - floor; CERTIFIED if ceiling >= 80, margin >= 50.
    python certify.py results/v3.json [results/v3_7b.json ...]
"""
import json
import sys
from collections import defaultdict


def grp(sel, c):
    g = defaultdict(list)
    for r in sel:
        g[r["group"]].append(r[c] == r["answer"])
    return 100 * sum(all(v) for v in g.values()) / len(g)


def item(sel, c):
    return 100 * sum(r[c] == r["answer"] for r in sel) / len(sel)


for path in sys.argv[1:]:
    if path.endswith(".jsonl"):        # live partial file
        rows = [json.loads(l) for l in open(path) if l.strip()]
        d = {"args": {"model": "(partial)"}, "md5": None}
        if not rows:
            continue
    else:
        d = json.load(open(path))
        rows = d["rows"]
    print(f"\n## {path}  model={d['args']['model']}  md5={d.get('md5')}")
    print("| family | groups | ceiling full grp (item) | oracle grp (item) | blind grp | window grp | other grp (item) | other->donor % | floor grp | margin | certified |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for fam in dict.fromkeys(r["family"] for r in rows):
        sel = [r for r in rows if r["family"] == fam]
        ng = len({r["group"] for r in sel})
        have = lambda c: c in sel[0]       # noqa: E731
        # drop a trailing incomplete group
        cnt = defaultdict(int)
        for r in sel:
            cnt[r["group"]] += 1
        full_n = max(cnt.values())
        sel = [r for r in sel if cnt[r["group"]] == full_n]
        if not sel:
            continue
        ng = len({r["group"] for r in sel})
        fl = max(grp(sel, c) for c in ("blind", "window", "other") if have(c))
        ceil = grp(sel, "full")
        # donor rate: other-variant memory answers the DONOR's gold option
        n = len({r["variant"] for r in sel})
        byg = defaultdict(dict)
        for r in sel:
            byg[r["group"]][r["variant"]] = r
        don = [r["other"] == byg[r["group"]][(r["variant"] + 1) % n]["answer"] for r in sel] if have("other") else []
        ok = ceil >= 80 and ceil - fl >= 50
        print(f"| {fam} | {ng} | {ceil:.0f} ({item(sel, 'full'):.0f}) | "
              f"{(f"{grp(sel, 'oracle'):.0f} ({item(sel, 'oracle'):.0f})" if have('oracle') else '-')} | {grp(sel, 'blind'):.0f} | "
              f"{grp(sel, 'window'):.0f} | " + (f"{grp(sel, 'other'):.0f} ({item(sel, 'other'):.0f}) | "
              f"{100 * sum(don) / len(don):.0f} | " if don else "- | - | ")
              + f"{fl:.0f} | {ceil - fl:.0f} | {'YES' if ok else 'no'} |"
              + "".join(f" {c}: {grp(sel, c):.0f} ({item(sel, c):.0f})" for c in sel[0]
                        if c.startswith("oracle_all|") and "_" not in c.split("|")[1]))
