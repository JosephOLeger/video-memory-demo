"""Group / item accuracy per family for any condition present in result jsons."""
import collections
import json
import sys

for p in sys.argv[1:]:
    rows = json.load(open(p))["rows"]
    print("==", p)
    for fam in dict.fromkeys(r["family"] for r in rows):
        s = [r for r in rows if r["family"] == fam]
        conds = [k for k in s[0] if k + "_lg" in s[0]]
        out = []
        for c in conds:
            g = collections.defaultdict(list)
            for r in s:
                g[r["group"]].append(r[c] == r["answer"])
            out.append(f"{c} {100 * sum(all(v) for v in g.values()) / len(g):.0f} "
                       f"({100 * sum(r[c] == r['answer'] for r in s) / len(s):.0f})")
        print(f"   {fam} [{len(s) // 4} grp]: " + " | ".join(out))
