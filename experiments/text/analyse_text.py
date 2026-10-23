"""Tables for the text ladder: per question, raw group (4/4 right) and contrastive group, +- bootstrap SE."""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
by = defaultdict(dict)       # (fam, cond) -> group -> {variant: row}
for r in rows:
    by[(r["family"], r["cond"])].setdefault(r["group"], {})[r["variant"]] = r


def stats(groups):
    q, raw, con, dh = [], [], [], []
    for g, vs in groups.items():
        if len(vs) < 4:
            continue
        lg = np.array([vs[v]["lg"] for v in range(4)])
        ans = [vs[v]["answer"] for v in range(4)]
        q += [vs[v]["correct"] for v in range(4)]
        raw.append(all(vs[v]["correct"] for v in range(4)))
        c = lg - lg.mean(0, keepdims=True)
        con.append(all(int(c[v].argmax()) == ans[v] for v in range(4)))
        if "donor_hit" in vs[0]:
            dh += [vs[v]["donor_hit"] for v in range(4)]
    n = len(raw)
    if n == 0:
        return None
    rng = np.random.default_rng(0)

    def se(x):
        x = np.array(x, float)
        return 100 * np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(500)])
    return dict(n=n, q=100 * np.mean(q), raw=100 * np.mean(raw), con=100 * np.mean(con),
                raw_se=se(raw), con_se=se(con), donor=100 * np.mean(dh) if dh else None)


ORD = ["count_end", "count_end2", "count_left", "count_start", "presence", "place", "order", "main_object",
       "main_object2", "colour", "number", "object_name"]
fams = sorted(set(k[0] for k in by), key=lambda f: ORD.index(f) if f in ORD else 99)
conds = sorted(set(k[1] for k in by), key=lambda c: (["oracle", "sel_best", "ep", "full", "fifo", "evfifo", "sel_rand",
                                                       "key", "mean", "lex", "summ", "other", "empty", "blind"].index(c.split("@")[0]), c))
print("cond | " + " | ".join(fams))
for c in conds:
    cells = []
    for f in fams:
        s = stats(by.get((f, c), {}))
        if s is None:
            cells.append("")
        else:
            extra = f" donor {s['donor']:.0f}" if s["donor"] is not None else ""
            cells.append(f"{s['con']:.0f}±{s['con_se']:.0f} / {s['raw']:.0f} ({s['q']:.0f}) n{s['n']}{extra}")
    print(f"{c} | " + " | ".join(cells))
print("\ncell = contrastive group % ± SE / raw group % (per question %) n groups")
