"""Contrastive (group-debiased) accuracy: letter logits minus their mean over the group's variants."""
import collections
import json
import sys

import torch

for p in sys.argv[1:]:
    rows = json.load(open(p))["rows"]
    print("==", p)
    for fam in dict.fromkeys(r["family"] for r in rows):
        s = [r for r in rows if r["family"] == fam]
        out = []
        for c in [k for k in s[0] if k + "_lg" in s[0]]:
            g = collections.defaultdict(list)
            for r in s:
                g[r["group"]].append(r)
            okg = oki = 0
            for v in g.values():
                m = torch.tensor([r[c + "_lg"] for r in v]).mean(0)
                res = [int((torch.tensor(r[c + "_lg"]) - m - 1e-6 * torch.arange(len(m))).argmax()) == r["answer"] for r in v]
                okg += all(res)
                oki += sum(res)
            out.append(f"{c} {100 * okg / len(g):.0f} ({100 * oki / len(s):.0f})")
        print(f"   {fam}: " + " | ".join(out))
