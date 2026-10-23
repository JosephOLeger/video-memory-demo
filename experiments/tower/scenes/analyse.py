"""Per-family breakdown of a ceiling json: accuracy by true value, predicted-letter histogram."""
import json
import sys
from collections import Counter, defaultdict

d = json.load(open(sys.argv[1]))
rows = d["rows"]
for fam in dict.fromkeys(r["family"] for r in rows):
    sel = [r for r in rows if r["family"] == fam]
    by = defaultdict(list)
    for r in sel:
        by[r["options"][r["answer"]]].append(r["full"] == r["answer"])
    pred = Counter(r["options"][r["full"]] for r in sel)
    print(f"== {fam}: acc by true value (full):",
          {k: f"{100*sum(v)/len(v):.0f}%/{len(v)}" for k, v in sorted(by.items())})
    print("   predicted values:", dict(pred.most_common()))
    if fam in ("init_count", "final_count", "n_left"):
        err = Counter(int(r["options"][r["full"]]) - int(r["options"][r["answer"]]) for r in sel)
        print("   error (pred-true):", dict(sorted(err.items())))
