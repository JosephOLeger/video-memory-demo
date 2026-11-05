import re, sys
for f in sys.argv[1:]:
    rows = [l for l in open(f) if l.startswith("  step")]
    def g(l, k):
        m = re.search(rf" {k} ([-0-9.e]+)", l); return float(m.group(1)) if m else None
    out = []
    for i in range(0, len(rows), 10):
        b = rows[i:i + 10]
        gs = [g(l, "gap") for l in b if g(l, "gap") is not None]
        mg = [g(l, "mem_gain") for l in b if g(l, "mem_gain") is not None]
        st = g(b[-1], "step")
        if gs: out.append(f"{int(st)}:{sum(gs)/len(gs):+.3f}")
        elif mg: out.append(f"{int(st)}:{sum(mg)/len(mg):.2f}")
    print(f.split("/")[-1], " ".join(out))
