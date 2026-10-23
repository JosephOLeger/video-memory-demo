import json, collections, sys
rows = json.load(open(sys.argv[1]))["rows"]
for fam in dict.fromkeys(r["family"] for r in rows):
    sel = [r for r in rows if r["family"] == fam]
    out = []
    for c in ("full", "blind", "last"):
        h = collections.Counter("ABCD"[r[c]] for r in sel)
        acc_by_letter = {L: round(100 * sum(r[c] == r["answer"] for r in sel if "ABCD"[r["answer"]] == L) / max(1, sum("ABCD"[r["answer"]] == L for r in sel))) for L in "ABCD"[:len(sel[0]["options"])]}
        out.append(f"{c}: pred {dict(sorted(h.items()))} acc|true {acc_by_letter}")
    print(fam, " || ".join(out))
