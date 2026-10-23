"""Recompute each item's answer from the rendered episode facts; count mismatches."""
import sys, collections; sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "snap1")
import gen_scenes as G
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
bad = collections.Counter(); tot = collections.Counter(); ex = {}
for fam in G.FAMILIES:
    for g in range(N):
        for it in G.make_group(fam, g, G.Style(qset=2)):
            f, o, ep = it["facts"], it["options"], it["episode"]
            if fam == "init_count": t = o.index(str(f["n0"]))
            elif fam == "final_count": t = o.index(str(f["n_final"]))
            elif fam == "n_left": t = o.index(str(f["n_left"]))
            elif fam == "first_left": t = next(i for i, x in enumerate(o) if x.endswith(f["left"][0]))
            elif fam == "hat_color": t = o.index(f["hat"])
            elif fam == "sign_number": t = o.index(str(f["sign"]))
            elif fam == "door_state":
                s = lambda b: "open" if b else "closed"
                t = next(i for i, x in enumerate(o) if x.startswith(s(f["door0"]) + " at the start") and x.endswith(s(f["door1"]) + " at the end"))
            elif fam == "order":
                c = o and it["prompt"].split("person in ")[1].split(" left")[0]
                ev = ep.events
                di = next(i for i, (k, a) in enumerate(ev) if k == "door")
                ci = next(i for i, (k, a) in enumerate(ev) if k == "leave" and ep.agents[a].color == c)
                t = 0 if di < ci else 1
            tot[fam] += 1
            if t != it["answer"]:
                bad[fam] += 1; ex.setdefault(fam, (g, it["variant"], it["answer"], t))
for fam in G.FAMILIES:
    print(f"{fam:12s} n={tot[fam]} label!=facts: {bad[fam]} ({100*bad[fam]/tot[fam]:.1f}%) ex={ex.get(fam)}")
