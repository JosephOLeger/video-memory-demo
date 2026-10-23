"""Comparaisons appariees LVB (binomiale exacte sur les discordants), par tranche de duree."""
import json, os, sys
from math import comb
L = os.path.expanduser("~/work/iclr/lvb")
def load(n):
    p = f"{L}/{n}.json"
    return {r["id"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else None
def p2(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def comp(name, A, B, ka, kb):
    ids = sorted(set(A) & set(B))
    print(f"\n{name}  (n={len(ids)})")
    groups = [None] + sorted({A[i]["duration_group"] for i in ids})
    for g in groups:
        sel = [i for i in ids if g is None or A[i]["duration_group"] == g]
        a = sum(A[i][ka] == A[i]["gold"] for i in sel); b = sum(B[i][kb] == B[i]["gold"] for i in sel)
        w = sum(A[i][ka] == A[i]["gold"] and B[i][kb] != B[i]["gold"] for i in sel)
        l = sum(A[i][ka] != A[i]["gold"] and B[i][kb] == B[i]["gold"] for i in sel)
        print(f"  {str(g or 'tout'):>6} n={len(sel):4d}  {100*a/len(sel):5.1f} vs {100*b/len(sel):5.1f}  "
              f"delta {100*(a-b)/len(sel):+5.1f}  ({w} gagnes / {l} perdus, p={p2(w, l):.3f})")
mem, mis, full, last = load("mem_s0_4000"), load("mis_s0_4000"), load("full256"), load("last16_s0_4000")
if mem:
    comp("bonne memoire vs Qwen 32 images", mem, mem, "mem", "base")
if mis:
    comp("autre video vs Qwen 32 images", mis, mis, "mem", "base")
if mem and mis:
    same = sum(mem[i]["base"] == mis[i]["base"] for i in set(mem) & set(mis))
    print(f"\n(reference identique dans les deux runs : {same}/{len(set(mem) & set(mis))})")
    comp("bonne memoire vs autre video", mem, mis, "mem", "mem")
if full and mem:
    comp("Qwen 256 images vs Qwen 32 images", full, mem, "base", "base")
if last:
    comp("16 dernieres + memoire vs 16 dernieres seules", last, last, "mem", "base")
    comp("16 dernieres + autre video vs 16 dernieres seules", last, last, "mis", "base")
    comp("16 dernieres : bonne memoire vs autre video", last, last, "mem", "mis")
