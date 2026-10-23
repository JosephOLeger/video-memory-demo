import sys, collections, math; sys.path.insert(0, "snap1")
import gen_scenes as G
N = 1500
def chi(c):
    n = sum(c.values()); k = len(c); e = n / k
    return round(sum((v - e) ** 2 / e for v in c.values()), 1), k - 1
fl = collections.Counter(); hat = collections.Counter(); ansT = collections.defaultdict(list)
side = collections.Counter(); bg = collections.Counter(); slot_first = collections.Counter()
ntok = collections.defaultdict(list)
for g in range(N):
    for it in G.make_group("first_left", g):
        f = it["facts"]; fl[f["left"][0]] += 1
        ep = it["episode"]; a = next(a for a in ep.agents if a.color == f["left"][0])
        slot_first[a.slot] += 1
    for it in G.make_group("hat_color", g):
        hat[it["facts"]["hat"]] += 1
    for fam in ("init_count", "final_count", "n_left", "door_state"):
        for it in G.make_group(fam, g):
            ansT[fam].append((it["options"][it["answer"]], it["episode"].n_frames))
    sc = G.make_scene(__import__("random").Random(f"init_count-{g}"))
    bg[(sc.door_x,)] += 1
print("first leaver colour:", dict(fl), "chi2/df", chi(fl))
print("first leaver slot:", dict(sorted(slot_first.items())), "chi2/df", chi(slot_first))
print("hat colour:", dict(hat), "chi2/df", chi(hat))
for fam, l in ansT.items():
    by = collections.defaultdict(list)
    for a, T in l: by[a].append(T)
    print(fam, "mean T (frames) by answer:", {k: round(sum(v) / len(v), 1) for k, v in sorted(by.items())})
# power
z = 1.959964, 0.841621
print("\n== items for a 5-pt paired difference (McNemar, alpha .05 two-sided, power .8)")
for pd in (0.10, 0.15, 0.20, 0.30, 0.40):
    d = 0.05; n = (z[0] * math.sqrt(pd) + z[1] * math.sqrt(pd - d * d)) ** 2 / d ** 2
    print(f"  discordant rate {pd:.2f}: n = {math.ceil(n)}")
for p in (0.4, 0.6):
    pbar = p + 0.025; n = (z[0] * math.sqrt(2 * pbar * (1 - pbar)) + z[1] * math.sqrt(p * (1 - p) + (p + .05) * (1 - p - .05))) ** 2 / .05 ** 2
    print(f"  unpaired, p={p}: n per arm = {math.ceil(n)}")
print("== seeds per arm for a 5-pt difference given seed SD s (two-sample t approx)")
for s in (2, 3, 4, 5):
    print(f"  seed SD {s} pts: {math.ceil(2 * (z[0] + z[1]) ** 2 * s * s / 25)} seeds/arm (+1-2 for t)")
