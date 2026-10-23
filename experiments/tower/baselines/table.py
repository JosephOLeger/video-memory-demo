import json, glob, os
B = os.path.expanduser("~/work/iclr/baselines"); S = os.path.expanduser("~/work/iclr/stream")
runs = [(f"deep_s{s}", f"{S}/C1_color_s{s}.json") for s in range(3)]
runs += [(os.path.basename(f)[:-5], f) for f in sorted(glob.glob(f"{B}/*_s[0-9].json"))]
runs += [("mixfps:" + os.path.basename(f)[:-5], f) for f in sorted(glob.glob(f"{B}/mixfps/*_s[0-9].json"))]
p = lambda x: "-" if x is None else f"{100*x:.0f}"
print("| run | pas | memoire | autre video | fenetre | enseignant | dist 2-4 (mem/autre) | dist 4-10 (mem/autre) | premier pas >=90 | duree |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n, f in runs:
    d = json.load(open(f)); h = d["history"]; e = h[-1]; a = e["all"]
    first = next((x["step"] for x in h if x["all"]["mem"] >= 0.9), None)
    d24, d410 = e["shape_d2-4"], e["shape_d4-10"]
    print(f"| {n} | {e['step']} | {p(a['mem'])} | {p(a['wrong'])} | {p(a['window'])} | {p(a['teacher'])} | "
          f"{p(d24['mem'])}/{p(d24['wrong'])} | {p(d410['mem'])}/{p(d410['wrong'])} | {first or '-'} | "
          f"{d.get('elapsed_s', 0) / 3600:.1f} h |")
print()
for n, f in runs:
    h = json.load(open(f))["history"]
    print(n, " ".join(f"{x['step']}:{p(x['all']['mem'])}/{p(x['all']['wrong'])}" for x in h))
