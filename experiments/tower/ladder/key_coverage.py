"""Keyframe accuracy conditioned on whether a 1-s key landed on the evidence (run locally: needs PIL)."""
import json, random, sys, importlib.util
from collections import defaultdict
spec = importlib.util.spec_from_file_location("g", "gen_scenes_snapshot.py"); G = importlib.util.module_from_spec(spec)
sys.modules["g"] = G; spec.loader.exec_module(G)
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/val.jsonl")]
P, eps, acc = 176, {}, defaultdict(list)
for r in rows:
    if not r["cond"].startswith("key@") or "L" in r["cond"]:
        continue
    D, fam, g = r["D"], r["family"], r["group"]
    if (fam, g) not in eps:
        eps[(fam, g)] = G.make_group(fam, g)
    e = eps[(fam, g)][r["variant"]]["episode"]; start = P - D - r["E"]
    phase = random.Random(f"phase-{fam}-{g}-{D}").randrange(8)
    keys = {P - 1 - phase - 8 * k for k in range(30)}
    if fam == "sign_number":
        f0, f1 = [(a, b) for (ev, a, b) in e.timeline if ev[0] == "sign"][0]
    else:
        f0, f1 = 0, (e.timeline[0][1] if e.timeline else e.n_frames)
    acc[(fam, bool(keys & {start + f // 2 for f in range(f0, f1)}))].append(r["correct"])
for k, v in sorted(acc.items()):
    print(k, len(v), f"{100 * sum(v) / len(v):.0f}%")
