"""Is Qwen right for the right reason? Same items under frame-set perturbations.
conds: full (ceiling.py inputs, mrope assumes 24 fps), full_fps (true 2 fps timing),
first4 (only the 4-s still start), half (every other frame), shuf (frames in random order),
noev (start + end holds only: the evidence-bearing middle removed)."""
import sys, json, random, collections
sys.path.insert(0, "snap1")
import numpy as np, torch
import gen_scenes as G
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
fams = sys.argv[1].split(","); NG = int(sys.argv[2]); G0 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
dev = "cuda"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", dtype=torch.bfloat16).to(dev).eval()
lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in G.LET]

@torch.no_grad()
def score(frames, prompt, n_opt, truefps):
    content = [{"type": "video"}, {"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if len(frames) % 2: frames = frames[1:]
    if truefps:
        inp = proc(text=[text], videos=[frames], return_tensors="pt", video_metadata=[{"fps": G.FPS, "total_num_frames": len(frames)}], do_sample_frames=False)
    else:
        inp = proc(text=[text], videos=[frames], fps=G.FPS, return_tensors="pt")
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    lg = model(**inp, use_cache=False).logits[0, -1].float()[lid[:n_opt]]
    return int(lg.argmax()), float(inp["second_per_grid_ts"][0])

rows = []
for fam in fams:
    for g in range(G0, G0 + NG):
        for it in G.make_group(fam, g):
            fr = G.render(it["episode"]); st = it["episode"].style
            r = {"family": fam, "group": g, "variant": it["variant"], "answer": it["answer"]}
            rng = random.Random(g * 10 + it["variant"])
            perm = list(range(len(fr))); rng.shuffle(perm)
            C = {"full": (fr, False), "full_fps": (fr, True), "first4": (fr[:st.hold0], True),
                 "half": (fr[::2], False), "shuf": (fr[perm], False),
                 "noev": (np.concatenate([fr[:st.hold0], fr[-st.hold1:]]), True)}
            for c, (x, tf) in C.items():
                r[c], r[c + "_spg"] = score(x, it["prompt"], len(it["options"]), tf)
            rows.append(r)
    sel = [r for r in rows if r["family"] == fam]
    line = " ".join(f"{c} {100*np.mean([r[c]==r['answer'] for r in sel]):.0f}" for c in C)
    grp = collections.defaultdict(list)
    for r in sel: grp[r["group"]].append(r["full"] == r["answer"])
    print(f"{fam:12s} n={len(sel)} {line}  grp(full) {100*np.mean([all(v) for v in grp.values()]):.0f}  spg full={sel[0]['full_spg']:.3f} fps={sel[0]['full_fps_spg']:.3f}", flush=True)
json.dump(rows, open(f"ceiling_probe_{'_'.join(fams)}.json", "w"))
print("FINI")
