"""Near-duplicate scenes among Perception Test valid videos (same room / same setup):
32x18 colour thumbnail of the frame at 1 s, cosine of mean-centred vectors."""
import os, json, sys, numpy as np, decord, random
D = "/home/josephleger/work/iclr/ptest/data/videos"
vids = sorted(f[:-4] for f in os.listdir(D) if f.endswith(".mp4"))
random.Random(0).shuffle(vids); vids = vids[:int(sys.argv[1])]
X, ok = [], []
for v in vids:
    try:
        vr = decord.VideoReader(f"{D}/{v}.mp4", width=32, height=18)
        fr = vr[min(len(vr) - 1, int(vr.get_avg_fps()))].asnumpy().astype(np.float32).ravel()
        fr -= fr.mean(); fr /= (np.linalg.norm(fr) + 1e-6); X.append(fr); ok.append(v)
    except Exception as e:
        pass
X = np.stack(X); S = X @ X.T; np.fill_diagonal(S, -1)
best = S.max(1)
qs = [json.loads(l) for l in open("/home/josephleger/work/iclr/ptest/questions_all.jsonl")]
print(f"videos {len(ok)}; nearest-neighbour cosine quantiles 50/90/99%:", np.quantile(best, [.5, .9, .99]).round(3))
for th in (0.9, 0.95, 0.98):
    print(f"  frac with a near-duplicate (cos>{th}): {np.mean(best > th):.3f}")
pairs = [(float(S[i, j]), ok[i], ok[j]) for i in range(len(ok)) for j in range(i + 1, len(ok)) if S[i, j] > 0.97]
pairs.sort(reverse=True)
print("top pairs:", pairs[:12])
json.dump({"ids": ok, "best": best.tolist(), "pairs": pairs[:200]}, open("ptest_dupes.json", "w"))
