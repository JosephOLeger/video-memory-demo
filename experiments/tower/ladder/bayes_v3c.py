"""Colour-statistic Bayes bounds for count and colour, v3b (before) vs v3c (after).

Same segment as bayes_mass.py (the 13 s mean@120 segment holding the 6 s tableau + 7 s of the
group's filler, frames averaged). Three feature sets, all blind to shape and layout:
  mass   fraction of pixels near 14 palette colours (bayes_mass.py, TOL 24)
  hist   coarse 3D colour histogram, 8x8x8 bins (512 features)  -- stronger, palette-free
  vit    mean over the 256 tokens of the Qwen2.5-VL vision tower output (post-merger, 2048-d)
         for ONE second of the tableau (frames 0-1), no filler: a stronger probe than the
         segment mean (no filler noise); what a fully pooled visual memory could carry [--vit]
Decoder: multinomial logistic regression (L2), fit on TRAIN groups, tested on EVAL groups;
label = option text; prediction = argmax over the 4 options of the group. Per question and
per GROUP (4 variants right).

    python -I bayes_v3c.py --gen v3c --train 400 --eval 200 --out runs3c/bayes_v3c.json [--vit]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "v3c"))      # frozen v3 copy + gen_tableau_v3c.py
import gen_tableau_v3c as C  # noqa: E402

T = C.T
STYLE = T.TStyle(count_gap=True, door_bright=True)
P, B, D = 160, 12, 120
EVERY = P // B
PAL = {**{c: T.COLORS[c] for c in T.CNAMES}, "object": T.OBJ_COL, "black": (0, 0, 0),
       "skin": (235, 190, 150), "daylight": (150, 210, 255), "doorgrass": (90, 190, 80),
       "door": (125, 72, 30)}
CENT = np.array(list(PAL.values()), np.int16)
TOL = 24
GEN = None
VIT_SEG = False   # --vit-seg: mean over the whole 13 s segment (6 s tableau + 7 s filler)


def group(fam, g, split):
    if GEN == "v3c":
        return C.make_group(fam, g, STYLE, split=split)
    return T.make_group(fam, g, STYLE, split=split)


def seg_frames(it):
    t_end = P - D
    k = (P - t_end) // EVERY
    s0, s1 = P - EVERY * (k + 1), P - EVERY * k
    a, b = t_end - T.E, t_end
    fill = T.render_filler(it["filler_seed"], 2 * (s1 - s0), 2 * s0)
    fill[2 * (a - s0):2 * (b - s0)] = C.render_tableau(it["tableau"], T.E * T.FPS)
    return fill


def feats_px(frames):
    f = frames.reshape(frames.shape[0], -1, 3).astype(np.int16)
    mass = np.array([(np.abs(f - c).max(-1) <= TOL).mean() for c in CENT])
    q = (f // 32).astype(np.int32)
    idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=512) / idx.size
    return mass, hist


def feats(args):
    fam, g, split = args
    out = []
    for it in group(fam, g, split):
        fr = seg_frames(it)
        m, h = feats_px(fr)
        out.append({"group": g, "variant": it["variant"], "answer": it["answer"],
                    "options": it["options"], "mass": m.tolist(), "hist": h.tolist(),
                    "hist_tab": feats_px(fr[2 * 4:2 * 4 + 2])[1].tolist()})
    return out


def softmax_fit(X, y, n_cls, l2=1e-3, it=3000):
    import torch
    Xt = torch.tensor(X, dtype=torch.float64)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xt = (Xt - mu) / sd
    W = torch.zeros(Xt.shape[1], n_cls, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_cls, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=it)
    yt = torch.tensor(y)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda Z: ((torch.tensor(np.asarray(Z), dtype=torch.float64) - mu) / sd @ W + b).detach().numpy()


def evaluate(tr, ev, key, l2):
    lab_of = lambda r: r["options"][r["answer"]]  # noqa: E731
    labels = sorted({lab_of(r) for r in tr + ev})
    lab = {v: i for i, v in enumerate(labels)}
    f = softmax_fit(np.array([r[key] for r in tr]), [lab[lab_of(r)] for r in tr], len(labels), l2=l2)

    def sc(rows):
        S = f(np.array([r[key] for r in rows]))
        ok, by = [], {}
        for r, s in zip(rows, S):
            c = int(np.argmax([s[lab[o]] if o in lab else -1e9 for o in r["options"]]))
            ok.append(c == r["answer"])
            by.setdefault(r["group"], []).append(c == r["answer"])
        return [round(100 * np.mean(ok), 1), round(100 * np.mean([all(v) for v in by.values()]), 1)]
    return {"eval": sc(ev), "train": sc(tr)}


def vit_feats(rows_by_split, fam):
    """Mean vision-tower token of the segment, per item (GPU)."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    global _VIS
    if "_VIS" not in globals():
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                                                               dtype=torch.bfloat16)
        _VIS = (m.model.visual.to("cuda").eval(), AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct"))
        del m
    vis, proc = _VIS
    for split, rows in rows_by_split.items():
        cache = {}
        for r in rows:
            g = r["group"]
            if g not in cache:
                cache[g] = group(fam, g, split)
            it = cache[g][r["variant"]]
            fr = seg_frames(it) if VIT_SEG else C.render_tableau(it["tableau"], 2)
            vp = proc.video_processor(videos=[fr], return_tensors="pt", do_sample_frames=False)
            with torch.no_grad():
                emb = vis(vp["pixel_values_videos"].to("cuda", torch.bfloat16),
                          grid_thw=vp["video_grid_thw"].to("cuda")).pooler_output
            r["vit"] = emb.float().mean(0).cpu().numpy().tolist()


def main():
    global GEN, VIT_SEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="v3c", choices=["v3b", "v3c"])
    ap.add_argument("--families", default="count,colour")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--eval-start", type=int, default=10_000, dest="eval_start")
    ap.add_argument("--vit", action="store_true")
    ap.add_argument("--vit-seg", action="store_true", dest="vit_seg")
    ap.add_argument("--vit-train", type=int, default=200, dest="vit_train")
    ap.add_argument("--vit-eval", type=int, default=100, dest="vit_eval")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    GEN = a.gen
    VIT_SEG = a.vit_seg
    res = {}
    with Pool(a.procs) as pool:
        for fam in a.families.split(","):
            tr = sum(pool.map(feats, [(fam, g, "train") for g in range(a.train)]), [])
            ev = sum(pool.map(feats, [(fam, g, "eval") for g in range(a.eval_start, a.eval_start + a.eval)]), [])
            # within-group spread of the 512-bin histogram (0 = pixel-identical colour mass)
            spread = []
            for g in {r["group"] for r in ev}:
                H = np.array([r["hist"] for r in ev if r["group"] == g])
                spread.append(float(np.abs(H - H.mean(0)).sum(1).max()))
            out = {"mass": evaluate(tr, ev, "mass", 1e-3), "hist": evaluate(tr, ev, "hist", 1e-3),
                   "hist_tableau_only": evaluate(tr, ev, "hist_tab", 1e-3),
                   "max_within_group_L1_hist": max(spread), "n_train_groups": a.train,
                   "n_eval_groups": a.eval}
            if a.vit:
                trv = [r for r in tr if r["group"] < a.vit_train]
                evv = [r for r in ev if r["group"] < a.eval_start + a.vit_eval]
                vit_feats({"train": trv, "eval": evv}, fam)
                out["vit_mean"] = evaluate(trv, evv, "vit", 1e-2)
                out["vit_n"] = [a.vit_train, a.vit_eval]
            res[fam] = out
            print(a.gen, fam, json.dumps(out), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"gen": a.gen, "results": res}, open(a.out, "w"), indent=1)
    print("BAYES-FINI")


if __name__ == "__main__":
    main()
