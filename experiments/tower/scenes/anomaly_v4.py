"""Anomaly check: can a TRIVIAL classifier tell episode frames from the rest of the stream?

For v3 (cartoon room in procedural outdoor filler) and v4 (overlay on real footage), build
balanced sets of 1-second units (2 frames): EPISODE units (the 6 s of the episode) against
STREAM units (the 6 s just before + 6 s from the post/window part of the same stream, so the
same video). Two feature sets, each with an L2 logistic regression (torch, standardized):
  colour : per-unit colour statistics (mean/std per channel, 8-bin histogram per channel,
           saturation mean/std, gradient energy) -> 56 dims
  qwen   : Qwen2.5-VL-3B vision tower, post-merger tokens mean-pooled per unit -> 2048 dims
Fit on TRAIN-split groups, test on EVAL-split groups (different videos for v4).
Expectation: v3 near 100 % (style anomaly); v4 lower, the only cue being the overlay.

    CUDA_VISIBLE_DEVICES=0 python anomaly_v4.py --groups-train 60 --groups-eval 30 --out results/anomaly_v4.json
"""
import argparse
import json

import numpy as np
import torch

import gen_composite_v4 as V
import gen_tableau as T


def colour_feats(fr):        # fr: [2, H, W, 3] uint8
    x = fr.astype(np.float32) / 255.0
    f = [x.mean((0, 1, 2)), x.std((0, 1, 2))]
    for c in range(3):
        f.append(np.histogram(x[..., c], bins=8, range=(0, 1))[0] / x[..., c].size)
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    f.append([sat.mean(), sat.std()])
    g = np.abs(np.diff(x.mean(-1), axis=1)).mean() + np.abs(np.diff(x.mean(-1), axis=2)).mean()
    f.append([g])
    return np.concatenate([np.ravel(a) for a in f])


def units(gen, split, groups, fams):
    """yield (episode_frames[12], stream_frames[12]) per item (variant 0 and 2 of each group)."""
    for fam in fams:
        for g in range(groups):
            if gen == "v3":
                items = T.make_group(fam, g, T.TStyle(count_gap=True, door_bright=True), split=split)
            else:
                items = V.make_group(fam, g, split)
            for v in (0, 2):
                it = items[v]
                if gen == "v3":
                    s = T.stream(it)
                    ep = s["tab"]
                    other = np.concatenate([s["pre"][-4:], s["post"][:4], s["win"][-4:]])
                else:
                    s = V.stream(it)
                    ep = s["ep"]
                    other = np.concatenate([s["pre"][-4:], s["post"][:4], s["win"][-4:]])
                yield ep, other


class QwenVis:
    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", dtype=torch.bfloat16).to("cuda").eval()
        self.proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    @torch.no_grad()
    def __call__(self, frames):  # [2k, H, W, 3] -> [k, 2048]
        vp = self.proc.video_processor(videos=[frames], return_tensors="pt", do_sample_frames=False)
        pv = vp["pixel_values_videos"].to("cuda", torch.bfloat16)
        thw = vp["video_grid_thw"].to("cuda")
        emb = self.model.model.visual(pv, grid_thw=thw).pooler_output
        t, h, w = thw[0].tolist()
        return emb.reshape(t, h * w // 4, -1).float().mean(1).cpu().numpy()


def build(gen, split, groups, fams, qv):
    Xc, Xq, y = [], [], []
    for ep, other in units(gen, split, groups, fams):
        for lab, fr in ((1, ep), (0, other)):
            for k in range(0, len(fr), 2):
                Xc.append(colour_feats(fr[k:k + 2]))
                y.append(lab)
            if qv is not None:
                Xq.extend(list(qv(fr)))
    return np.array(Xc), (np.array(Xq) if qv is not None else None), np.array(y)


def logreg(Xtr, ytr, Xte, yte, l2=1e-3, steps=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        acc_tr = (((A @ w + b) > 0).float() == yt).float().mean().item()
        acc_te = (((B @ w + b) > 0).numpy() == yte).mean()
    return round(100 * acc_tr, 1), round(100 * float(acc_te), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-train", type=int, default=60, dest="gtr")
    ap.add_argument("--groups-eval", type=int, default=30, dest="gte")
    ap.add_argument("--no-qwen", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qv = None if a.no_qwen else QwenVis()
    fams = V.FAMILIES
    res = {}
    for gen in ("v3", "v4"):
        Ctr, Qtr, ytr = build(gen, "train", a.gtr, fams, qv)
        Cte, Qte, yte = build(gen, "eval", a.gte, fams, qv)
        r = {"n_train": len(ytr), "n_eval": len(yte), "colour (train, eval acc %)": logreg(Ctr, ytr, Cte, yte)}
        if qv is not None:
            r["qwen (train, eval acc %)"] = logreg(Qtr, ytr, Qte, yte, l2=1e-2)
        res[gen] = r
        print(gen, r, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("ANOMALY-FINI")


if __name__ == "__main__":
    main()
