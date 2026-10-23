"""Question 3: how much of each family can a COLOUR-MASS statistic decode, with no shapes?

The main-grid `mean@120` memory averages 13 s segments; the 6 s target tableau sits in one
segment with 7 s of the group's filler (P=160, every=13, D=120 -> tableau groups [34,40) inside
segment [30,43)). Feature of an item = that segment's average per-frame colour histogram over a
fixed palette (8 shirt colours, object colour, black, skin, door/daylight colours), i.e. the
colour MASS a linear average can carry. Shapes, positions and layout are thrown away.

Decoders, fitted on the TRAIN split (seed salt, other decor), tested on EVAL groups:
  * multinomial logistic regression on the histogram -> score of each option (argmax)
  * for count also a 1-parameter rule: n = shirt mass / unit (unit fitted on train)
Reported per item and per GROUP (all 4 variants right), like the ladder.

    python -I bayes_mass.py --train 400 --eval 200 --out runs3/bayes_mass.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "v3"))
import gen_tableau as T  # noqa: E402

STYLE = T.TStyle(count_gap=True, door_bright=True)
P, B, D = 160, 12, 120
EVERY = P // B
PAL = {**{c: T.COLORS[c] for c in T.CNAMES}, "object": T.OBJ_COL, "black": (0, 0, 0),
       "skin": (235, 190, 150), "daylight": (150, 210, 255), "doorgrass": (90, 190, 80),
       "door": (125, 72, 30)}
NAMES = list(PAL)
CENT = np.array([PAL[k] for k in NAMES], np.int16)
TOL = 24


def hist(frames):
    """Mean over frames of the fraction of pixels within TOL (max-abs) of each palette colour."""
    f = frames.reshape(frames.shape[0], -1, 3).astype(np.int16)
    out = np.zeros(len(NAMES))
    for i, c in enumerate(CENT):
        out[i] = (np.abs(f - c).max(-1) <= TOL).mean()
    return out


def seg_frames(it):
    """Frames of the segment holding the tableau, as in ladder3 main grid at D=120."""
    t_end = P - D                     # tableau groups [t_end-6, t_end)
    k = (P - t_end) // EVERY          # segment index from the end containing t_end-1
    s0, s1 = P - EVERY * (k + 1), P - EVERY * k
    a, b = t_end - T.E, t_end
    assert s0 <= a and b <= s1, (s0, s1, a, b)
    fill = T.render_filler(it["filler_seed"], 2 * (s1 - s0), 2 * s0)
    tab = T.render_tableau(it["tableau"], T.E * T.FPS)
    fill[2 * (a - s0):2 * (b - s0)] = tab
    return fill


def feats(args):
    fam, g, split = args
    out = []
    for it in T.make_group(fam, g, STYLE, split=split):
        out.append({"family": fam, "group": g, "variant": it["variant"], "answer": it["answer"],
                    "options": it["options"], "x": hist(seg_frames(it)).tolist()})
    return out


def softmax_fit(X, y, n_cls, l2=1e-3, it=3000, lr=0.5):
    import torch
    Xt = torch.tensor(X, dtype=torch.float64)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xt = (Xt - mu) / sd
    W = torch.zeros(Xt.shape[1], n_cls, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_cls, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=lr, max_iter=it)
    yt = torch.tensor(y)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda Z: ((torch.tensor(Z, dtype=torch.float64) - mu) / sd @ W + b).detach().numpy()


def label_space(fam, r):
    """Class label = the option's TEXT value, so one classifier serves every group."""
    return r["options"][r["answer"]]


def score(rows, pred_fn):
    by = {}
    ok = []
    for r in rows:
        c = int(pred_fn(r))
        ok.append(c == r["answer"])
        by.setdefault(r["group"], []).append(c == r["answer"])
    return 100 * np.mean(ok), 100 * np.mean([all(v) for v in by.values()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="count,colour,big_object,sign_number,hat_color")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--eval-start", type=int, default=10_000, dest="eval_start")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    with Pool(a.procs) as pool:
        for fam in a.families.split(","):
            tr = sum(pool.map(feats, [(fam, g, "train") for g in range(a.train)]), [])
            ev = sum(pool.map(feats, [(fam, g, "eval") for g in range(a.eval_start, a.eval_start + a.eval)]), [])
            labels = sorted({label_space(fam, r) for r in tr + ev})
            lab = {v: i for i, v in enumerate(labels)}
            f = softmax_fit(np.array([r["x"] for r in tr]), [lab[label_space(fam, r)] for r in tr], len(labels))

            def pred_lr(r):
                s = f(np.array([r["x"]]))[0]
                return int(np.argmax([s[lab[o]] if o in lab else -1e9 for o in r["options"]]))
            out = {"logreg": score(ev, pred_lr), "logreg_train": score(tr, pred_lr),
                   "n_eval_groups": a.eval, "n_train_groups": a.train}
            if fam == "count":
                shirt = [NAMES.index(c) for c in T.CNAMES]
                m = np.array([sum(r["x"][i] for i in shirt) for r in tr])
                n = np.array([int(label_space(fam, r)) for r in tr], float)
                unit = float((m * n).sum() / (n * n).sum())

                def pred_mass(r):
                    est = sum(r["x"][i] for i in shirt) / unit
                    return int(np.argmin([abs(int(o) - est) for o in r["options"]]))
                out["mass_rule"] = score(ev, pred_mass)
            res[fam] = out
            print(fam, json.dumps(out), flush=True)
    json.dump({"palette": NAMES, "tol": TOL, "results": res}, open(a.out, "w"), indent=1)
    print("BAYES-FINI")


if __name__ == "__main__":
    main()
