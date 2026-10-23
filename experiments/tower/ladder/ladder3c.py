"""[v3c copy of ladder3.py: generator ladder/v3c/gen_tableau_v3c.py] Untrained memory ladder on generator v3 ("snapshot tableaux"), EVAL split only.

Stream (1 group = 1 s = 2 frames, 448 px): P s of the v3 procedural outdoor filler (seeded per
GROUP: pixel-identical across the 4 variants), the 6 s target tableau ending D s before the
window, then a W s filler window. Memories at a matched budget of B groups x 256 tokens
(B = 12 -> 3 072 tokens = 12 MiB):

    oracle   the target tableau (6 s)                       6 MiB
    *_shuf   same memory with the 256 token positions permuted (colour mass kept, shapes gone)
    tail     the last 1 s of the tableau (static tableau)    1 MiB
    other    tableau of ANOTHER variant of the same group      6 MiB
    fifo     last B s evicted, 448 px                          12 MiB
    fifo224  last 4B s evicted, 224 px (same tokens)           12 MiB
    key      1 s every P/B s over the whole past, random phase per (group, D)   12 MiB
    mean     past cut in B segments of P/B s, tokens averaged per position     12 MiB
    empty    window only;  blind  nothing

Load study (--load): the target room has TURQUOISE walls (never a shirt colour, never used
elsewhere) and the question names it. L distractor tableaux (the other variants of the same
group, ordinary walls) sit BETWEEN the target and the window, evenly spaced.
    oracle_all  all L+1 tableaux back to back (ceiling under load)
    oracle      the target tableau only
    key, mean   as above, at D

    CUDA_VISIBLE_DEVICES=1 python ladder3.py --families count --groups 2 --out runs3/smoke.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

V3 = os.path.join(HERE, "v3c")           # frozen copy: v3 generator + gen_tableau_v3c.py
sys.path.insert(0, V3)
import gen_tableau_v3c as T  # noqa: E402

GEN_MD5 = hashlib.md5(open(T.__file__, "rb").read() + open(T.T.__file__, "rb").read() + open(T.G.__file__, "rb").read()).hexdigest()[:8]
STYLE = T.TStyle(count_gap=True, door_bright=True)     # v3b, certified
GIST = ["count", "colour", "big_object", "room_state", "side"]
CUE_WALL = (64, 200, 200)
CUE_INTRO = ("The video stream shows outdoor scenery and, at several points, rooms with cartoon "
             "people, each wearing a coloured shirt. Only one room has turquoise walls. The "
             "question is about the room with the turquoise walls. ")


def tab_frames(it, wall=None):
    tb = it["tableau"]
    if wall is not None:
        tb = dataclasses.replace(tb, scene=dataclasses.replace(tb.scene, wall=wall))
    return T.render_tableau(tb, T.E * T.FPS)


def encode_pair(host, frames):
    return {"hi": host.encode(frames), "lo": host.encode(C.lowres(frames))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="count,colour,big_object,sign_number,hat_color")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gstart", type=int, default=10_000)
    ap.add_argument("--past", type=int, default=160)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--budget", type=int, default=12, help="B groups at 448 px per memory")
    ap.add_argument("--fifo-dists", default="0,120", dest="fifo_dists")
    ap.add_argument("--km-dist", type=int, default=120, dest="km_dist", help="D for key and mean")
    ap.add_argument("--load", default=None, help="e.g. 0,2,5 -> load study instead of main grid")
    ap.add_argument("--load-dist", type=int, default=120, dest="load_dist")
    ap.add_argument("--only", default=None)
    ap.add_argument("--res", default="hi", choices=["hi", "lo"],
                    help="resolution of oracle/other/tail/window (ceiling check at 224 px)")
    ap.add_argument("--early", type=int, default=40,
                    help="groups (summed over shards) after which a saturated cell stops")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    P, W, B = a.past, a.window, a.budget
    every = P // B                     # key spacing = segment length (13 s for P=160, B=12)
    fams = a.families.split(",")
    loads = [int(x) for x in a.load.split(",")] if a.load else None

    host = C.Host()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["family"], r["group"], r["variant"], r["cond"]))
    fo = open(a.out, "a")
    t0, nf = time.time(), 0
    # early stopping: per (family, cond), group-level outcomes seen by THIS shard
    grp_hist = defaultdict(list)
    early_n = max(1, -(-a.early // a.nshards))
    stopped = set()

    def emit(it, cond, lg, ntok, mem, extra=None):
        nonlocal nf
        pred = int(lg.argmax())
        r = {"family": it["family"], "kind": "gist" if it["family"] in GIST else "detail",
             "group": it["group"], "variant": it["variant"], "answer": it["answer"],
             "n_opt": len(it["options"]), "cond": cond, "pred": pred,
             "correct": int(pred == it["answer"]), "lg": [round(x, 3) for x in lg.tolist()],
             "ntok": ntok, "mem_tok": 0 if mem is None else mem.shape[0] * mem.shape[1],
             "mem_mib": 0.0 if mem is None else round(mem.shape[0] * mem.shape[1] * 4096 / 2 ** 20, 2),
             "gen": GEN_MD5}
        if extra:
            r.update(extra)
        fo.write(json.dumps(r) + "\n")
        nf += 1
        return r["correct"]

    gi = 0
    for g in range(a.gstart, a.gstart + a.groups):
        for fam in fams:
            gi += 1
            if gi % a.nshards != a.shard:
                continue
            items = T.make_group(fam, g, STYLE, split="eval")
            # filler shared by the group: v3 procedural scenery, P + W seconds
            fill = T.render_filler(items[0]["filler_seed"], 2 * (P + W), 0)
            fz = encode_pair(host, fill)
            tabs = [encode_pair(host, tab_frames(it)) for it in items]
            res = a.res
            win = fz[res][P:P + W]
            ok_grp = defaultdict(list)

            def need(it, c):
                if only is not None and c.split("|")[0] not in only and c not in only:
                    return False
                if (fam, c) in stopped:
                    return False
                return (fam, it["group"], it["variant"], c) not in done

            def stream_with(placements):
                """placements: [(enc, end_group)] -> (hi, lo) streams with tableaux written in."""
                hi, lo = fz["hi"][:P + W].clone(), fz["lo"][:P + W].clone()
                for enc, end in placements:
                    n = enc["hi"].shape[0]
                    hi[end - n:end] = enc["hi"]
                    lo[end - n:end] = enc["lo"]
                return hi, lo

            def km(hi, D, tag):
                ph = random.Random(f"phase-{fam}-{g}-{D}-{tag}").randrange(every)
                idx = sorted(i for i in range(P - 1 - ph, -1, -every))[-B:]
                segs = [hi[P - every * (k + 1):P - every * k].float().mean(0) for k in range(B)][::-1]
                return hi[idx], torch.stack(segs).to(hi.dtype), ph

            if loads is None:
                # ---------------- main grid
                grp_eb = None
                for v, it in enumerate(items):
                    tb = tabs[v]
                    p = it["prompt"]
                    n_opt = len(it["options"])
                    if need(it, "oracle"):
                        lg, nt = host.score(tb[res], win, p, n_opt)
                        ok_grp["oracle"].append(emit(it, "oracle", lg, nt, tb[res]))
                    if need(it, "tail"):
                        lg, nt = host.score(tb[res][-1:], win, p, n_opt)
                        ok_grp["tail"].append(emit(it, "tail", lg, nt, tb[res][-1:]))
                    if need(it, "other"):
                        dv = (v + 1) % len(items)
                        assert items[dv]["answer"] != it["answer"]
                        lg, nt = host.score(tabs[dv][res], win, p, n_opt)
                        ok_grp["other"].append(emit(it, "other", lg, nt, tabs[dv][res], {
                            "donor_answer": items[dv]["answer"],
                            "donor_hit": int(int(lg.argmax()) == items[dv]["answer"])}))
                    if need(it, "empty"):
                        if grp_eb is None:
                            grp_eb = (host.score(None, win, p, n_opt), host.score(None, None, p, n_opt))
                        (el, en), (bl, bn) = grp_eb
                        ok_grp["empty"].append(emit(it, "empty", el, en, None))
                        ok_grp["blind"].append(emit(it, "blind", bl, bn, None))
                    for D in [int(x) for x in a.fifo_dists.split(",")]:
                        hi, lo = stream_with([(tb, P - D)])
                        for m, mem in (("fifo", hi[P - B:P]), ("fifo224", lo[P - 4 * B:P])):
                            c = f"{m}@{D}"
                            if need(it, c):
                                lg, nt = host.score(mem, hi[P:P + W] if res == "hi" else lo[P:P + W], p, n_opt)
                                buf = B if m == "fifo" else 4 * B
                                ok_grp[c].append(emit(it, c, lg, nt, mem, {
                                    "D": D, "mem": m, "in_buf": int(D + T.E <= buf)}))
                    D = a.km_dist
                    if need(it, "oracle_shuf"):
                        perm = torch.randperm(tb[res].shape[1], generator=torch.Generator().manual_seed(g))
                        lg, nt = host.score(tb[res][:, perm], win, p, n_opt)
                        emit(it, "oracle_shuf", lg, nt, tb[res])
                    if need(it, f"key@{D}") or need(it, f"mean@{D}") or need(it, f"mean_shuf@{D}"):
                        hi, lo = stream_with([(tb, P - D)])
                        kmem, mmem, ph = km(hi, D, "main")
                        # spatial token shuffle: keeps colour mass, destroys shapes (question 3)
                        perm = torch.randperm(mmem.shape[1], generator=torch.Generator().manual_seed(g))
                        for m, mem in (("key", kmem), ("mean", mmem), ("mean_shuf", mmem[:, perm])):
                            c = f"{m}@{D}"
                            if need(it, c):
                                lg, nt = host.score(mem, hi[P:P + W], p, n_opt)
                                ok_grp[c].append(emit(it, c, lg, nt, mem, {"D": D, "mem": m, "phase": ph}))
                    fo.flush()
            else:
                # ---------------- load study (turquoise target, L distractors in between)
                D = a.load_dist
                ctabs = [encode_pair(host, tab_frames(it, CUE_WALL)) for it in items]
                for v, it in enumerate(items):
                    p = CUE_INTRO + it["prompt"][len(T.INTRO):]
                    n_opt = len(it["options"])
                    for L in loads:
                        dist = [tabs[(v + 1 + (j % 3)) % 4] for j in range(L)]
                        gap = D // (L + 1)
                        pl = [(ctabs[v], P - D)] + [(dist[j], P - D + (j + 1) * gap) for j in range(L)]
                        hi, lo = stream_with(pl)
                        allm = torch.cat([ctabs[v]["hi"]] + [d["hi"] for d in dist])
                        conds = [("oracle_all", allm), ("oracle", ctabs[v]["hi"])]
                        # alternative A: the L+1 tableaux separated by 2 s of the group's filler
                        gapm = [ctabs[v]["hi"]]
                        for d in dist:
                            gapm += [fz["hi"][:2], d["hi"]]
                        conds.append(("oracle_gap", torch.cat(gapm)))
                        # alternative B: one labelled video per tableau, chronological
                        order = [ctabs[v]["hi"]] + [d["hi"] for d in dist]
                        c = f"oracle_sep|L{L}"
                        if need(it, c):
                            segs = [(f"Room clip {k + 1} of {L + 1}:", x) for k, x in enumerate(order)]
                            segs.append(("The most recent seconds of the video stream:", hi[P:P + W]))
                            lg, nt = C.score_multi(host, segs, p, n_opt)
                            ok_grp[c].append(emit(it, c, lg, nt, torch.cat(order), {"D": D, "L": L, "mem": "oracle_sep"}))
                        kmem, mmem, ph = km(hi, D, f"L{L}")
                        conds += [("key", kmem), ("mean", mmem)]
                        for m, mem in conds:
                            c = f"{m}|L{L}"
                            if need(it, c):
                                lg, nt = host.score(mem, hi[P:P + W], p, n_opt)
                                ok_grp[c].append(emit(it, c, lg, nt, mem, {"D": D, "L": L, "mem": m}))
                        c = f"empty|L{L}"
                        if need(it, c):
                            lg, nt = host.score(None, hi[P:P + W], p, n_opt)
                            ok_grp[c].append(emit(it, c, lg, nt, None, {"L": L}))
                    fo.flush()
            # early stopping bookkeeping (group-level, this shard)
            for c, v in ok_grp.items():
                if len(v) == len(items):
                    grp_hist[(fam, c)].append(int(all(v)))
            for (f, c), h in grp_hist.items():
                if f == fam and (f, c) not in stopped and len(h) >= early_n and c.split("@")[0].split("|")[0] in (
                        "oracle", "oracle_all", "tail", "other", "empty", "blind", "fifo", "fifo224"):
                    m = 100 * sum(h) / len(h)
                    if m >= 95 or m == 0:
                        stopped.add((f, c))
                        print(f"  early-stop {f} {c}: group acc {m:.0f} over {len(h)} groups", flush=True)
            print(f"{fam} g{g} done [{nf} rows, {(time.time() - t0) / 60:.1f} min]", flush=True)
    print("LADDER3-FINI", flush=True)


if __name__ == "__main__":
    main()
