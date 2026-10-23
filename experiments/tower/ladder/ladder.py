"""Untrained memory ladder on streamed episodes.

Stream (1 group = 1 s = 2 frames at 2 fps, 448 px square):
    [ past: P groups of FineVideo filler, target episode ends D s before the window,
      optional L distractor episodes earlier ] [ window: W groups of filler ]  -> question

Memories (all read by the frozen host in context, as a first video before the window):
    oracle   the target episode groups (448)
    fifo     the last N evicted groups at 224 px (N=80 s)
    key      1 group (1 s) every S s over the whole past, random phase per group (448)
    mean     past cut into P/S segments of S s; visual tokens averaged per position (448)
    empty    window only;  blind  no video at all
    fifo448  the last 20 evicted groups at 448 px (same token budget as fifo)
    tail     only the last 5 s of the episode (its final still hold): must NOT pass as gist
    other    oracle memory of ANOTHER VARIANT of the same group (same scene, filler,
             options; different answer): target accuracy must fall below chance and the
             donor's answer must be picked about as often as the oracle is right
    solo     episode alone with the generator's own prompt (= the generator ceiling)

    CUDA_VISIBLE_DEVICES=0 python ladder.py --families flat,init_count --groups 2 --out runs/smoke.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

# ------------------------------------------------------------------ generator (snapshot)
GEN = os.environ.get("LADDER_GEN", os.path.join(HERE, "gen_scenes_snapshot.py"))


def load_gen():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_scenes_snap", GEN)
    G = importlib.util.module_from_spec(spec)
    sys.modules["gen_scenes_snap"] = G
    spec.loader.exec_module(G)
    G._hash = hashlib.md5(open(GEN, "rb").read()).hexdigest()[:8]
    return G


STREAM_INTRO = ("Somewhere earlier in this video stream there was a clip of cartoon people, each "
                "wearing a differently coloured shirt, in a room with a door. The question is "
                "about that clip. ")
LOAD_INTRO = ("Several clips of cartoon people, each wearing a differently coloured shirt, in a "
              "room with a door appeared earlier in this video stream. The question is about the "
              "MOST RECENT of these clips. ")

# ------------------------------------------------------------------ calibration family: flat screens
FLAT = {"red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
        "orange": (255, 165, 0), "purple": (128, 0, 128), "white": (255, 255, 255),
        "black": (0, 0, 0)}
TAIL = "Answer with the option's letter from the given choices directly."


def flat_group(g):
    rng = random.Random(f"flat-{g}")
    cols = rng.sample(list(FLAT), 4)
    q = ("At one point the whole screen turned a single solid colour for a few seconds. "
         "Which colour was it?")
    p = "\n".join([q] + [f"{C.LET[i]}. {c}" for i, c in enumerate(cols)] + [TAIL])
    out = []
    for v, c in enumerate(cols):
        def frames(c=c):
            f = np.zeros((8, C.HI, C.HI, 3), np.uint8)
            f[:] = FLAT[c]
            return f
        out.append({"family": "flat", "kind": "calib", "group": g, "variant": v, "answer": v,
                    "n_opt": 4, "frames": frames, "p_stream": p, "p_load": p, "p_solo": p,
                    "options": cols})
    return out


def scene_group(G, fam, g):
    out = []
    for it in G.make_group(fam, g):
        ep = it["episode"]
        p = it["prompt"]
        assert p.startswith(G.INTRO), "generator INTRO changed"
        body = p[len(G.INTRO):].replace("the video", "the clip")
        out.append({"family": fam, "kind": "gist" if fam in G.GIST else "detail", "group": g,
                    "variant": it["variant"], "answer": it["answer"],
                    "n_opt": len(it["options"]), "frames": (lambda ep=ep: G.render(ep)),
                    "p_stream": STREAM_INTRO + body, "p_load": LOAD_INTRO + body, "p_solo": p,
                    "options": it["options"]})
    return out


def make_groups(G, fam, g):
    return flat_group(g) if fam == "flat" else scene_group(G, fam, g)


# ------------------------------------------------------------------ stream + memories
def place(stream, ep, end):
    """Overwrite stream[end-E:end] with ep (both [*, n, d]); returns start."""
    E = ep.shape[0]
    assert end - E >= 0, (end, E)
    stream[end - E:end] = ep
    return end - E


def memories(a, filler, ep, D, phase, distractors=()):
    """Build {name: (mem tensor, window tensor)} for one stream."""
    P, W = a.past, a.window
    hi = filler[a.res][:P + W].clone()          # base resolution (448 by default)
    lo = filler["lo"][:P + W].clone()
    s0 = place(hi, ep[a.res], P - D)
    place(lo, ep["lo"], P - D)
    # distractors: earlier than the target, non overlapping, 2 s gaps
    end = s0 - 2
    for dep, gap in distractors:
        end -= gap
        if end - dep[a.res].shape[0] < 0:
            break
        st = place(hi, dep[a.res], end)
        place(lo, dep["lo"], end)
        end = st - 2
    win = hi[P:P + W]
    out = {}
    out["fifo"] = lo[P - a.fifo:P]
    out["fifo448"] = hi[P - a.fifo_hi:P]
    idx = list(range(P - 1 - phase, -1, -a.key_every))[::-1]
    out["key"] = hi[idx]
    nseg = P // a.seg
    segs = [hi[P - a.seg * (k + 1):P - a.seg * k].float().mean(0) for k in range(nseg)][::-1]
    out["mean"] = torch.stack(segs).to(hi.dtype)
    return out, win


def mib(t):
    return 0.0 if t is None else t.shape[0] * t.shape[1] * C.HIDDEN * 2 / 2 ** 20


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="flat")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gstart", type=int, default=0)
    ap.add_argument("--dists", default="0,30,90")
    ap.add_argument("--load", type=int, default=3, help="distractor episodes for the load condition")
    ap.add_argument("--load-dist", type=int, default=30, dest="load_dist")
    ap.add_argument("--past", type=int, default=176)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--fifo", type=int, default=80)
    ap.add_argument("--fifo-hi", type=int, default=20, dest="fifo_hi",
                    help="FIFO at 448 px, same token budget as --fifo at 224 px")
    ap.add_argument("--res", default="hi", choices=["hi", "lo"],
                    help="base resolution of window/oracle/key/mean/tail/other: hi=448, lo=224 (4x fewer tokens)")
    ap.add_argument("--tail", type=int, default=5, help="groups of the episode end kept by `tail`")
    ap.add_argument("--key-every", type=int, default=8, dest="key_every")
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--n-filler", type=int, default=120, dest="n_filler")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of conditions to compute")
    a = ap.parse_args()
    dists = [int(x) for x in a.dists.split(",")]
    only = set(a.only.split(",")) if a.only else None

    G = load_gen()
    host = C.Host()
    cache = C.EmbCache(host, a.cache, ram=24)
    fvs = C.fv_ids(a.n_filler)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["family"], r["group"], r["variant"], r["cond"]))
    fo = open(a.out, "a")
    t0, nf = time.time(), 0

    def filler_for(fam, g):
        h = int(hashlib.md5(f"{fam}-{g}".encode()).hexdigest(), 16)
        vid = fvs[h % len(fvs)]
        return cache.get(f"fv_{vid}_{a.past + a.window}",
                         lambda: C.read_filler(vid, a.past + a.window)), vid

    def episode(it):
        key = f"ep_{it['family']}_{it['group']}_{it['variant']}_{G._hash}"
        return cache.get(key, lambda: C.pad_even(it["frames"]()))

    def emit(it, cond, lg, ntok, mem, extra=None):
        nonlocal nf
        pred = int(lg.argmax())
        r = {"family": it["family"], "kind": it["kind"], "group": it["group"],
             "variant": it["variant"], "answer": it["answer"], "n_opt": it["n_opt"],
             "cond": cond, "pred": pred, "correct": int(pred == it["answer"]),
             "lg": [round(x, 3) for x in lg.tolist()], "ntok": ntok,
             "mem_tok": 0 if mem is None else mem.shape[0] * mem.shape[1],
             "mem_mib": round(mib(mem), 2)}
        if extra:
            r.update(extra)
        fo.write(json.dumps(r) + "\n")
        nf += 1

    MEMS = ("fifo", "fifo448", "key", "mean")
    fams = a.families.split(",")
    gi = 0
    for g in range(a.gstart, a.gstart + a.groups):        # groups outer: partial runs stay balanced
        for fam in fams:
            gi += 1
            if gi % a.nshards != a.shard:
                continue
            items = make_groups(G, fam, g)
            filler, vid = filler_for(fam, g)
            rng = random.Random(f"ladder-{fam}-{g}")
            # key phase drawn per (group, D), shared by the 4 variants (audit #11)
            phases = {D: random.Random(f"phase-{fam}-{g}-{D}").randrange(a.key_every)
                      for D in dists + [a.load_dist]}
            phase_load = random.Random(f"phase-{fam}-{g}-load").randrange(a.key_every)
            dfams = [f for f in fams if f != "flat"] or ["flat"]
            dists_eps = []
            for k in range(a.load):
                df = rng.choice(dfams)
                ditems = make_groups(G, df, 20_000 + 97 * g + k)
                dists_eps.append((episode(rng.choice(ditems)), rng.randrange(2, 8)))
            grp_eb = None
            for it in items:
                def need(c):
                    if only is not None and c not in only:
                        return False
                    return (it["family"], it["group"], it["variant"], c) not in done
                ep = episode(it)
                E = ep[a.res].shape[0]
                win = filler[a.res][a.past:a.past + a.window]
                if need("solo"):
                    lg, nt = host.score(ep[a.res], None, it["p_solo"], it["n_opt"])
                    emit(it, "solo", lg, nt, ep[a.res])
                if need("oracle"):
                    lg, nt = host.score(ep[a.res], win, it["p_stream"], it["n_opt"])
                    emit(it, "oracle", lg, nt, ep[a.res])
                if need("other"):
                    # counterfactual donor: SAME group (same scene, filler, options), next variant
                    dv = (it["variant"] + 1) % len(items)
                    dit = items[dv]
                    assert dit["answer"] != it["answer"] and dit["options"] == it["options"]
                    oep = episode(dit)
                    lg, nt = host.score(oep[a.res], win, it["p_stream"], it["n_opt"])
                    emit(it, "other", lg, nt, oep[a.res],
                         {"donor_answer": dit["answer"],
                          "donor_hit": int(int(lg.argmax()) == dit["answer"])})
                if need("tail") and it["family"] != "flat":
                    t = ep[a.res][-a.tail:]
                    lg, nt = host.score(t, win, it["p_stream"], it["n_opt"])
                    emit(it, "tail", lg, nt, t)
                # empty / blind are identical across the variants of a group: score once, copy
                if need("empty"):
                    if grp_eb is None:
                        e_lg, e_nt = host.score(None, win, it["p_stream"], it["n_opt"])
                        b_lg, b_nt = host.score(None, None, it["p_stream"], it["n_opt"])
                        grp_eb = (e_lg, e_nt, b_lg, b_nt)
                    e_lg, e_nt, b_lg, b_nt = grp_eb
                    emit(it, "empty", e_lg, e_nt, None)
                    emit(it, "blind", b_lg, b_nt, None)
                buf = {"fifo": a.fifo, "fifo448": a.fifo_hi}
                for D in dists:
                    if a.past - D - E < 0:
                        continue
                    if not any(need(f"{m}@{D}") for m in MEMS):
                        continue
                    mems, w = memories(a, filler, ep, D, phases[D])
                    for m, mem in mems.items():
                        if need(f"{m}@{D}"):
                            lg, nt = host.score(mem, w, it["p_stream"], it["n_opt"])
                            emit(it, f"{m}@{D}", lg, nt, mem,
                                 {"D": D, "E": E, "mem": m, "L": 0,
                                  "in_buf": int(D + E <= buf[m]) if m in buf else None})
                lconds = [f"{m}@{a.load_dist}L{a.load}" for m in MEMS]
                if a.load and any(need(c) for c in lconds):
                    D = a.load_dist
                    mems, w = memories(a, filler, ep, D, phase_load, dists_eps)
                    for m, mem in mems.items():
                        c = f"{m}@{D}L{a.load}"
                        if need(c):
                            lg, nt = host.score(mem, w, it["p_load"], it["n_opt"])
                            emit(it, c, lg, nt, mem, {"D": D, "E": E, "mem": m, "L": a.load})
                fo.flush()
            print(f"{fam} g{g} done  [{nf} rows, {(time.time() - t0) / 60:.1f} min, "
                  f"filler {vid}]", flush=True)
    print("LADDER-FINI", flush=True)


if __name__ == "__main__":
    main()
