"""22/10 : graft inputs for the certified scene benchmark v3 (gen_tableau), ladder3 layout.

Stream (ladder/ladder3.py + core.py, 2 fps, 448 px square, 1 group = 2 frames = 1 s):
    [ past: P=160 s of filler, the 6 s tableau ENDING D s before the window ]
    [ window: W=4 s of the same filler ]  -> question (generator prompt, unchanged as in ladder3)
The graft's memory receives the past cut into clips of CLIP s (CLIP*2 frames), each clip
encoded ALONE by the frozen host (layer-24 states, 1 token in STRIDE, PCA-projected with
--basis, i.e. exactly ``branch.down(h)``), M-RoPE time asserted at 2 fps (spg = 1.0 s).
The student sees the window only, in the ladder's ``empty`` format, plus its memory.

Speed: as in core.Host, the vision tower attends within ONE temporal group, so the visual
tokens of each group are computed once (filler groups once per filler video, tableau groups
once per variant) and injected into the LLM pass of each clip (stopped at layer 24). Checked
at start-up against the plain processor path (``--check``).

Counterfactual groups are kept: the 4 variants of a group share filler, window, prompt and
options; only the clip(s) holding the tableau differ ("patches").

Filler (--filler):
  * proc : the v3 procedural outdoor scenery seeded per group (filler_seed), EXACTLY ladder3's
           eval streams (default for --split eval: gstart 10000, D in {0,120}).
  * fv   : FineVideo. train: TRAIN split (stream_targets.is_eval False) minus the 120 fillers
           of ladder v1; a pool of --pool videos, each encoded once on a clip grid
           (start 10 + CLIP*j); a group takes a random clip-aligned slice (default for train).
           eval: ladder v1 mapping (core.fv_ids(120), md5(f"{fam}-{g}")), start 10 s.
Families: count, colour, big_object, sign_number, hat_color; style v3b for all (= ladder3),
generator = the frozen copy ladder/v3/gen_tableau.py (md5 identical to scenes/).

Outputs (under --out):
  fill_<split>/<vid>.pt    {"vid","start","clip_s","z": [J, n_tok, 128] fp16}
  <split>/<fam>_<shard>.pt list of groups {family, group, split, vid, j0, start, P, W, clip_s,
                           dists, prompt, options, answers[4], facts[4],
                           patches[D][variant] = {clip index in the past: [n_tok,128] fp16}}
Sharding is by filler video (each process owns disjoint videos and their groups).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/work/titans-mac/src python prep_v3.py --split train \\
        --groups 400 --pool 60 --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/work/iclr/ladder")
sys.path.insert(0, f"{HOME}/work/iclr/ladder/v3")             # frozen generator copy (ladder3)
import core as C                     # noqa: E402  (read-only import: layout, filler reader)
import gen_tableau as T              # noqa: E402

FAMS = ["count", "colour", "big_object", "sign_number", "hat_color"]
STYLE = T.TStyle(count_gap=True, door_bright=True)                  # v3b, = ladder3.STYLE
FPS, SIDE = C.FPS, C.HI
assert FPS == T.FPS and SIDE == T.SIZE


def make_group(fam, g, split):
    return T.make_group(fam, g, STYLE, split=split)


def tableau_frames(it):
    return T.render_tableau(it["tableau"], T.E * FPS)               # [12, 448, 448, 3]


def h32(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def read(vid, seconds, start):
    return C.read_filler(vid, seconds, start_s=start)               # 2 fps, 448 square


def filler_frames(rec, seconds, t0):
    """``seconds`` s of the record's filler starting t0 s into its stream (past starts at 0)."""
    if rec["filler"] == "proc":
        return T.render_filler(rec["filler_seed"], 2 * seconds, 2 * t0)
    return read(rec["vid"], seconds, rec["start"] + t0)


class Encoder:
    """Frozen host: per-group visual tokens, then layer-24 states of one clip, stride, PCA."""

    class _Stop(Exception):
        pass

    def __init__(self, basis, stride, clip_s, device="cuda"):
        from transformers import AutoProcessor

        from titans.graft.branch import fold_pca
        from titans.graft.check_qwen import load_host
        from titans.graft.fit_basis import video_inputs
        from titans.graft.qwen import default_layer, find_decoder_layers
        self.model = load_host(C.MODEL, torch.bfloat16, device)
        self.proc = AutoProcessor.from_pretrained(C.MODEL)
        self.device, self.stride, self.nf = device, stride, 2 * clip_s
        layers = find_decoder_layers(self.model)
        self.site = default_layer(len(layers))
        self.got = []

        def hook(_m, _i, out):
            self.got.append((out[0] if isinstance(out, tuple) else out).detach())
            raise Encoder._Stop
        layers[self.site].register_forward_hook(hook)
        W, b = fold_pca(torch.load(basis, map_location="cpu", weights_only=False)["pca"])
        self.W, self.b = W.float().to(device), b.float().to(device)
        self.vtok = self.model.config.video_token_id
        self._inject = None
        inner = self.model.model
        orig = inner.get_video_features

        def fake(pv, thw, **kw):
            if self._inject is None:
                return orig(pv, thw, **kw)
            return SimpleNamespace(pooler_output=(self._inject,))
        inner.get_video_features = fake
        # every clip has the same text, shape, grid and time step: one template (fixed path)
        tpl = video_inputs(self.proc, torch.zeros(self.nf, SIDE, SIDE, 3), "", FPS)
        assert float(tpl["second_per_grid_ts"][0]) == 1.0
        self.tpl = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tpl.items()}
        self.mask = self.tpl["input_ids"][0] == self.vtok

    @torch.no_grad()
    def vis(self, frames: np.ndarray, chunk: int = 16) -> torch.Tensor:
        """uint8 [F,448,448,3] (F even) -> bf16 [F/2, 256, 2048] on GPU (core.Host.encode)."""
        assert frames.shape[0] % 2 == 0
        out = []
        for i in range(0, frames.shape[0], 2 * chunk):
            vp = self.proc.video_processor(videos=[frames[i:i + 2 * chunk]], return_tensors="pt",
                                           do_sample_frames=False)
            pv = vp["pixel_values_videos"].to(self.device, torch.bfloat16)
            thw = vp["video_grid_thw"].to(self.device)
            emb = self.model.model.visual(pv, grid_thw=thw).pooler_output
            t, h, w = thw[0].tolist()
            out.append(emb.reshape(t, h * w // 4, -1))
        return torch.cat(out)

    @torch.no_grad()
    def llm(self, groups: torch.Tensor) -> torch.Tensor:
        """[nf/2, 256, 2048] visual tokens of ONE clip -> [n_tok/stride, 128] fp16 (CPU)."""
        assert groups.shape[0] * 2 == self.nf
        self._inject = groups.reshape(-1, groups.shape[-1]).to(self.device)
        self.got.clear()
        try:
            self.model(**self.tpl, use_cache=False)
        except Encoder._Stop:
            pass
        finally:
            self._inject = None
        h = self.got[0][0][self.mask][::self.stride].float()
        return (h @ self.W.T + self.b).half().cpu()

    @torch.no_grad()
    def plain(self, frames: np.ndarray) -> torch.Tensor:
        """Reference path (processor + full vision), for --check."""
        from titans.graft.fit_basis import video_inputs
        inp = video_inputs(self.proc, torch.from_numpy(frames).float(), "", FPS)
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        self.got.clear()
        try:
            self.model(**inp, use_cache=False)
        except Encoder._Stop:
            pass
        h = self.got[0][0][inp["input_ids"][0] == self.vtok][::self.stride].float()
        return (h @ self.W.T + self.b).half().cpu()


def save(out, gdir, a, quiet=False):
    for fam, groups in out.items():
        groups.sort(key=lambda r: r["group"])
        dst = f"{gdir}/{fam}_{a.tag}{a.shard}.pt"
        torch.save(groups, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        if not quiet:
            print(f"{fam}: {len(groups)} groups -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--families", default=",".join(FAMS))
    ap.add_argument("--groups", type=int, default=400, help="groups per family")
    ap.add_argument("--gstart", type=int, default=None, help="default: train 0, eval 10000 (ladder3)")
    ap.add_argument("--dists", default="0,120", help="eval D values (ladder3 fifo-dists / km-dist)")
    ap.add_argument("--filler", choices=["proc", "fv"], default=None,
                    help="default: eval proc (= ladder3), train fv (FineVideo train split)")
    ap.add_argument("--past", type=int, default=160)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--clip", type=int, default=8, help="seconds per memory clip")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--pool", type=int, default=60, help="train filler videos")
    ap.add_argument("--pool-clips", type=int, default=30, dest="pool_clips")
    ap.add_argument("--basis", default=f"{HOME}/work/titans-mac/cache/qwen_l24_pca128.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--check", type=int, default=1)
    ap.add_argument("--tag", default="", help="file name tag, to extend a split without overwriting")
    ap.add_argument("--out", default=f"{HOME}/work/iclr/graft_v3/data")
    a = ap.parse_args()
    assert a.past % a.clip == 0 and a.window <= a.clip, "clip grid must tile the past"
    gstart = a.gstart if a.gstart is not None else (0 if a.split == "train" else 10_000)
    a.filler = a.filler or ("proc" if a.split == "eval" else "fv")
    ncl, gpc = a.past // a.clip, a.clip                # clips in the past, groups per clip
    fdir, gdir = f"{a.out}/fill_{a.split}", f"{a.out}/{a.split}"
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(gdir, exist_ok=True)
    from titans.graft.stream_targets import is_eval
    ladder_fv = C.fv_ids(120)
    pool = None
    if a.split == "train" and a.filler == "fv":
        rows = [json.loads(l) for l in open(os.path.join(C.FV, "manifest.jsonl"))]
        need = 10 + a.clip * a.pool_clips + 2
        pool = sorted(r["id"] for r in rows if not is_eval(r["id"]) and r["id"] not in set(ladder_fv)
                      and r["duration"] >= need)
        random.Random("graft_v3-pool").shuffle(pool)
        pool = pool[:a.pool]
        assert not set(pool) & set(ladder_fv)
    enc = Encoder(a.basis, a.stride, a.clip)
    if a.check:
        fr = read(ladder_fv[0], a.clip, 50.0)
        z0, z1 = enc.plain(fr), enc.llm(enc.vis(fr))
        rel = float((z1.float() - z0.float()).norm() / z0.float().norm())
        print(f"check injected vs processor path: rel diff {rel:.2e}", flush=True)
        assert rel < 2e-2, rel
    meta = {"args": vars(a), "fps": FPS, "side": SIDE, "site": enc.site, "spg": 1.0,
            "basis": a.basis, "ncl": ncl, "families": a.families.split(","),
            "pool": pool, "gen": T.__file__}
    json.dump(meta, open(f"{a.out}/meta_{a.split}_{a.shard}.json", "w"), indent=1)

    # 1. plan: every (family, group) -> filler key, slice, D values; keep this shard's fillers
    plan = defaultdict(list)
    for fam in a.families.split(","):
        for g in range(gstart, gstart + a.groups):
            rng = random.Random(f"graft_v3-{fam}-{g}")
            dists = ([int(x) for x in a.dists.split(",")] if a.split == "eval"
                     else [rng.randrange(0, a.past - T.E + 1)])
            if a.filler == "proc":
                key, j0 = f"v3-{a.split}-{fam}-{g}", 0                 # = make_group filler_seed
            elif a.split == "eval":
                key, j0 = ladder_fv[h32(f"{fam}-{g}") % len(ladder_fv)], 0   # = ladder.filler_for
            else:
                key = pool[h32(f"pool-{fam}-{g}") % len(pool)]
                j0 = rng.randrange(0, a.pool_clips - ncl)          # window inside clip j0+ncl
            if h32(f"shard-{key}") % a.nshards == a.shard:
                plan[key].append((fam, g, j0, dists))
    print(f"shard {a.shard}/{a.nshards}: {len(plan)} filler videos, "
          f"{sum(map(len, plan.values()))} groups", flush=True)

    # 2. per filler video: visual tokens of every group once, clip states, then its groups
    out = defaultdict(list)
    done = set()
    for fam in a.families.split(","):                    # resume from (partial) files
        dst = f"{gdir}/{fam}_{a.tag}{a.shard}.pt"
        if os.path.exists(dst):
            out[fam] = torch.load(dst, weights_only=False)
            done |= {(fam, r["group"]) for r in out[fam]}
    if done:
        print(f"resume: {len(done)} groups already done", flush=True)
    t0, n = time.time(), 0
    for vi, (vid, todo) in enumerate(sorted(plan.items())):
        todo = [x for x in todo if (x[0], x[1]) not in done]
        if not todo:
            continue
        nclips = a.pool_clips if (a.filler == "fv" and a.split == "train") else ncl
        fr = (T.render_filler(vid, 2 * nclips * a.clip, 0) if a.filler == "proc"
              else read(vid, nclips * a.clip, 10.0))
        G = torch.cat([enc.vis(fr[i:i + 128]) for i in range(0, fr.shape[0], 128)]).cpu()
        del fr
        z = torch.stack([enc.llm(G[k * gpc:(k + 1) * gpc]) for k in range(nclips)])
        torch.save({"vid": vid, "start": 10.0, "clip_s": a.clip, "z": z}, f"{fdir}/{vid}.pt")
        for fam, g, j0, dists in todo:
            items = make_group(fam, g, a.split)
            assert a.filler != "proc" or items[0]["filler_seed"] == vid
            tabs = [enc.vis(tableau_frames(it)).cpu() for it in items]      # [6, 256, 2048] each
            base = G[j0 * gpc:(j0 + ncl) * gpc]                                # the P s of the past
            patches = {}
            for D in dists:
                g1 = a.past - D
                g0 = g1 - T.E
                ks = sorted({x // gpc for x in range(g0, g1)})
                pv = []
                for tb in tabs:
                    st = base.clone()
                    st[g0:g1] = tb
                    pv.append({k: enc.llm(st[k * gpc:(k + 1) * gpc]) for k in ks})
                patches[D] = pv
            out[fam].append({"family": fam, "kind": "gist" if fam in T.GIST else "detail",
                             "group": g, "split": a.split, "prompt": items[0]["prompt"],
                             "options": items[0]["options"], "answers": [it["answer"] for it in items],
                             "facts": [it["facts"] for it in items], "P": a.past, "W": a.window,
                             "clip_s": a.clip, "filler": a.filler, "fill_key": vid,
                             "vid": None if a.filler == "proc" else vid,
                             "filler_seed": vid if a.filler == "proc" else None,
                             "j0": j0, "start": 10.0 + a.clip * j0,
                             "dists": dists, "patches": patches})
            n += 1
        print(f"  video {vi + 1}/{len(plan)} {vid}: {len(todo)} groups | {n} groups, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if vi % 10 == 9:
            save(out, gdir, a, quiet=True)                   # partial files, readable at any time
    save(out, gdir, a)
    print(f"PREP-V3-FINI {n} groups {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
