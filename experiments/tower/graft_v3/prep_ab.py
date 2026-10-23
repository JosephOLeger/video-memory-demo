"""23/10 : data for the A/B graft experiment ("gist only" vs "everything").

Layout = ladder3 / prep_v3: 2 fps, 448 px, 1 group = 2 frames = 1 s; past P = 160 s cut in 8 s
clips, each clip encoded ALONE by frozen Qwen2.5-VL-3B up to layer 24 (M-RoPE time asserted,
spg = 1.0 s), 256 tokens kept per clip, projected on the PCA basis. Window W = 4 s after the past.

Differences with prep_v3 (all deliberate):
  * basis: 512 dims, refit on layer 24 with the CORRECTED fps path (mode ``basis``), in domain
    (train split: procedural filler, rich tableaux, student window + question tokens).
  * token subsampling: 1 token in 8 per group with a ROTATING offset (group g keeps raster
    positions g%8, g%8+8, ...), so 8 consecutive seconds cover all 256 positions. prep_v3 kept
    columns 0 and 8 only, which never sees most of a small sign.
  * training data: procedural filler (bank of --nfill streams, 30 clips each, train seeds),
    rich tableaux (gen_ab), dense free-form questions + TEACHER top-k log-probs on the answer
    tokens. Teacher = frozen Qwen with the tableau visible (ladder oracle format: "Memory of
    earlier parts..." + tableau video, "The most recent seconds..." + window video, question).
  * eval data: the v3 bench (ladder3 streams, per-group procedural filler, groups 10000+,
    D in {0, 120}), families count + colour from v3c, the others from v3 (frozen copies in
    ab_gen/), plus the graft-independent controls computed once: window, oracle and
    other-variant MCQ letter logits, and free-form greedy answers (oracle, window).

Modes:
  basis : collect layer-24 states, fit PCA k  ->  cache_ab/qwen_l24_pca{k}_fpsfix.pt
  fill  : train filler bank                   ->  data_ab/fill/fill_{i}.pt  (z [30, 256, k] fp16)
  train : training groups (sharded)           ->  data_ab/train/ab_{tag}{shard}.pt
  eval  : bench groups (sharded by family)    ->  data_ab/eval/{fam}_{shard}.pt
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
sys.path.insert(0, HERE)
import gen_ab as GA                  # noqa: E402

T, C3 = GA.T, GA.C3
FPS, SIDE = T.FPS, T.SIZE
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MEM_TXT = "Memory of earlier parts of the video stream:"
WIN_TXT = "The most recent seconds of the video stream:"
LET = "ABCD"
STYLE = T.TStyle(count_gap=True, door_bright=True)          # v3b = ladder3
BENCH = ["count", "colour", "big_object", "room_state", "side", "sign_number", "hat_color"]
V3C_FAMS = ("count", "colour")
P_PAST, W_WIN, CLIP, STRIDE = 160, 4, 8, 8
NCL = P_PAST // CLIP
FILL_CLIPS = 30
TOPK = 20
PH = "\x00Q\x00"                                             # prompt placeholder


def bench_group(fam, g, split="eval"):
    if fam in V3C_FAMS:
        return C3.make_group(fam, g, STYLE, split=split)
    return T.make_group(fam, g, STYLE, split=split)


def h32(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def to_dev(d, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in d.items()}


# ---------------------------------------------------------------------------- host
class Host:
    """Frozen Qwen + (a) layer-24 clip encoder with injected visual tokens, (b) prefix/suffix
    chat utilities (one prefix forward with KV cache, a batch of question suffixes)."""

    class _Stop(Exception):
        pass

    def __init__(self, basis=None, device="cuda", model=None, proc=None):
        from transformers import AutoProcessor

        from titans.graft.check_qwen import load_host
        from titans.graft.fit_basis import video_inputs
        from titans.graft.qwen import default_layer, find_decoder_layers
        self.model = model if model is not None else load_host(MODEL, torch.bfloat16, device)
        self.proc = proc if proc is not None else AutoProcessor.from_pretrained(MODEL)
        self.tok = self.proc.tokenizer
        self.device = device
        layers = find_decoder_layers(self.model)
        self.site = default_layer(len(layers))
        self.got, self.stop = [], False

        def hook(_m, _i, out):
            if self.stop:
                self.got.append((out[0] if isinstance(out, tuple) else out).detach())
                raise Host._Stop
        self._hook = layers[self.site].register_forward_hook(hook)
        self.W = self.b = None
        if basis:
            self.set_basis(basis)
        self.vtok = self.model.config.video_token_id
        self._inject = None
        inner = self.model.model
        orig = inner.get_video_features

        def fake(pv, thw, **kw):
            if self._inject is None:
                return orig(pv, thw, **kw)
            from types import SimpleNamespace
            inj = self._inject if isinstance(self._inject, tuple) else (self._inject,)
            return SimpleNamespace(pooler_output=inj)
        inner.get_video_features = fake
        tpl = video_inputs(self.proc, torch.zeros(2 * CLIP, SIDE, SIDE, 3), "", FPS)
        assert float(tpl["second_per_grid_ts"][0]) == 1.0
        self.tpl = to_dev(tpl, device)
        self.mask = self.tpl["input_ids"][0] == self.vtok
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def set_basis(self, path, k=None):
        from titans.graft.branch import fold_pca
        pca = torch.load(path, map_location="cpu", weights_only=False)["pca"]
        W, b = fold_pca(pca)
        if k:
            W, b = W[:k], b[:k]
        self.W, self.b = W.float().to(self.device), b.float().to(self.device)

    # ---- clip encoding
    @torch.no_grad()
    def vis(self, frames: np.ndarray, chunk: int = 16) -> torch.Tensor:
        """uint8 [F,448,448,3] -> bf16 [F/2, 256, 2048] (vision tower attends within a group)."""
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
    def clip_states(self, groups: torch.Tensor) -> torch.Tensor:
        """[8, 256, 2048] visual tokens of ONE clip -> layer-24 states, rotating 1-in-8 -> [256, 2048]."""
        ng = groups.shape[0]
        assert ng * 2 == 2 * CLIP
        self._inject = groups.reshape(-1, groups.shape[-1]).to(self.device)
        self.got.clear()
        self.stop = True
        try:
            self.model(**self.tpl, use_cache=False)
        except Host._Stop:
            pass
        finally:
            self._inject, self.stop = None, False
        h = self.got[0][0][self.mask].view(ng, -1, self.got[0].shape[-1])      # [8, 256, 2048]
        npos = h.shape[1]
        idx = torch.stack([(g % STRIDE) + STRIDE * torch.arange(npos // STRIDE) for g in range(ng)])
        return torch.gather(h, 1, idx.to(h.device)[..., None].expand(-1, -1, h.shape[-1])).reshape(-1, h.shape[-1])

    @torch.no_grad()
    def clip_z(self, groups):
        h = self.clip_states(groups).float()
        return (h @ self.W.T + self.b).half().cpu()

    # ---- chat with prefix cache
    def chat_prefix(self, pieces):
        """pieces [(label, frames)] -> (prefix inputs on device, suffix template text)."""
        content, vids = [], []
        for lab, fr in pieces:
            content += [{"type": "text", "text": lab}, {"type": "video"}]
            vids.append(fr)
        content.append({"type": "text", "text": PH})
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        pre, post = text.split(PH)
        assert pre.endswith("<|vision_end|>"), pre[-40:]
        inp = self.proc(text=[pre], videos=vids, return_tensors="pt", do_sample_frames=False,
                        video_metadata=[{"fps": FPS, "total_num_frames": len(v)} for v in vids])
        from titans.graft.fit_basis import check_temporal
        check_temporal(inp, FPS, self.proc, n_videos=len(vids))
        return to_dev(inp, self.device), post

    def suffix_ids(self, prompt, post, answer=None):
        """Token ids of the question (+ template) and, if given, of the answer + <|im_end|>."""
        q = self.tok.encode(prompt + post, add_special_tokens=False)
        if answer is None:
            return q, []
        a = self.tok.encode(answer, add_special_tokens=False) + [self.im_end]
        return q, a

    def prefix_forward(self, pre, batch=1, grad=False):
        """Run the prefix with a KV cache; returns (cache, next position)."""
        from transformers import DynamicCache
        cache = DynamicCache()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            self.model.model.rope_deltas = None
            pos, _ = self.model.model.get_rope_index(pre["input_ids"], pre["mm_token_type_ids"],
                                                     video_grid_thw=pre["video_grid_thw"],
                                                     second_per_grid_ts=pre["second_per_grid_ts"],
                                                     attention_mask=pre["attention_mask"])
            inp = {k: v for k, v in pre.items()}
            if batch > 1:
                for k in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    inp[k] = inp[k].expand(batch, -1)
                inp["pixel_values_videos"] = inp["pixel_values_videos"].repeat(batch, 1)
                inp["video_grid_thw"] = inp["video_grid_thw"].repeat(batch, 1)
                inp["second_per_grid_ts"] = inp["second_per_grid_ts"].repeat(batch)
                pos = pos.expand(-1, batch, -1)
            self.model(**inp, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return cache, int(pos.max()) + 1

    def suffix_forward(self, cache, p0, seqs, expand=True, prefix_len=None):
        """seqs: list of token-id lists (right-padded). Cache batch 1 is repeated to len(seqs).
        Returns the last hidden states [B, Lmax, 2048] (apply ``head`` on the positions you need)."""
        B = len(seqs)
        L = max(len(s) for s in seqs)
        ids = torch.full((B, L), self.tok.pad_token_id or 0, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        ids = ids.to(self.device)
        if expand and B > 1:
            cache.batch_repeat_interleave(B)
        plen = cache.get_seq_length()
        pos = (p0 + torch.arange(L, device=self.device)).view(1, 1, L).expand(3, B, L)
        am = torch.ones(B, plen + L, dtype=torch.long, device=self.device)
        out = self.model.model(input_ids=ids, position_ids=pos, attention_mask=am, past_key_values=cache,
                               use_cache=True, cache_position=torch.arange(plen, plen + L, device=self.device))
        return out.last_hidden_state

    def head(self, h):
        """Hidden states [..., 2048] -> float32 logits (lm_head on the chosen positions only)."""
        return self.model.lm_head(h).float()

    @torch.no_grad()
    def full_logits(self, pieces, prompt, answer):
        """Reference path: one plain forward of [pieces, prompt, answer] (for the self-check)."""
        content, vids = [], []
        for lab, fr in pieces:
            content += [{"type": "text", "text": lab}, {"type": "video"}]
            vids.append(fr)
        content.append({"type": "text", "text": prompt})
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        text = text + answer + "<|im_end|>"
        inp = self.proc(text=[text], videos=vids, return_tensors="pt", do_sample_frames=False,
                        video_metadata=[{"fps": FPS, "total_num_frames": len(v)} for v in vids])
        inp = to_dev(inp, self.device)
        self.model.model.rope_deltas = None
        return self.model(**inp, use_cache=False).logits[0].float(), inp["input_ids"][0]


def answer_positions(q, a):
    """Suffix-local positions whose logits predict the answer tokens (and the final im_end)."""
    s = len(q)
    return list(range(s - 1, s - 1 + len(a)))


@torch.no_grad()
def teacher_topk(host, pieces, qs, k=TOPK, vis=None):
    """For each question dict (prompt, answer): top-k log-probs at the answer positions.
    ``vis``: precomputed visual tokens of the pieces ([groups, 256, 2048] each), injected."""
    pre, post = host.chat_prefix(pieces)
    if vis is not None:
        host._inject = tuple(x.reshape(-1, x.shape[-1]).to(host.device) for x in vis)
    try:
        cache, p0 = host.prefix_forward(pre)
    finally:
        host._inject = None
    seqs, spans = [], []
    for q in qs:
        qi, ai = host.suffix_ids(q["prompt"], post, q["answer"])
        seqs.append(qi + ai)
        spans.append((answer_positions(qi, ai), ai))
    mq = [i for i, q in enumerate(qs) if q.get("mcq")]          # lettered versions, same prefix
    for i in mq:
        seqs.append(host.suffix_ids(qs[i]["mcq"]["prompt"], post)[0])
    hs = host.suffix_forward(cache, p0, seqs)
    lid = [host.tok.encode(c, add_special_tokens=False)[0] for c in LET]
    out = []
    for i, (pp, ai) in enumerate(spans):
        lp = F.log_softmax(host.head(hs[i, pp]), -1)
        v, ix = lp.topk(k, -1)
        gold_lp = lp.gather(1, torch.tensor(ai, device=lp.device)[:, None])[:, 0]
        out.append({"ans_ids": ai, "top_ids": ix.int().cpu(), "top_lp": v.half().cpu(),
                    "gold_lp": gold_lp.cpu(), "argmax_ok": bool((lp.argmax(-1).cpu() == torch.tensor(ai)).all())})
    for j, i in enumerate(mq):
        n = len(qs[i]["mcq"]["options"])
        L = len(seqs[len(spans) + j])
        lg = host.head(hs[len(spans) + j, L - 1])[lid[:n]]
        out[i]["mcq_lp"] = F.log_softmax(lg, -1).cpu()
        out[i]["mcq_ok"] = int(lg.argmax()) == qs[i]["mcq"]["gold"]
    return out


# ---------------------------------------------------------------------------- frames
def fill_seed(i):
    return f"ab-train-fill-{i}"


def window_frames_train(rec):
    t = (rec["j0"] + NCL) * CLIP
    return T.render_filler(fill_seed(rec["fill"]), 2 * W_WIN, 2 * t)


def window_frames_eval(rec):
    return T.render_filler(rec["filler_seed"], 2 * W_WIN, 2 * P_PAST)


def place_distractors(rng, D, n):
    """Start seconds (in the 160 s past) of n 6 s calques, anywhere, >= 2 s from the target and
    from each other. Returns the list that fits (may be shorter than n)."""
    g1 = P_PAST - D
    busy = [(g1 - T.E - 2, g1 + 2)]
    out = []
    for _ in range(50):
        if len(out) >= n:
            break
        s0 = rng.randrange(0, P_PAST - T.E + 1)
        if all(s0 + T.E + 2 <= lo or s0 >= hi for lo, hi in busy):
            out.append(s0)
            busy.append((s0 - 2, s0 + T.E + 2))
    return sorted(out)


def patches_distract(host, clip_frames_fn, tabs_vis, D, calques):
    """Target tableau (per variant) at D + shared calques [(s0, frames->frames fn)].
    Returns (shared {clip: z} for clips with calques only, per-variant {clip: z})."""
    g1 = P_PAST - D
    g0 = g1 - T.E
    tk = sorted({x // CLIP for x in range(g0, g1)})
    base = {}
    cal_groups = {}
    for s0, fn in calques:                       # fn(filler frames of the 6 s) -> drawn frames
        ks = sorted({x // CLIP for x in range(s0, s0 + T.E)})
        for k in ks:
            if k not in base:
                base[k] = host.vis(clip_frames_fn(k))
        fr = np.concatenate([clip_frames_fn(k) for k in ks])[2 * (s0 - ks[0] * CLIP):2 * (s0 - ks[0] * CLIP + T.E)]
        cv = host.vis(fn(fr))
        for x in range(s0, s0 + T.E):
            cal_groups[x] = cv[x - s0]
    for k in tk:
        if k not in base:
            base[k] = host.vis(clip_frames_fn(k))
    withcal = {}
    for k, b in base.items():
        st = b.clone()
        for x, gv in cal_groups.items():
            if x // CLIP == k:
                st[x - k * CLIP] = gv
        withcal[k] = st
    shared = {k: host.clip_z(withcal[k]) for k in sorted({x // CLIP for x in cal_groups}) if k not in tk}
    per = []
    for tv in tabs_vis:
        pv = {}
        for k in tk:
            st = withcal[k].clone()
            for x in range(g0, g1):
                if x // CLIP == k:
                    st[x - k * CLIP] = tv[x - g0]
            pv[k] = host.clip_z(st)
        per.append(pv)
    return shared, per


def patches_for(host, clip_frames_fn, tabs_vis, D):
    """z of the clip(s) holding the tableau, per variant. clip_frames_fn(k) -> 16 frames of clip k."""
    g1 = P_PAST - D
    g0 = g1 - T.E
    ks = sorted({x // CLIP for x in range(g0, g1)})
    base = {k: host.vis(clip_frames_fn(k)) for k in ks}
    out = []
    for tv in tabs_vis:
        pv = {}
        for k in ks:
            st = base[k].clone()
            for x in range(g0, g1):
                if x // CLIP == k:
                    st[x - k * CLIP] = tv[x - g0]
            pv[k] = host.clip_z(st)
        out.append(pv)
    return out


# ---------------------------------------------------------------------------- modes
def mode_basis(host, a):
    """Layer-24 states, train split only: filler clips, tableau clips, student window+question."""
    from titans.worldmodel.bottleneck import fit_pca
    rng = random.Random("ab-basis")
    X, t0 = [], time.time()
    n_fill = n_tab = n_stu = 0
    for g in range(a.basis_groups):
        items = GA.make_group_ab(10_000_000 + g)          # disjoint from training groups
        seed = f"ab-basis-fill-{g}"
        fr = T.render_filler(seed, 2 * CLIP, 0)
        fv = host.vis(fr)
        X.append(host.clip_states(fv).float().cpu()[rng.sample(range(256), 64)])
        n_fill += 64
        it = items[rng.randrange(4)]
        tv = host.vis(GA.render_tableau(it["tableau"], 2 * T.E))
        st = fv.clone()
        off = rng.randrange(0, CLIP - T.E + 1)
        st[off:off + T.E] = tv
        X.append(host.clip_states(st).float().cpu()[rng.sample(range(256), 128)])
        n_tab += 128
        # student side: window + question, all positions at layer 24
        q = rng.choice(GA.questions_for(it, rng))
        win = T.render_filler(seed, 2 * W_WIN, 2 * CLIP)
        pre, post = host.chat_prefix([(WIN_TXT, win)])
        qi, _ = host.suffix_ids(q["prompt"], post)
        ids = torch.cat([pre["input_ids"], torch.tensor([qi], device=host.device)], 1)
        mm = torch.cat([pre["mm_token_type_ids"], torch.zeros(1, len(qi), dtype=pre["mm_token_type_ids"].dtype,
                                                              device=host.device)], 1)
        inp = dict(pre, input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm)
        host.got.clear()
        host.stop = True
        try:
            with torch.no_grad():
                host.model.model.rope_deltas = None
                host.model(**inp, use_cache=False)
        except Host._Stop:
            pass
        finally:
            host.stop = False
        h = host.got[0][0].float().cpu()
        sel = rng.sample(range(h.shape[0]), min(96, h.shape[0]))
        X.append(h[sel])
        n_stu += len(sel)
        if g % 50 == 0:
            print(f"  basis {g}/{a.basis_groups} {time.time() - t0:.0f}s", flush=True)
    X = torch.cat(X)
    print(f"activations {tuple(X.shape)} (filler {n_fill}, tableau {n_tab}, student {n_stu})", flush=True)
    ch = X.std(0)
    pca = fit_pca(X, a.k)
    print(f"PCA k={a.k}: {100 * pca['explained']:.1f} % variance; channel std ratio "
          f"{float(ch.max() / ch.median()):.0f}", flush=True)
    os.makedirs(os.path.dirname(a.basis), exist_ok=True)
    torch.save({"pca": pca, "model": MODEL, "layer": host.site, "d_model": X.shape[1],
                "tokens": int(X.shape[0]), "mix": {"filler": n_fill, "tableau": n_tab, "student": n_stu},
                "fps": FPS, "spg": 1.0, "gen": GA.GEN_MD5, "subsample": "rotating 1/8"}, a.basis)
    print("saved", a.basis)


def mode_fill(host, a):
    d = f"{a.out}/fill"
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for i in range(a.fill_start, a.nfill):
        if i % a.nshards != a.shard:
            continue
        dst = f"{d}/fill_{i}.pt"
        if os.path.exists(dst):
            continue
        fr = T.render_filler(fill_seed(i), 2 * CLIP * FILL_CLIPS, 0)
        z = []
        for j in range(FILL_CLIPS):
            z.append(host.clip_z(host.vis(fr[2 * CLIP * j:2 * CLIP * (j + 1)])))
        torch.save({"seed": fill_seed(i), "z": torch.stack(z)}, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        print(f"  fill {i} {time.time() - t0:.0f}s", flush=True)
    print(f"PREP-AB-FILL-FINI {(time.time() - t0) / 60:.1f} min", flush=True)


def mode_train(host, a):
    d = f"{a.out}/train"
    os.makedirs(d, exist_ok=True)
    dst = f"{d}/ab_{a.tag}{a.shard}.pt"
    out = torch.load(dst, weights_only=False) if os.path.exists(dst) else []
    done = {r["group"] for r in out}
    t0, n, ok = time.time(), 0, []
    gl = [g for g in range(a.gstart, a.gstart + a.groups) if g % a.nshards == a.shard and g not in done]
    for g in gl:
        rng = random.Random(f"ab-plan-{g}")
        fi, j0 = rng.randrange(a.nfill), rng.randrange(0, FILL_CLIPS - NCL)
        D = rng.randrange(0, P_PAST - T.E + 1)
        items = GA.make_group_ab(g)
        rec = {"group": g, "fill": fi, "j0": j0, "D": D, "facts": [it["facts"] for it in items]}
        seed = fill_seed(fi)
        tabs = [GA.render_tableau(it["tableau"], 2 * T.E) for it in items]
        tvis = [host.vis(t) for t in tabs]
        nd = rng.randint(a.dmin, a.dmax)
        cal = []
        for s0 in place_distractors(rng, D, nd):
            ctb = GA.sample_calque(random.Random(f"ab-cal-{g}-{s0}"))
            cal.append((s0, lambda fr, ctb=ctb: GA.render_calque(ctb, fr)))
        rec["calques"] = [s0 for s0, _ in cal]
        rec["shared"], rec["patches"] = patches_distract(
            host, lambda k: T.render_filler(seed, 2 * CLIP, 2 * CLIP * (j0 + k)), tvis, D, cal)
        win = window_frames_train(rec)
        wvis = host.vis(win)
        qs_all = []
        for v, it in enumerate(items):
            qs = GA.questions_for(it, random.Random(f"ab-q-{g}-{v}"))
            for q in qs:
                m = GA.mcq_for(q, it["facts"], random.Random(f"ab-m-{g}-{v}-{q['qfam']}"), q["intro"])
                if m:
                    q["mcq"] = {"prompt": m[0], "options": m[1], "gold": m[2]}
            for q, t in zip(qs, teacher_topk(host, [(MEM_TXT, tabs[v]), (WIN_TXT, win)], qs, vis=[tvis[v], wvis])):
                q["teacher"] = t
                ok.append((q["qfam"], t["argmax_ok"]))
                if "mcq_ok" in t:
                    ok.append((q["qfam"] + "_mcq", t["mcq_ok"]))
            qs_all.append(qs)
        rec["questions"] = qs_all
        out.append(rec)
        n += 1
        if n % 20 == 0 or n == len(gl):
            torch.save(out, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
            from collections import defaultdict
            acc = defaultdict(list)
            for f_, o_ in ok:
                acc[f_].append(o_)
            print(f"  train {n}/{len(gl)} {(time.time() - t0) / n:.2f} s/group | teacher greedy-exact "
                  + " ".join(f"{k} {100 * np.mean(v):.0f}" for k, v in sorted(acc.items())), flush=True)
    print(f"PREP-AB-TRAIN-FINI {n} groups {(time.time() - t0) / 60:.1f} min", flush=True)


@torch.no_grad()
def mcq_letters(host, pieces, prompt, n_opt):
    pre, post = host.chat_prefix(pieces)
    cache, p0 = host.prefix_forward(pre)
    qi, _ = host.suffix_ids(prompt, post)
    lg = host.head(host.suffix_forward(cache, p0, [qi])[0, -1])
    lid = [host.tok.encode(c, add_special_tokens=False)[0] for c in LET[:n_opt]]
    return lg[lid].cpu()


@torch.no_grad()
def greedy(host, pieces, prompt, max_new=6):
    pre, post = host.chat_prefix(pieces)
    cache, p0 = host.prefix_forward(pre)
    seq = host.suffix_ids(prompt, post)[0]
    out = []
    lg = host.head(host.suffix_forward(cache, p0, [seq])[0, -1])
    p = p0 + len(seq)
    for _ in range(max_new):
        t = int(lg.argmax())
        if t == host.im_end:
            break
        out.append(t)
        lg = host.head(host.suffix_forward(cache, p, [[t]], expand=False)[0, -1])
        p += 1
    return host.tok.decode(out)


def mode_eval(host, a):
    d = f"{a.out}/eval"
    os.makedirs(d, exist_ok=True)
    dists = [int(x) for x in a.dists.split(",")]
    t0 = time.time()
    for fam in a.families.split(","):
        dst = f"{d}/{fam}_{a.tag}{a.shard}.pt"
        out = torch.load(dst, weights_only=False) if os.path.exists(dst) else []
        done = {r["group"] for r in out}
        for g in range(a.gstart, a.gstart + a.groups):
            if g % a.nshards != a.shard or g in done:
                continue
            items = bench_group(fam, g, "eval")
            seed = items[0]["filler_seed"]
            fr = T.render_filler(seed, 2 * (P_PAST + W_WIN), 0)
            fz = torch.stack([host.clip_z(host.vis(fr[2 * CLIP * j:2 * CLIP * (j + 1)])) for j in range(NCL)])
            win = fr[2 * P_PAST:]
            tabs = [render_tab(it) for it in items]
            tvis = [host.vis(t) for t in tabs]
            patches = {D: patches_for(host, lambda k: fr[2 * CLIP * k:2 * CLIP * (k + 1)], tvis, D)
                       for D in dists}
            n = len(items)
            free = [GA.eval_free_question(it) for it in items]
            ctrl = {"window": mcq_letters(host, [(WIN_TXT, win)], items[0]["prompt"], len(items[0]["options"])),
                    "window_free": greedy(host, [(WIN_TXT, win)], free[0]["prompt"])}
            for v in range(n):
                o = (v + 1) % n
                ctrl[f"oracle{v}"] = mcq_letters(host, [(MEM_TXT, tabs[v]), (WIN_TXT, win)],
                                                 items[v]["prompt"], len(items[v]["options"]))
                ctrl[f"other{v}"] = mcq_letters(host, [(MEM_TXT, tabs[o]), (WIN_TXT, win)],
                                                items[v]["prompt"], len(items[v]["options"]))
                ctrl[f"oracle_free{v}"] = greedy(host, [(MEM_TXT, tabs[v]), (WIN_TXT, win)], free[v]["prompt"])
            out.append({"family": fam, "kind": GA.kind_of(fam), "group": g, "split": "eval",
                        "filler_seed": seed, "prompt": items[0]["prompt"], "options": items[0]["options"],
                        "answers": [it["answer"] for it in items], "facts": [it["facts"] for it in items],
                        "free": free, "dists": dists, "fill_z": fz, "patches": patches, "ctrl": ctrl})
            if len(out) % 10 == 0:
                torch.save(out, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
                print(f"  eval {fam} {len(out)} groups {time.time() - t0:.0f}s", flush=True)
        torch.save(out, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        print(f"{fam}: {len(out)} groups -> {dst}", flush=True)
    print(f"PREP-AB-EVAL-FINI {(time.time() - t0) / 60:.1f} min", flush=True)


def render_tab(it):
    return GA.render_tableau(it["tableau"], 2 * T.E)


@torch.no_grad()
def self_check(host):
    """prefix+suffix logits == plain forward (teacher path), clip_z rotating subsample sane."""
    it = GA.make_group_ab(10_000_000)[0]
    tab = render_tab(it)
    win = T.render_filler("ab-check", 2 * W_WIN, 0)
    q = GA.questions_for(it, random.Random(0))[0]
    ref, ids = host.full_logits([(MEM_TXT, tab), (WIN_TXT, win)], q["prompt"], q["answer"])
    pre, post = host.chat_prefix([(MEM_TXT, tab), (WIN_TXT, win)])
    qi, ai = host.suffix_ids(q["prompt"], post, q["answer"])
    allids = torch.cat([pre["input_ids"][0].cpu(), torch.tensor(qi + ai)])
    assert torch.equal(allids, ids.cpu()), "tokenisation of prefix+suffix != plain chat"
    cache, p0 = host.prefix_forward(pre)
    lg = host.head(host.suffix_forward(cache, p0, [qi + ai, qi + ai]))   # batch of 2, cache repeated
    L = pre["input_ids"].shape[1]
    r = ref[L:]
    err = float((lg[0] - r).abs().max() / r.abs().max())
    err2 = float((lg[1] - lg[0]).abs().max())
    agree = float((lg[0].argmax(-1) == r.argmax(-1)).float().mean())
    print(f"self-check prefix/suffix vs plain: max rel err {err:.2e}, batch rows diff {err2:.2e}, "
          f"argmax agree {agree:.3f}", flush=True)
    assert err < 5e-2 and agree > 0.95, (err, agree)
    q2 = dict(q)
    a0 = teacher_topk(host, [(MEM_TXT, tab), (WIN_TXT, win)], [q2])[0]
    a1 = teacher_topk(host, [(MEM_TXT, tab), (WIN_TXT, win)], [q2], vis=[host.vis(tab), host.vis(win)])[0]
    e = float((a0["top_lp"].float() - a1["top_lp"].float()).abs().max())
    print(f"self-check injected visual tokens in teacher: max |dlogp| top-k {e:.3f}, "
          f"same top-1 {bool((a0['top_ids'][:, 0] == a1['top_ids'][:, 0]).all())}", flush=True)
    assert e < 0.35, e          # bf16 noise of the vision tower (gold p within ~0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["basis", "fill", "train", "eval", "check"])
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--basis", default=f"{HERE}/cache_ab/qwen_l24_pca512_fpsfix.pt")
    ap.add_argument("--basis-groups", type=int, default=400, dest="basis_groups")
    ap.add_argument("--out", default=f"{HERE}/data_ab")
    ap.add_argument("--nfill", type=int, default=200)
    ap.add_argument("--fill-start", type=int, default=0, dest="fill_start")
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--gstart", type=int, default=None)
    ap.add_argument("--families", default=",".join(BENCH))
    ap.add_argument("--dists", default="0,120")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dmin", type=int, default=1, help="distractor calques per training stream, min")
    ap.add_argument("--dmax", type=int, default=4, help="distractor calques per training stream, max")
    a = ap.parse_args()
    if a.gstart is None:
        a.gstart = 10_000 if a.mode == "eval" else 0
    host = Host(None if a.mode in ("basis", "check") else a.basis)
    print(f"host layer {host.site}, gen {GA.GEN_MD5}, mode {a.mode}", flush=True)
    self_check(host)
    if a.mode == "check":
        return
    os.makedirs(a.out, exist_ok=True)
    json.dump({"mode": a.mode, "args": vars(a), "gen": GA.GEN_MD5, "v3c_dir": GA.V3C_DIR},
              open(f"{a.out}/meta_{a.mode}_{a.tag}{a.shard}.json", "w"), indent=1)
    {"basis": mode_basis, "fill": mode_fill, "train": mode_train, "eval": mode_eval}[a.mode](host, a)


if __name__ == "__main__":
    main()
