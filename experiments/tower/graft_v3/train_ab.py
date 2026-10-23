"""23/10 : A/B graft training -- arm A "gist only" vs arm B "everything" (new file; train_v3 untouched).

Data: prep_ab.py (ladder3 layout; rich training tableaux; dense free-form questions; teacher
top-k log-probs). Student = frozen Qwen2.5-VL-3B on the WINDOW only ("The most recent seconds of
the video stream:" + 4 s window + question), plus the grafted memory written with the 160 s past
(own variant, question hidden at write time: the memory only ever sees the past clips).

Loss per question (answer tokens + final <|im_end|>, teacher-forced):
    KL(teacher || student) on the teacher's top-k (+ one "rest" bucket)    [lam_kl]
  + CE(gold answer tokens | own memory)                                   [lam_ce]
  + relu(margin + CE(gold | own) - CE(gold | OTHER-variant memory))       [lam_con]
    (other variant = same group: same filler, window, room; only the tableau differs; the term
     is used only when the gold answer is false for the other variant)
Per item:  L = mean over its gist questions  [+ mean over its detail questions, arm B]
Arm A samples --q-gist questions (one per gist family); arm B samples the SAME gist questions
plus --q-detail detail questions, with the two means weighted 1:1, so the gist gradient of B
equals A's in expectation and B differs only by having to ALSO keep the details.

Speed: per item, one prefix forward (window, graft reading the memory) with KV cache, then the
batch of question suffixes on the repeated cache; memories of the batch written in one pass.

Eval (bench v3, prep_ab eval split, ladder3 streams, D in --eval-dists): per item own / other
variant (+ donor rate) letter logits, contrastive per group (ladder3 method), free-form greedy
exact match (own); controls window / oracle / other-tableau from the prep (graft independent).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/work/titans-mac/src python train_ab.py --arm A --steps 20 \\
        --out runs_ab/smoke_A
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_ab as GA                  # noqa: E402
import prep_ab as PA                 # noqa: E402

from titans.graft.branch import MemoryBranch, memory_kwargs_from_args   # noqa: E402
from titans.graft.qwen import graft                                     # noqa: E402
from titans.graft.train_load import pr, take, write_stream              # noqa: E402
from titans.graft.write_ckpt import write_stream_ckpt                   # noqa: E402

LET = "ABCD"


def expand(p, B):
    return {k: v.expand(B, *v.shape[1:]) for k, v in p.items()}


def load_list(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        out += torch.load(f, map_location="cpu", weights_only=False)
    return out


class Data:
    """Memory input. A stored clip is [8 groups x 32 tokens]; group g holds raster positions
    g%8 + 8j (prep_ab). --mem-stride k keeps, in group g, the tokens j = g%k, g%k + k, ...: the
    phase rotates with g, so a static tableau is still seen at many positions."""

    def __init__(self, root, mem_stride):
        self.root, self.ms, self.fill = root, mem_stride, {}
        k = mem_stride
        self.idx = torch.cat([g * 32 + torch.arange(g % k, 32, k) for g in range(PA.CLIP)])

    def sub(self, x):
        return x[self.idx] if self.ms > 1 else x

    def fill_z(self, i):
        if i not in self.fill:
            self.fill[i] = torch.load(f"{self.root}/fill/fill_{i}.pt", map_location="cpu")["z"]
        return self.fill[i]

    def past_train(self, rec, v, device):
        z = list(self.fill_z(rec["fill"])[rec["j0"]:rec["j0"] + PA.NCL])
        for k, zk in rec.get("shared", {}).items():         # distractor calques (all variants)
            z[k] = zk
        for k, zk in rec["patches"][v].items():
            z[k] = zk
        return torch.cat([self.sub(x) for x in z], 0).float().to(device)[None]

    def past_eval(self, rec, D, v, device):
        z = list(rec["fill_z"])
        for k, zk in rec["patches"][D][v].items():
            z[k] = zk
        return torch.cat([self.sub(x) for x in z], 0).float().to(device)[None]


def write(branch, z, a):
    if a.mem_type == "pool":                     # trained slot memory (baselines/bl.py SlotMemory)
        return branch.memory.write(z)
    if a.ckpt_seg:
        st, S, _, g3 = write_stream_ckpt(branch, z, a.chunk, a.ckpt_seg)
    else:
        st, S, _, g3 = write_stream(branch, z, a.chunk)
    return st, S, g3


_WIN = {}


def window_prefix(host, frames_fn, key):
    if key not in _WIN:
        if len(_WIN) >= 8:
            _WIN.pop(next(iter(_WIN)))
        _WIN[key] = host.chat_prefix([(PA.WIN_TXT, frames_fn())])
    return _WIN[key]


def read_suffix(host, wrapper, pre, carry, seqs, grad, batch_prefix=1):
    """Graft reads ``carry`` (batch 1 or batch_prefix) during the prefix and the suffixes."""
    wrapper.carry, wrapper.token_offset = carry, 0
    cache, p0 = host.prefix_forward(pre, batch=batch_prefix, grad=grad)
    B = len(seqs)
    if batch_prefix == 1 and B > 1:
        wrapper.carry = (expand(carry[0], B), expand(carry[1], B))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return host.suffix_forward(cache, p0, seqs, expand=batch_prefix == 1)


LID = None


def q_losses(host, hs, spans, qs, teacher=True, kl_ok_only=True, mcq_rows=None):
    """Per question: (kl, ce, kl_mcq) on the answer positions.
    kl_ok_only: the free-form KL is used only where the teacher's own argmax is the gold answer
    (CE on the gold everywhere). mcq_rows[i]: suffix row of question i's lettered version: KL of
    the student's letter distribution to the teacher's (the teacher is ~bench-ceiling there)."""
    out = []
    for i, (pp, ai) in enumerate(spans):
        lq = F.log_softmax(host.head(hs[i, pp]), -1)                     # [La, V]
        gold = torch.tensor(ai, device=lq.device)
        ce = -lq.gather(1, gold[:, None])[:, 0].mean()
        kl = torch.zeros((), device=lq.device)
        klm = torch.zeros((), device=lq.device)
        t = qs[i].get("teacher")
        if teacher and mcq_rows is not None and mcq_rows.get(i) is not None and "mcq_lp" in t:
            r, L = mcq_rows[i]
            n = len(qs[i]["mcq"]["options"])
            ls = F.log_softmax(host.head(hs[r, L - 1])[LID[:n]], -1)
            lt = t["mcq_lp"].float().to(ls.device)
            klm = (lt.exp() * (lt - ls)).sum()
        if teacher and (not kl_ok_only or t["argmax_ok"]):
            ids = t["top_ids"].long().to(lq.device)
            lpt = t["top_lp"].float().to(lq.device)
            pt = lpt.exp()
            lqk = lq.gather(1, ids)
            p_rest = (1 - pt.sum(1)).clamp_min(1e-6)
            q_rest = (1 - lqk.exp().sum(1)).clamp_min(1e-6)
            kl = ((pt * (lpt - lqk)).sum(1) + p_rest * (p_rest.log() - q_rest.log())).mean()
        out.append((kl, ce, klm))
    return out


def sample_questions(rng, qs, arm, a):
    """One question per gist family (up to --q-gist), + detail ones for arm B."""
    def pick(kind, n):
        by = defaultdict(list)
        for q in qs:
            if q["kind"] == kind:
                by[q["family"]].append(q)
        fams = list(by)
        rng.shuffle(fams)
        return [rng.choice(by[f]) for f in fams[:n]]
    g = pick("gist", a.q_gist)
    d = pick("detail", a.q_detail) if arm == "B" else []
    return g, d


# ---------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(host, branch, wrapper, data, ev, a, free=True):
    branch.eval()
    rows = []
    lid_all = [host.tok.encode(c, add_special_tokens=False)[0] for c in LET]
    for rec in ev:
        n = len(rec["answers"])
        nopt = len(rec["options"])
        lid = lid_all[:nopt]
        pre, post = window_prefix(host, lambda: PA.T.render_filler(rec["filler_seed"], 2 * PA.W_WIN, 2 * PA.P_PAST),
                                  ("ev", rec["filler_seed"]))
        qi, _ = host.suffix_ids(rec["prompt"], post)
        fq, _ = host.suffix_ids(rec["free"][0]["prompt"], post)
        for D in a.eval_dists:
            if D not in rec["patches"]:
                continue
            st, S, _ = write(branch, torch.cat([data.past_eval(rec, D, v, a.device) for v in range(n)]), a)
            perm = [(v + 1) % n for v in range(n)]
            own = host.head(read_suffix(host, wrapper, pre, (st, S), [qi] * n, False, batch_prefix=n)[:, -1])[:, lid]
            sto = {k: t[perm] for k, t in st.items()}
            So = {k: t[perm] for k, t in S.items()}
            oth = host.head(read_suffix(host, wrapper, pre, (sto, So), [qi] * n, False, batch_prefix=n)[:, -1])[:, lid]
            preds = [""] * n
            if free:
                wrapper.carry, wrapper.token_offset = (st, S), 0
                cache, p0 = host.prefix_forward(pre, batch=n)
                h = host.suffix_forward(cache, p0, [fq] * n, expand=False)
                toks = [[] for _ in range(n)]
                done = [False] * n
                p = p0 + len(fq)
                for _ in range(6):
                    nxt = host.head(h[:, -1]).argmax(-1).tolist()
                    for v in range(n):
                        if not done[v]:
                            if nxt[v] == host.im_end:
                                done[v] = True
                            else:
                                toks[v].append(nxt[v])
                    if all(done):
                        break
                    h = host.suffix_forward(cache, p, [[t] for t in nxt], expand=False)
                    p += 1
                preds = [host.tok.decode(t) for t in toks]
            c = rec["ctrl"]
            for v in range(n):
                gold = rec["answers"][v]
                fr = rec["free"][v]
                rows.append({"family": rec["family"], "kind": rec["kind"], "group": rec["group"],
                             "variant": v, "D": D, "gold": gold, "donor": rec["answers"][perm[v]],
                             "lg_mem": [round(x, 3) for x in own[v].tolist()],
                             "lg_other": [round(x, 3) for x in oth[v].tolist()],
                             "lg_oracle": [round(x, 3) for x in c[f"oracle{v}"].tolist()],
                             "lg_window": [round(x, 3) for x in c["window"].tolist()],
                             "lg_otab": [round(x, 3) for x in c[f"other{v}"].tolist()],
                             "free_mem": preds[v], "free_gold": fr["answer"],
                             "free_oracle": c[f"oracle_free{v}"], "free_window": c["window_free"]})
    branch.train()
    return rows


def summarise(rows):
    """Per family / kind / D: item %, group %, contrastive group % (ladder3), donor %, free exact %."""
    conds = ["mem", "other", "oracle", "window", "otab"]

    def cell(sel):
        out = {"n_items": len(sel)}
        by = defaultdict(list)
        for r in sel:
            by[(r["family"], r["group"], r["D"])].append(r)
        groups = [rs for rs in by.values() if len(rs) == 4]
        out["n_groups"] = len(groups)
        for c in conds:
            it, gr, cg = [], [], []
            for rs in groups:
                L = np.array([r[f"lg_{c}"] for r in rs])
                ok = [int(np.argmax(L[i]) == rs[i]["gold"]) for i in range(len(rs))]
                Lc = L - L.mean(0, keepdims=True)
                okc = [int(np.argmax(Lc[i]) == rs[i]["gold"]) for i in range(len(rs))]
                it += ok
                gr.append(all(ok))
                cg.append(all(okc))
            if groups:
                out[c] = 100 * np.mean(it)
                out[c + "_grp"] = 100 * np.mean(gr)
                out[c + "_cgrp"] = 100 * np.mean(cg)
                out[c + "_cgrp_se"] = 100 * np.std(cg) / max(1, math.sqrt(len(cg)))
        if groups:
            out["donor"] = 100 * np.mean([int(np.argmax(r["lg_other"]) == r["donor"]) for rs in groups for r in rs])
            for c in ("mem", "oracle", "window"):
                out[f"free_{c}"] = 100 * np.mean([GA.norm_answer(r[f"free_{c}"]) == GA.norm_answer(r["free_gold"])
                                                  for rs in groups for r in rs])
        return out
    res = {"all": cell(rows)}
    for key in ("kind", "family", "D"):
        for val in sorted({r[key] for r in rows}, key=str):
            res[f"{key}={val}"] = cell([r for r in rows if r[key] == val])
    for fam in sorted({r["family"] for r in rows}):
        for D in sorted({r["D"] for r in rows}):
            res[f"{fam}@{D}"] = cell([r for r in rows if r["family"] == fam and r["D"] == D])
    return res


def show(tag, res):
    for k, s in res.items():
        if "mem" not in s:
            continue
        print(f"  [{tag}] {k:20s} g {s['n_groups']:3d} | own {s['mem']:5.1f} grp {s['mem_grp']:5.1f} "
              f"CGRP {s['mem_cgrp']:5.1f}±{s['mem_cgrp_se']:.1f} | other {s['other']:5.1f} cgrp {s['other_cgrp']:5.1f}"
              f" donor {s['donor']:5.1f} | window cgrp {s['window_cgrp']:5.1f} | oracle cgrp {s['oracle_cgrp']:5.1f}"
              f" | free own {s['free_mem']:5.1f} oracle {s['free_oracle']:5.1f} window {s['free_window']:5.1f}",
              flush=True)


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--data", default=f"{HERE}/data_ab")
    ap.add_argument("--train", default="train/*.pt")
    ap.add_argument("--eval", default="eval/*.pt")
    ap.add_argument("--eval-families", default=",".join(PA.BENCH), dest="eval_families")
    ap.add_argument("--basis", default=f"{HERE}/cache_ab/qwen_l24_pca512_fpsfix.pt")
    ap.add_argument("--q-gist", type=int, default=4, dest="q_gist")
    ap.add_argument("--q-detail", type=int, default=4, dest="q_detail")
    ap.add_argument("--lam-kl", type=float, default=1.0, dest="lam_kl")
    ap.add_argument("--lam-ce", type=float, default=1.0, dest="lam_ce")
    ap.add_argument("--lam-con", type=float, default=1.0, dest="lam_con")
    ap.add_argument("--lam-mcq", type=float, default=0.0, dest="lam_mcq",
                    help="KL to the teacher's letter distribution on the lettered version of each question")
    ap.add_argument("--kl-all", action="store_false", dest="kl_ok_only",
                    help="free-form KL on every question (default: only where the teacher's argmax is right)")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--mem-stride", type=int, default=4, dest="mem_stride")
    ap.add_argument("--ckpt-seg", type=int, default=64, dest="ckpt_seg")
    ap.add_argument("--d-mem", type=int, default=512, dest="d_mem")
    ap.add_argument("--memory-hidden", type=int, default=256, dest="memory_hidden",
                    help="memory MLP width; size = 2*d_mem*hidden fp32 (512x256 = 1 MiB)")
    ap.add_argument("--mem-type", choices=["deep", "pool"], default="deep", dest="mem_type",
                    help="pool = trained witness at the same byte budget: SLOTS cases of d_mem floats, "
                         "each the mean of P consecutive writes, FIFO on the cases, softmax read "
                         "(SlotMemory of ~/work/iclr/baselines/bl.py, unchanged)")
    ap.add_argument("--slots", type=int, default=0,
                    help="pool: number of cases; 0 = same bytes as the deep memory (2*hidden*d_mem/d_mem)")
    ap.add_argument("--pool", type=int, default=0,
                    help="pool: writes averaged per case; 0 = ceil(writes per stream / slots), i.e. the "
                         "WHOLE past fits; smaller P keeps only the last slots*P writes")
    ap.add_argument("--tau-init", type=float, default=10.0, dest="tau_init")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--alpha-init", type=float, default=1.27e-4, dest="alpha_init")
    ap.add_argument("--muon", default="unit")
    ap.add_argument("--init", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    ap.add_argument("--eval-groups", type=int, default=20, dest="eval_groups", help="per family, periodic eval")
    ap.add_argument("--eval-dists", default="120", dest="eval_dists")
    ap.add_argument("--final-eval-groups", type=int, default=150, dest="final_eval_groups")
    ap.add_argument("--final-eval-dists", default="0,120", dest="final_eval_dists")
    ap.add_argument("--no-free", action="store_true", dest="no_free")
    ap.add_argument("--eval-only", action="store_true", dest="eval_only")
    ap.add_argument("--early-frac", type=float, default=0.4, dest="early_frac",
                    help="from this fraction of the run, stop if CE flat and eval dead")
    ap.add_argument("--save-every", type=int, default=500, dest="save_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    data = Data(a.data, a.mem_stride)
    tr = [] if a.eval_only else load_list(f"{a.data}/{a.train}")
    fams = set(a.eval_families.split(","))
    ev_all = [r for r in load_list(f"{a.data}/{a.eval}") if r["family"] in fams]

    def sub(ev, n):
        seen = defaultdict(int)
        out = []
        for r in sorted(ev, key=lambda r: (r["family"], r["group"])):
            seen[r["family"]] += 1
            if seen[r["family"]] <= n:
                out.append(r)
        return out
    ev = sub(ev_all, a.eval_groups)
    print(f"arm {a.arm}: train {len(tr)} groups, eval {len(ev)} / {len(ev_all)} groups "
          f"({sorted({r['family'] for r in ev_all})})", flush=True)

    host = PA.Host(None, a.device)
    global LID
    LID = [host.tok.encode(c, add_special_tokens=False)[0] for c in LET]
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    if a.init:
        blob = torch.load(a.init, map_location="cpu", weights_only=False)
        for k in ("d_mem", "memory_hidden", "alpha_init", "muon", "mem_type", "slots", "pool", "tau_init"):
            setattr(a, k, blob["args"][k])
    branch = MemoryBranch(int(host.model.config.text_config.hidden_size), a.d_mem, chunk_size=a.chunk,
                          write_stride=1, down_pca=pca, **memory_kwargs_from_args(a)).to(a.device)
    if a.mem_type == "pool":
        sys.path.insert(0, os.path.expanduser("~/work/iclr/baselines"))
        from bl import SlotMemory
        n_writes = PA.NCL * 256 // a.mem_stride
        a.slots = a.slots or (2 * a.memory_hidden * a.d_mem) // a.d_mem
        a.pool = a.pool or math.ceil(n_writes / a.slots)
        branch.memory = SlotMemory(a.d_mem, a.slots, a.pool, a.tau_init).to(a.device)
        print(f"pool memory: {a.slots} cases x {a.d_mem} = {a.slots * a.d_mem * 4 / 2 ** 20:.2f} MiB fp32, "
              f"P = {a.pool} (writes per stream {n_writes}: the last {a.slots * a.pool} writes are kept)", flush=True)
    if a.init:
        branch.load_state_dict(blob["branch"])
    branch.train()
    wrapper = graft(host.model, branch, layer=host.site)
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    with torch.no_grad():
        st0 = write(branch, torch.zeros(1, PA.NCL * 256 // a.mem_stride, a.d_mem, device=a.device), a)[0]
    print(f"branch trainable {sum(p.numel() for p in params)} params; memory content after a stream "
          f"{ {k: tuple(v.shape) for k, v in st0.items()} } "
          f"= {sum(v[0].numel() for v in st0.values()) * 4 / 2 ** 20:.2f} MiB fp32", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    a.eval_dists = [int(x) for x in a.eval_dists.split(",")]

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    hist, glog, t0 = [], [], time.time()

    def report(step, evset, tag, free=True):
        te = time.time()
        rows = evaluate(host, branch, wrapper, data, evset, a, free=free)
        res = summarise(rows)
        hist.append({"step": step, "tag": tag, **res})
        show(f"{tag} {step}", res)
        print(f"  ({tag} {len(rows)} items, {time.time() - te:.0f} s)", flush=True)
        with open(f"{a.out}_{tag}{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return res

    with torch.no_grad():                         # batched write == one write per stream
        r0 = ev[0]
        D0 = [d for d in a.eval_dists if d in r0["patches"]][0]
        zz = torch.cat([data.past_eval(r0, D0, v, a.device) for v in range(2)])
        sb, _, _ = write(branch, zz, a)
        s1, _, _ = write(branch, zz[1:2], a)
        err = max(float((sb[k][1:2] - s1[k]).abs().max() / (s1[k].abs().max() + 1e-12)) for k in s1)
        print(f"check batched write vs single: max rel err {err:.1e}; writes per stream {zz.shape[1]}", flush=True)
        assert err < 1e-3, err
    if a.eval_only:
        a.eval_dists = [int(x) for x in a.final_eval_dists.split(",")]
        report(0, sub(ev_all, a.final_eval_groups), "final", free=not a.no_free)
        json.dump({"args": vars(a), "history": hist}, open(a.out + "_evalonly.json", "w"), indent=1)
        return

    run, tstep, ce0 = {}, [], None
    tm = defaultdict(float)

    def tick(name, t_):
        torch.cuda.synchronize()
        tm[name] += time.time() - t_
        if os.environ.get("AB_MEM"):
            print(f"    mem after {name}: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB "
                  f"(peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f})", flush=True)
        return time.time()

    for step in range(a.steps):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        bs = rng.sample(tr, a.batch)
        try:
            opt.zero_grad(set_to_none=True)
            tot = defaultdict(float)
            t_ = time.time()
            pick = []
            for rec in bs:
                v = rng.randrange(4)
                o = (v + rng.randrange(1, 4)) % 4
                pick.append((v, o))
            zv = torch.cat([data.past_train(r, v, a.device) for r, (v, _) in zip(bs, pick)])
            zo = torch.cat([data.past_train(r, o, a.device) for r, (_, o) in zip(bs, pick)])
            t_ = tick("data", t_)
            st, S, gates = write(branch, zv, a)
            t_ = tick("write_own", t_)
            pr_w = pr(st["w1"][0].detach()) if "w1" in st else float("nan")
            if a.lam_con > 0:
                with torch.no_grad():
                    so, So, _ = write(branch, zo, a)
                t_ = tick("write_other", t_)
            std = {k: x.detach().requires_grad_() for k, x in st.items()}
            Sd = {k: x.detach() for k, x in S.items()}
            for i, (rec, (v, o)) in enumerate(zip(bs, pick)):
                pre, post = window_prefix(host, lambda: PA.window_frames_train(rec), ("tr", rec["fill"], rec["j0"]))
                qg, qd = sample_questions(rng, rec["questions"][v], a.arm, a)
                qs = qg + qd
                seqs, spans = [], []
                for q in qs:
                    qi, ai = host.suffix_ids(q["prompt"], post, q["answer"])
                    seqs.append(qi + ai)
                    spans.append((PA.answer_positions(qi, ai), ai))
                mrows = {}
                if a.lam_mcq > 0:
                    for j, q in enumerate(qs):
                        if q.get("mcq"):
                            mi = host.suffix_ids(q["mcq"]["prompt"], post)[0]
                            mrows[j] = (len(seqs), len(mi))
                            seqs.append(mi)
                t_ = tick("data", t_)
                hs = read_suffix(host, wrapper, pre, (take(std, i), take(Sd, i)), seqs, True)
                lo = q_losses(host, hs, spans, qs, kl_ok_only=a.kl_ok_only, mcq_rows=mrows)
                for j in range(len(qs)):
                    tot["kl_kept"] += float(qs[j]["teacher"]["argmax_ok"] or not a.kl_ok_only) / len(qs) / len(bs)
                t_ = tick("student_fwd", t_)
                con = [torch.zeros((), device=a.device)] * len(qs)
                if a.lam_con > 0:
                    use = [j for j, q in enumerate(qs) if GA.answer(q, rec["facts"][o]) != q["answer"]]
                    if use:
                        hso = read_suffix(host, wrapper, pre, (take(so, i), take(So, i)),
                                          [seqs[j] for j in use], True)    # free-form rows only
                        lo_o = q_losses(host, hso, [spans[j] for j in use], [qs[j] for j in use], teacher=False)
                        for jj, j in enumerate(use):
                            con[j] = F.relu(a.margin + lo[j][1] - lo_o[jj][1])
                            tot["gap"] += float(lo_o[jj][1] - lo[j][1]) / len(use)
                    t_ = tick("student_fwd", t_)
                parts = []
                for kind, sel in (("gist", range(len(qg))), ("detail", range(len(qg), len(qs)))):
                    sel = list(sel)
                    if not sel:
                        continue
                    lk = torch.stack([a.lam_kl * lo[j][0] + a.lam_ce * lo[j][1] + a.lam_mcq * lo[j][2]
                                      + a.lam_con * con[j] for j in sel]).mean()
                    parts.append(lk)
                    tot[f"ce_{kind}"] += float(torch.stack([lo[j][1] for j in sel]).mean()) / len(bs)
                    tot[f"kl_{kind}"] += float(torch.stack([lo[j][0] for j in sel]).mean()) / len(bs)
                    tot[f"con_{kind}"] += float(torch.stack([con[j] for j in sel]).mean()) / len(bs)
                (sum(parts) / len(bs)).backward()
                t_ = tick("student_bwd", t_)
            ks = [k for k in st if std[k].grad is not None and st[k].requires_grad]
            if ks:                               # pool: the cases are means of z, no write parameters
                torch.autograd.backward([st[k] for k in ks], [std[k].grad for k in ks])
            t_ = tick("write_bwd", t_)
        except torch.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True)
            print(f"  OOM at step {step}: peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB; "
                  f"phases so far {dict(tm)}; {str(e)[:160]}", flush=True)
            torch.cuda.empty_cache()
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
        opt.step()
        tstep.append(time.time() - ts)
        tot["gap"] /= len(bs)
        for k, val in tot.items():
            run[k] = val if k not in run else 0.95 * run[k] + 0.05 * val
        if ce0 is None and step >= 20:
            ce0 = run["ce_gist"]
        if step % 5 == 0 or step + 1 == a.steps:
            al, et, th = gates
            gl = {"step": step, "eta_med": float(et.median()), "alpha_med": float(al.median()),
                  "theta_med": float(th.median()), "pr_w1": pr_w, "s_step": tstep[-1], "gnorm": gn,
                  "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30, **run}
            glog.append(gl)
            print(f"  {step:5d} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(run.items()))
                  + f" |g| {gn:.2f} eta {gl['eta_med']:.3f} alpha {gl['alpha_med']:.1e} theta {gl['theta_med']:.3f}"
                  f" pr {pr_w:.1f} {tstep[-1]:.1f} s/step (mean {np.mean(tstep):.1f}) peak {gl['peak_gib']:.1f} GiB"
                  " | " + " ".join(f"{k} {v / (step + 1):.1f}s" for k, v in tm.items()), flush=True)
        last = step + 1 == a.steps
        if a.save_every and ((step + 1) % a.save_every == 0 or last):
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a)}, f"{a.out}_at{step + 1}.pt")
        if a.eval_every and ((step + 1) % a.eval_every == 0) and not last:
            res = report(step + 1, ev, "eval", free=not a.no_free)
            if (step + 1) >= a.early_frac * a.steps and ce0 is not None:
                flat = run["ce_gist"] > 0.95 * ce0
                dead = res["all"].get("mem_cgrp", 0) < 5 and run.get("gap", 0) < 0.05
                if flat and dead:
                    print(f"EARLY-STOP at {step + 1}: ce_gist {run['ce_gist']:.3f} vs {ce0:.3f} (flat), "
                          f"own cgrp {res['all'].get('mem_cgrp', 0):.1f}, gap {run.get('gap', 0):.3f}", flush=True)
                    break
        if step % 50 == 0 or last:
            json.dump({"args": vars(a), "history": hist, "gates": glog, "s_per_step": float(np.mean(tstep)),
                       "phase_s_per_step": {k: v / (step + 1) for k, v in tm.items()},
                       "elapsed_s": time.time() - t0}, open(a.out + ".json", "w"), indent=1)
    a.eval_dists = [int(x) for x in a.final_eval_dists.split(",")]
    report(step + 1, sub(ev_all, a.final_eval_groups), "final", free=not a.no_free)
    json.dump({"args": vars(a), "history": hist, "gates": glog, "s_per_step": float(np.mean(tstep)) if tstep else None,
               "phase_s_per_step": {k: v / (step + 1) for k, v in tm.items()},
               "elapsed_s": time.time() - t0}, open(a.out + ".json", "w"), indent=1)
    print(f"TRAIN-AB-FINI mean {np.mean(tstep) if tstep else float('nan'):.2f} s/step", flush=True)


if __name__ == "__main__":
    main()
