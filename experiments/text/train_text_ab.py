"""Text A/B: does a memory trained on GIST questions keep the gist and drop the detail?

Host: Qwen2.5-VL-3B (text only), frozen. Graft: MemoryBranch at layer 24 (titans snapshot in
pylib/, git 039826b). The past (P tokens) is encoded once to layer 24 WITHOUT the graft, projected by
a fixed text PCA basis, taken 1 token in `write_stride`, and written into a fresh memory WITH graph
(`write_stream`, as in train_mcq). The window + question + answer is then read through the graft with
writes off (the question is hidden at write time).

Arms (same joint v2 stories, fresh at every step, train split; eval on the salted eval split):
    A  questions on gist families only        (count_start, order, main_object)
    B  questions on all families              (gist above + colour, number, object_name)
    held-out gist family `place`: never trained, in both arms' eval (generalisation).
    C  no question: distillation on the text continuation (prepared, launch only on approval)
K questions per story (same K for A and B), both variants of the step's pair are used.

Loss for A/B, per question, on the free short answer tokens:
    CE(answer | window, question, own memory)
  + lam_kl * KL(teacher || student) on answer tokens, only for questions the teacher gets right
      (teacher = same host, graft off, the family's fact sentences visible in context)
  + lam_c * relu(margin + CE_own - CE_other)   (CE_other: same question, memory of the OTHER variant)
Arm C: sum_t w_t KL(full || student) / sum_t w_t on the last `--c-targets` tokens of the window,
    w_t = KL(full || window only) (only tokens that depend on the past count).

Eval: MCQ letter logits, contrastive by group over the 4 variants, own memory / other variant's
memory / window only (graft off), at D in --eval-dists; plus free-answer exact match (own memory).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=pylib python train_text_ab.py --arm A --steps 30 --out runs_ab/smoke_A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pylib"))
import gen_story2 as G  # noqa: E402
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args, add_memory_flags  # noqa: E402
from titans.graft.qwen import graft, find_decoder_layers, disabled  # noqa: E402
from titans.graft.train_load import write_stream, take, pr  # noqa: E402

LET = "ABCD"
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"
# presence is out of training: not certified in joint v2 (0 % once distractors are present) and the
# teacher gets it right only 45-70 % in free answer. It stays in the eval (uncertified, reported apart).
TRAIN_FAMS = {"A": ["count_start", "order", "main_object"],
              "B": ["count_start", "order", "main_object", "colour", "number", "object_name"]}
HELD_OUT = "place"


class Stop(Exception):
    pass


class TextHost:
    def __init__(self, model_id, device, site=24):
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.lid = [self.tok.encode(c, add_special_tokens=False)[0] for c in LET]
        self.site = site
        self.layers = find_decoder_layers(self.model)
        self.pad = self.tok.pad_token_id
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def t(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def layer_states(self, ids_batch):
        """[B, P] ids -> layer-`site` output [B, P, D] (fp32), stopping the forward there."""
        got = []

        def hook(_m, _i, out):
            got.append(out[0] if isinstance(out, tuple) else out)
            raise Stop
        h = self.layers[self.site].register_forward_hook(hook)
        try:
            self.model(input_ids=ids_batch, use_cache=False)
        except Stop:
            pass
        finally:
            h.remove()
        return got[0].float()


NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10"}


def norm_ans(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    w = [NUMW.get(x, x) for x in s.split() if x not in ("the", "a", "an", "at", "in", "on", "her", "his", "people")]
    return " ".join(w)


def ans_ok(gold, out):
    g, o = norm_ans(gold), norm_ans(out)
    return bool(g) and (g == o or f" {g} " in f" {o} ")


def fit_basis(host, k, n_stories, path):
    from titans.worldmodel.bottleneck import fit_pca
    xs = []
    for i in range(n_stories):
        st = G.make_story(5_000_000 + i, "train")
        L = G.layout(st, host.tok, P=4096, W=256, D=random.Random(i).randrange(0, 2000))
        ids = torch.tensor([L[0]["past"]], device=host.device)
        h = host.layer_states(ids)[0]
        idx = torch.randperm(h.shape[0])[:1024]
        xs.append(h[idx].cpu())
    X = torch.cat(xs)
    pca = fit_pca(X, k)
    torch.save({"pca": pca, "layer": host.site, "n": X.shape[0], "source": "gen_story2 train split"}, path)
    print(f"basis: {X.shape[0]} states, k={k}, explained {pca['explained']:.3f} -> {path}", flush=True)
    return pca


def build_qa(host, win, it, answer=True):
    """student ids: window + question (+ answer). returns ids, answer positions (slice into logits)."""
    q = host.t(SYS + "You are reading a long story in pieces.\n\nThe most recent part of the story:\n") + win + \
        host.t(f"\n\nQuestion: {it['question']}\nAnswer with a few words." + END)
    a = host.t(it["answer_text"]) if answer else []
    return q, a


def build_teacher(host, facts, win, it):
    q = host.t(SYS + "You are reading a long story in pieces.\n\nNotes kept from earlier parts of the story:\n") + facts + \
        host.t("\n\nThe most recent part of the story:\n") + win + \
        host.t(f"\n\nQuestion: {it['question']}\nAnswer with a few words." + END)
    return q, host.t(it["answer_text"])


def pad_batch(seqs, pad):
    m = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad] * (m - len(s)) for s in seqs])
    att = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs])
    return ids, att


def answer_logp(model, ids, att, qlens, alens, device):
    """log-softmax over vocab at the positions predicting the answer tokens -> list of [la, V]."""
    out = model(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False).logits
    res = []
    for i, (lq, la) in enumerate(zip(qlens, alens)):
        res.append(F.log_softmax(out[i, lq - 1:lq - 1 + la].float(), -1))
    return res


def set_carry(wrapper, state, S, rows):
    """batched read carry: row r of the read batch uses memory `rows[r]` of the write batch."""
    idx = torch.tensor(rows, device=next(iter(state.values())).device)
    wrapper.carry = ({k: v.index_select(0, idx) for k, v in state.items()},
                     {k: v.index_select(0, idx) for k, v in S.items()})
    wrapper.token_offset = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    add_memory_flags(ap, write_stride=8)
    ap.add_argument("--basis", default=os.path.join(HERE, "basis_l24_text_pca{k}.pt"))
    ap.add_argument("--past", type=int, default=8192)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--dmax", type=int, default=6000)
    ap.add_argument("--k-questions", type=int, default=3, dest="kq")
    ap.add_argument("--lam-kl", type=float, default=1.0, dest="lam_kl")
    ap.add_argument("--lam-c", type=float, default=1.0, dest="lam_c")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--c-targets", type=int, default=64, dest="c_targets")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    ap.add_argument("--eval-groups", type=int, default=30, dest="eval_groups")
    ap.add_argument("--eval-dists", default="3000,6000", dest="eval_dists")
    ap.add_argument("--final-eval-groups", type=int, default=100, dest="final_eval_groups")
    ap.add_argument("--mem-type", choices=["deep", "pool"], default="deep", dest="mem_type",
                    help="pool = trained slot memory at the same byte budget (SlotMemory, ~/work/iclr/baselines/bl.py)")
    ap.add_argument("--slots", type=int, default=0, help="pool: cases; 0 = same floats as the deep memory")
    ap.add_argument("--pool", type=int, default=0,
                    help="pool: writes averaged per case; 0 = all the past fits; 1 = raw FIFO of the last writes")
    ap.add_argument("--tau-init", type=float, default=10.0, dest="tau_init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    host = TextHost(a.model, a.device)
    tok = host.tok
    bpath = a.basis.format(k=a.d_mem)
    if os.path.exists(bpath):
        pca = torch.load(bpath, map_location="cpu", weights_only=False)["pca"]
    else:
        pca = fit_basis(host, a.d_mem, 64, bpath)
    branch = MemoryBranch(2048, a.d_mem, chunk_size=a.chunk, write_stride=a.write_stride, down_pca=pca,
                          **memory_kwargs_from_args(a)).to(a.device)
    n_writes = -(-a.past // a.write_stride)
    if a.mem_type == "pool":
        sys.path.insert(0, os.path.expanduser("~/work/iclr/baselines"))
        from bl import SlotMemory
        a.slots = a.slots or (2 * a.memory_hidden * a.d_mem) // a.d_mem
        a.pool = a.pool or math.ceil(n_writes / a.slots)
        branch.memory = SlotMemory(a.d_mem, a.slots, a.pool, a.tau_init).to(a.device)
        print(f"pool memory: {a.slots} cases x {a.d_mem} floats, P = {a.pool} (writes per stream {n_writes}; "
              f"the last {a.slots * a.pool} writes are kept)", flush=True)
    branch.train()
    wrapper = graft(host.model, branch, layer=host.site)
    # extinction monitors: |delta| / |h| at the graft site during reads
    mon = {"up": [], "h": []}
    branch.up.register_forward_hook(lambda m, i, o: mon["up"].append(float(o.detach().float().norm(dim=-1).mean())))
    wrapper.layer.register_forward_hook(lambda m, i, o: mon["h"].append(
        float((o[0] if isinstance(o, tuple) else o).detach().float().norm(dim=-1).mean())))
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    if a.mem_type == "pool":
        mem_numel = a.slots * a.d_mem
        st_init = None
    else:
        st0, _ = branch.init_carry(1)
        mem_numel = sum(v.numel() for v in st0.values())
    kv_tok = 36 * 2 * 2 * 128 * 2          # Qwen2.5-3B: layers x (K,V) x kv-heads x head-dim x bf16 bytes
    budget = {"state_numel": mem_numel, "state_kib_fp32": mem_numel * 4 / 1024,
              "kv_tokens_equiv_bf16": mem_numel * 2 / kv_tok, "hidden_tokens_equiv_bf16": mem_numel * 2 / (2048 * 2),
              "trainable_params": sum(p.numel() for p in params)}
    print("memory budget:", json.dumps(budget), flush=True)

    def write_mem(pasts):
        """list of past id lists -> (state, S) batched, with graph."""
        ids = torch.tensor(pasts, device=a.device)
        with disabled(wrapper):
            h = host.layer_states(ids)
        z = branch.down(h)[:, ::a.write_stride]
        if a.mem_type == "pool":
            return branch.memory.write(z)
        state, S, _, gates = write_stream(branch, z, a.chunk)
        return state, S, gates

    def lr_scale(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    @torch.no_grad()
    def evaluate(ngroups, tag):
        branch.eval()
        rows = []
        for D in [int(x) for x in a.eval_dists.split(",")]:
            for g in range(10_000, 10_000 + ngroups):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=a.past, W=a.window, D=D)
                state, S, _ = write_mem([l["past"] for l in lays])
                win = lays[0]["window"]
                for f in G.FAMILIES:
                    its = st["items"][f]
                    seqs = []
                    for it in its:
                        o = "\n".join(f"{LET[i]}. {t}" for i, t in enumerate(it["options"]))
                        seqs.append(host.t(SYS + "You are reading a long story in pieces.\n\nThe most recent part of the story:\n")
                                    + win + host.t(f"\n\nQuestion: {it['question']}\n{o}\nAnswer with the letter of the correct option." + END))
                    ids, att = pad_batch(seqs * 2, host.pad)
                    lens = [len(s) for s in seqs] * 2
                    set_carry(wrapper, state, S, [0, 1, 2, 3, 1, 2, 3, 0])     # own, then other (v+1)
                    out = host.model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), use_cache=False).logits
                    lg = torch.stack([out[i, lens[i] - 1, host.lid].float().cpu() for i in range(8)])
                    with disabled(wrapper):
                        ids4, att4 = pad_batch(seqs, host.pad)
                        o4 = host.model(input_ids=ids4.to(a.device), attention_mask=att4.to(a.device), use_cache=False).logits
                        lw = torch.stack([o4[i, len(seqs[i]) - 1, host.lid].float().cpu() for i in range(4)])
                    # free answer, own memory, greedy 8 tokens
                    fseqs = [build_qa(host, win, it, answer=False)[0] for it in its]
                    set_carry(wrapper, state, S, [0, 1, 2, 3])
                    gen = greedy(fseqs, 8)
                    for v, it in enumerate(its):
                        rows.append({"tag": tag, "D": D, "group": g, "family": f, "variant": v, "answer": it["answer"],
                                     "lg_mem": lg[v].tolist(), "lg_other": lg[4 + v].tolist(), "lg_win": lw[v].tolist(),
                                     "donor_answer": its[(v + 1) % 4]["answer"], "free": gen[v],
                                     "free_ok": int(ans_ok(it["answer_text"], gen[v]))})
        branch.train()
        return rows

    def greedy(seqs, n):
        cur = [list(s) for s in seqs]
        outs = [[] for _ in seqs]
        for _ in range(n):
            ids, att = pad_batch(cur, host.pad)
            lg = host.model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), use_cache=False).logits
            for i, s in enumerate(cur):
                t = int(lg[i, len(s) - 1].argmax())
                if t in (host.im_end, tok.eos_token_id) or (outs[i] and outs[i][-1] == -1):
                    outs[i].append(-1)
                    continue
                outs[i].append(t)
                s.append(t)
        return [tok.decode([t for t in o if t >= 0]) for o in outs]

    def summarize(rows):
        res = {}
        by = defaultdict(dict)
        for r in rows:
            by[(r["D"], r["family"], r["group"])][r["variant"]] = r
        acc = defaultdict(lambda: defaultdict(list))
        for (D, f, g), vs in by.items():
            if len(vs) < 4:
                continue
            for key in ("lg_mem", "lg_other", "lg_win"):
                L = torch.tensor([vs[v][key] for v in range(4)])
                C = L - L.mean(0, keepdim=True)
                acc[(D, f)][key].append(all(int(C[v].argmax()) == vs[v]["answer"] for v in range(4)))
            acc[(D, f)]["donor"].append(sum(int(torch.tensor(vs[v]["lg_other"]).argmax()) == vs[v]["donor_answer"]
                                            for v in range(4)) / 4)
            acc[(D, f)]["free"].append(sum(vs[v]["free_ok"] for v in range(4)) / 4)
        for (D, f), d in sorted(acc.items()):
            res[f"{f}@{D}"] = {k: round(100 * sum(v) / len(v), 1) for k, v in d.items()} | {"n": len(d["lg_mem"])}
        return res

    hist, log, t0 = [], [], time.time()
    kept = defaultdict(list)          # teacher kept (right) per family, for the KL mask
    fams = TRAIN_FAMS.get(a.arm)
    run = None
    for step in range(a.steps + 1):
        if a.eval_every and step % a.eval_every == 0 and (step > 0 or a.steps <= 100):
            ng = a.final_eval_groups if step == a.steps else a.eval_groups
            rows = evaluate(ng, step)
            s = summarize(rows)
            hist.append({"step": step, "eval": s})
            with open(a.out + f"_eval{step}.jsonl", "w") as fo:
                for r in rows:
                    fo.write(json.dumps(r) + "\n")
            print(f"  == eval {step} (contrastive group %: mem / other / window | donor | free)", flush=True)
            for k, v in s.items():
                print(f"     {k:22s} {v['lg_mem']:5.1f} / {v['lg_other']:5.1f} / {v['lg_win']:5.1f} | "
                      f"{v['donor']:5.1f} | {v['free']:5.1f}  n{v['n']}", flush=True)
        if step == a.steps:
            break
        for gr in opt.param_groups:
            gr["lr"] = a.lr * lr_scale(step)
        mon["up"].clear(); mon["h"].clear()
        g = 1_000_000 + a.seed * 100_000 + step
        st = G.make_story(g, "train")
        D = rng.randrange(0, a.dmax + 1)
        lays = G.layout(st, tok, P=a.past, W=a.window, D=D)
        v1, v2 = rng.sample(range(4), 2)
        win = lays[0]["window"]
        try:
            state, S, (al, et, th) = write_mem([lays[v1]["past"], lays[v2]["past"]])
            if a.arm in ("A", "B"):
                qf = rng.sample(fams, min(a.kq, len(fams)))
                stu, tea, meta = [], [], []
                for r, v in enumerate((v1, v2)):
                    for f in qf:
                        it = st["items"][f][v]
                        q, ans = build_qa(host, win, it)
                        facts = [x for s_, e_ in G.facts_of(lays[v], f) for x in lays[v]["past"][s_:e_]]
                        tq, tans = build_teacher(host, facts, win, it)
                        stu.append((q, ans, r))
                        tea.append((tq, tans))
                        meta.append(f)
                n = len(stu)
                # teacher (graft off, no grad): greedy free answer, judged on the normalized answer.
                # If right, its own answer tokens become the target (format of the host); else gold.
                with torch.no_grad(), disabled(wrapper):
                    m = max(len(q) for q, _ in tea)
                    x = torch.tensor([[host.pad] * (m - len(q)) + q for q, _ in tea], device=a.device)
                    am = torch.tensor([[0] * (m - len(q)) + [1] * len(q) for q, _ in tea], device=a.device)
                    gout = host.model.generate(input_ids=x, attention_mask=am, max_new_tokens=12, do_sample=False)
                    tok_ok, tgts = [], []
                    for i, (q, an) in enumerate(tea):
                        gt = [t for t in gout[i, m:].tolist() if t != host.pad]
                        for stop in (host.im_end, tok.eos_token_id):
                            if stop in gt:
                                gt = gt[:gt.index(stop)]
                        it_ = st["items"][meta[i]][(v1, v2)[stu[i][2]]]
                        ok = ans_ok(it_["answer_text"], tok.decode(gt)) and len(gt) > 0
                        tok_ok.append(ok)
                        tgts.append(gt if ok else an)
                    tea = [(q, t_) for (q, _), t_ in zip(tea, tgts)]
                    stu = [(q, t_, r) for (q, _, r), t_ in zip(stu, tgts)]
                    ids, att = pad_batch([q + an for q, an in tea], host.pad)
                    tlp = answer_logp(host.model, ids, att, [len(q) for q, _ in tea], [len(an) for _, an in tea], a.device)
                # student, own memory then other memory (rows 0..n-1 own, n..2n-1 other)
                seqs = [q + an for q, an, _ in stu]
                ids, att = pad_batch(seqs + seqs, host.pad)
                set_carry(wrapper, state, S, [r for _, _, r in stu] + [1 - r for _, _, r in stu])
                qlens = [len(q) for q, _, _ in stu] * 2
                alens = [len(an) for _, an, _ in stu] * 2
                slp = answer_logp(host.model, ids, att, qlens, alens, a.device)
                ce_own, ce_oth, kls = [], [], []
                for i, (q, an, r) in enumerate(stu):
                    tgt = torch.tensor(an, device=a.device)
                    ce_own.append(F.nll_loss(slp[i], tgt))
                    ce_oth.append(F.nll_loss(slp[n + i], tgt))
                    if tok_ok[i]:
                        kls.append((tlp[i].exp() * (tlp[i] - slp[i])).sum(-1).mean())
                ce_own, ce_oth = torch.stack(ce_own), torch.stack(ce_oth)
                l_ce = ce_own.mean()
                l_kl = torch.stack(kls).mean() if kls else torch.zeros((), device=a.device)
                l_c = F.relu(a.margin + ce_own - ce_oth).mean()
                loss = l_ce + a.lam_kl * l_kl + a.lam_c * l_c
                info = {"ce": float(l_ce), "kl": float(l_kl), "con": float(l_c), "gap": float((ce_oth - ce_own).mean()),
                        "teach_ok": sum(tok_ok) / n}
                for f_, ok_ in zip(meta, tok_ok):
                    kept[f_].append(int(ok_))
            else:
                # arm C: continuation distillation on the last c_targets tokens of the window
                T = a.c_targets
                full = [lays[v]["past"] + win for v in (v1, v2)]
                with torch.no_grad(), disabled(wrapper):
                    fo_ = host.model(input_ids=torch.tensor(full, device=a.device), use_cache=False,
                                     logits_to_keep=T + 1).logits[:, :-1].float()
                    lp_full = F.log_softmax(fo_, -1)
                    wo = host.model(input_ids=torch.tensor([win], device=a.device), use_cache=False,
                                    logits_to_keep=T + 1).logits[:, :-1].float()
                    lp_win = F.log_softmax(wo, -1)
                    w_t = (lp_full.exp() * (lp_full - lp_win)).sum(-1).clamp_min(0)       # [2, T]
                set_carry(wrapper, state, S, [0, 1])
                so = host.model(input_ids=torch.tensor([win, win], device=a.device), use_cache=False,
                                logits_to_keep=T + 1).logits[:, :-1].float()
                lp_s = F.log_softmax(so, -1)
                kl_t = (lp_full.exp() * (lp_full - lp_s)).sum(-1)                           # [2, T]
                loss = (w_t * kl_t).sum() / w_t.sum().clamp_min(1e-6)
                kl_win = (lp_full.exp() * (lp_full - lp_win)).sum(-1)
                gain = float(((kl_win - kl_t.detach()) * w_t).sum() / w_t.sum().clamp_min(1e-6))
                info = {"wkl": float(loss), "wkl_window": float((w_t * kl_win).sum() / w_t.sum().clamp_min(1e-6)),
                        "mem_gain": gain, "w_mean": float(w_t.mean()), "w_max": float(w_t.max())}
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  OOM at step {step}", flush=True)
            continue
        if not torch.isfinite(loss):
            print(f"  NON-FINITE loss at step {step}: STOP", flush=True)
            break
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
        opt.step()
        run = float(loss) if run is None else 0.95 * run + 0.05 * float(loss)
        if step % 10 == 0:
            if a.mem_type == "deep":
                s0, _ = branch.init_carry(1)
                wn = sum(float((state[k_][0].detach() - s0[k_][0]).norm()) for k_ in s0) / \
                    max(1e-9, sum(float(s0[k_][0].norm()) for k_ in s0))
            else:
                wn = float(state["slots"][0].norm(dim=-1).mean())
            dr = (sum(mon["up"]) / max(1, len(mon["up"]))) / max(1e-9, sum(mon["h"]) / max(1, len(mon["h"])))
            gl = {"step": step, "D": D, "loss": float(loss), "loss_run": run, "gn": gn, **info,
                  "write_rel_norm": wn, "delta_over_h": dr,
                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                  "kept": {f_: round(sum(v_) / len(v_), 3) for f_, v_ in kept.items()},
                  "eta_med": float(et.median()), "alpha_med": float(al.median()), "theta_med": float(th.median()),
                  "pr_w1": pr(state["w1"][0].detach()) if "w1" in state else None,
                  "s_per_step": (time.time() - t0) / (step + 1)}
            log.append(gl)
            print("  " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in gl.items() if k != "kept"), flush=True)
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a), "budget": budget},
                       a.out + ".pt")
        if step % 50 == 0 or step + 1 == a.steps:
            with open(a.out + ".json", "w") as fo:
                json.dump({"args": vars(a), "budget": budget, "history": hist, "log": log,
                           "elapsed_s": time.time() - t0}, fo, indent=1)
    with open(a.out + ".json", "w") as fo:
        json.dump({"args": vars(a), "budget": budget, "history": hist, "log": log, "elapsed_s": time.time() - t0},
                  fo, indent=1)
    print("TRAIN-TEXT-AB-FINI", flush=True)


if __name__ == "__main__":
    main()
