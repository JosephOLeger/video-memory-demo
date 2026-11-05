"""train_bl.py (26/10): train_text_ab.py adapted to BABILong-like data (TEXT_GEN=gen_babilong, default).

Changes against train_text_ab.py (everything else identical, same flags, same checkpoints):
  * prompts: the OFFICIAL BABILong prompt of each task (instruction + examples + post_prompt, context =
    the window for the student, notes + window for the teacher), via G.student_ids / G.teacher_ids;
  * targets: the teacher's own answer when it is right (official metric, G.judge), else the official
    answer format (item['target_text']);
  * eval on the dev split: option likelihood over the 4 full answer sentences (no letter MCQ), free answer
    judged with the official metric; families = the stories' own tasks;
  * --oracle-support: oracle write = end-of-sentence states of the SUPPORTING facts (kvfull oracle);
  * --official TASKS:LENGTHS: evaluation on the OFFICIAL BABILong test files (bl_data/babilong_test),
    official prompt and metric, streaming write of the context minus its last W tokens, conditions
    window only / own memory / memory of ANOTHER sample / full context (when it fits).

Original docstring of train_text_ab.py follows.

Text A/B: does a memory trained on GIST questions keep the gist and drop the detail?

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
import importlib  # noqa: E402
G = importlib.import_module(os.environ.get("TEXT_GEN", "gen_babilong"))  # gen_story3: cumulative families
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args, add_memory_flags  # noqa: E402
from titans.graft.qwen import graft, find_decoder_layers, disabled  # noqa: E402
from titans.graft.train_load import write_stream as _write_stream0, take, pr  # noqa: E402

FLUSH = {"n": 0, "auto": False, "log": [], "alpha": "zero"}


def write_stream(branch, z, chunk, *args, **kw):
    """write_stream, then optionally FLUSH the momentum: n extra writes with a zero gradient and the
    story's median gates, so what is still in S at the end of the stream lands in W before the read."""
    out = _write_stream0(branch, z, chunk, *args, **kw)
    if not (FLUSH["n"] or FLUSH["auto"]) or not hasattr(branch.memory, "_step"):
        return out
    mem = branch.memory
    state, S = out[0], out[1]
    _, _, _, al, et, th = mem._project(z)
    al_, et_, th_ = al.median(dim=1).values, et.median(dim=1).values, th.median(dim=1).values
    if FLUSH["alpha"] == "zero":          # no forgetting during the flush: it must only reveal what is in S
        al_ = torch.zeros_like(al_)
    n = FLUSH["n"]
    if FLUSH["auto"]:
        n = int(min(500, round(float(et_.mean()) / max(1e-6, 1 - float(et_.mean())))))
    zero = {k_: torch.zeros_like(v_) for k_, v_ in state.items()}
    for _ in range(n):
        state, S = mem._step(state, S, zero, al_, et_, th_)
    FLUSH["log"].append(n)
    return (state, S) + tuple(out[2:])

LET = "ABCD"
import itertools  # noqa: E402
PERMS = list(itertools.permutations(range(4)))
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"
# presence is out of training: not certified in joint v2 (0 % once distractors are present) and the
# teacher gets it right only 45-70 % in free answer. It stays in the eval (uncertified, reported apart).
TRAIN_FAMS = {"A": ["count_start", "order", "main_object"],
              "B": ["count_start", "order", "main_object", "colour", "number", "object_name"]}
HELD_OUT = "place"
if hasattr(G, "TRAIN_FAMS"):          # v3 generator: its own families, no held-out family
    TRAIN_FAMS, HELD_OUT = G.TRAIN_FAMS, G.HELD_OUT


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

    past_chunk = 0
    past_mode = "chunk"

    @torch.no_grad()
    def layer_states(self, ids_batch):
        """[B, P] ids -> layer-`site` output [B, P, D] (fp32), stopping the forward there.
        With `past_chunk` W > 0 the past is cut in independent blocks of W tokens (no attention across blocks,
        positions restart at 0 in each block): the streaming regime, cost linear in P."""
        W = self.past_chunk
        if W and ids_batch.shape[1] > W:
            B, P = ids_batch.shape
            outs = []
            for s0 in range(0, P, W):
                if self.past_mode == "slide":    # previous W tokens as context, keep the current block's states
                    c0 = max(0, s0 - W)
                    outs.append(self._layer_states_full(ids_batch[:, c0:s0 + W])[:, s0 - c0:])
                else:
                    outs.append(self._layer_states_full(ids_batch[:, s0:s0 + W]))
            return torch.cat(outs, 1)
        return self._layer_states_full(ids_batch)

    def _layer_states_full(self, ids_batch):
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
    torch.save({"pca": pca, "layer": host.site, "n": X.shape[0], "source": f"{G.__name__} train split",
                "past_chunk": host.past_chunk, "past_mode": host.past_mode}, path)
    print(f"basis: {X.shape[0]} states, k={k}, explained {pca['explained']:.3f} -> {path}", flush=True)
    return pca


def build_qa(host, win, it, answer=True):
    """student ids: window + question (+ answer). returns ids, answer positions (slice into logits)."""
    if hasattr(G, "student_ids"):
        return G.student_ids(host.tok, win, it), (host.t(it.get("target_text", it["answer_text"])) if answer else [])
    q = host.t(SYS + "You are reading a long story in pieces.\n\nThe most recent part of the story:\n") + win + \
        host.t(f"\n\nQuestion: {it['question']}\nAnswer with a few words." + END)
    a = host.t(it["answer_text"]) if answer else []
    return q, a


def build_teacher(host, facts, win, it):
    if hasattr(G, "teacher_ids"):
        return G.teacher_ids(host.tok, facts, win, it), host.t(it.get("target_text", it["answer_text"]))
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
    ap.add_argument("--basis", default=os.path.join(HERE, "basis_l24_bl_pca{k}.pt"))
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
    ap.add_argument("--past-choices", default="", dest="past_choices",
                    help="length curriculum: P drawn per step from this list (e.g. 2048,4096,8192); D drawn in "
                         "[0, dmax(P)] with dmax(P) = min(--dmax, P - 2000) or 600 at P 2048")
    ap.add_argument("--eval-pasts", default="", dest="eval_pasts",
                    help="evaluate at each of these P (rows keyed 'P/D'); default: --past only")
    ap.add_argument("--final-eval-groups", type=int, default=100, dest="final_eval_groups")
    ap.add_argument("--mem-type", choices=["deep", "pool", "kvfull", "hybrid"], default="deep", dest="mem_type",
                    help="pool = trained slot memory at the same byte budget (SlotMemory, ~/work/iclr/baselines/bl.py)")
    ap.add_argument("--slots", type=int, default=0, help="pool: cases; 0 = same floats as the deep memory")
    ap.add_argument("--pool", type=int, default=0,
                    help="pool: writes averaged per case; 0 = all the past fits; 1 = raw FIFO of the last writes")
    ap.add_argument("--tau-init", type=float, default=10.0, dest="tau_init")
    ap.add_argument("--proj-residual", action="store_true", dest="proj_residual",
                    help="proj arm started from the working deep recipe: k/v/q/up = their linear map (loaded from "
                         "--init) + a zero-init MLP; down learned from its PCA init at lr x --down-lr-mult")
    ap.add_argument("--down-lr-mult", type=float, default=1.0, dest="down_lr_mult")
    ap.add_argument("--proj-hidden", type=int, default=0, dest="proj_hidden",
                    help="'proj' arm: down learned (PCA init), k/v/q and up as d->H->d MLPs (GELU), same state size")
    ap.add_argument("--write-pool", type=int, default=0, dest="write_pool",
                    help="write the MEAN of k consecutive token states (k = 8: one write per 8 tokens, nothing skipped)")
    ap.add_argument("--write-attnpool", type=int, default=0, dest="write_attnpool",
                    help="write a LEARNED attention pooling of each block of k tokens (softmax(w.z) over the k tokens, "
                         "--attnpool-heads heads averaged, zero init = uniform mean)")
    ap.add_argument("--oracle-mix-steps", type=int, default=0, dest="oracle_mix_steps",
                    help="curriculum: with prob p (1 -> 0 linearly over N steps) a block holding fact tokens is "
                         "written as the mean of its FACT tokens only, else by the learned pooling (train only)")
    ap.add_argument("--oracle-mix-stream", action="store_true", dest="oracle_mix_stream",
                    help="curriculum at STREAM level: with prob p a whole stream is written like --oracle-write "
                         "(fact tokens one by one), else by the learned pooling")
    ap.add_argument("--attnpool-lr-mult", type=float, default=1.0, dest="attnpool_lr_mult")
    ap.add_argument("--attnpool-heads", type=int, default=2, dest="attnpool_heads")
    ap.add_argument("--oracle-values", action="store_true", dest="oracle_values",
                    help="oracle write of ONE token per queried fact (its value token; 8 tokens per story) "
                         "instead of every fact token")
    ap.add_argument("--pool-center", action="store_true", dest="pool_center",
                    help="pool: centre the slot keys (subtract their mean over slots) before the cosine read, "
                         "against the shared direction of the hidden states (diagnosed: near uniform read)")
    ap.add_argument("--head-lr-mult", type=float, default=1.0, dest="head_lr_mult",
                    help="lr multiplier for the memory's read head (branch.memory params: q/k/v projections, tau)")
    ap.add_argument("--addr-nce", type=float, default=0.0, dest="addr_nce",
                    help="lambda of an InfoNCE addressing loss: q at the last question token must pick the key of the "
                         "targeted value token among the written keys (learned temperature); oracle writes only")
    ap.add_argument("--key-decor", type=float, default=0.0, dest="key_decor",
                    help="lambda of a decorrelation penalty on the written keys: ||corr(K) - I||_F^2 / d per stream")
    ap.add_argument("--key-whiten", action="store_true", dest="key_whiten",
                    help="ZCA whitening of k and q with the running (EMA) mean/covariance of the written keys, scale kept")
    ap.add_argument("--mass-probe", type=int, default=0, dest="mass_probe",
                    help="eval only: is the queried count decodable from write mass (|W-W0|, |S|, n writes)?")
    ap.add_argument("--mass-family", default="count_arrivals", dest="mass_family")
    ap.add_argument("--oracle-sent-ends", type=int, default=0, dest="oracle_sent_ends",
                    help="oracle write = end-of-sentence states of K fact sentences drawn at random per story")
    ap.add_argument("--past-chunk", type=int, default=0, dest="past_chunk",
                    help="prefill the past in independent blocks of W tokens (streaming regime); 0 = full prefill")
    ap.add_argument("--past-mode", choices=["chunk", "slide"], default="chunk", dest="past_mode",
                    help="chunk: independent blocks; slide: each block of W prefilled with the previous W as context")
    ap.add_argument("--hybrid-slots", type=int, default=4, dest="hybrid_slots",
                    help="hybrid: raw 2048-d cache slots next to the deep memory (state budget = deep + slots x 2048)")
    ap.add_argument("--flush-steps", default="0", dest="flush_steps",
                    help="eval: after each write stream, apply the remaining momentum by N zero-gradient writes "
                         "(int, or 'auto' = eta/(1-eta) truncated at 500)")
    ap.add_argument("--flush-alpha", default="zero", choices=["zero", "median"], dest="flush_alpha")
    ap.add_argument("--eta-override", type=float, default=None, dest="eta_override_rw",
                    help="eval: pin the momentum gate eta at write time")
    ap.add_argument("--key-read", type=int, default=0, dest="key_read",
                    help="eval only: read each memory AT its own written keys, M_own(k_t) vs M_other(k_t) "
                         "(other story, same load); specificity, identification and recency profile")
    ap.add_argument("--addr-stats", type=int, default=0, dest="addr_stats",
                    help="eval-only: does the read depend on the question? over N stories: mean cos between reads for "
                         "different questions, variance share question vs story, rank of the targeted key under q")
    ap.add_argument("--key-stats", type=int, default=0, dest="key_stats",
                    help="eval-only diagnostic over N stories (oracle write positions): effective rank and mean |cos| "
                         "of the written keys k_t, pr(W1) and relative size of the written state")
    ap.add_argument("--probe-read", type=int, default=0, dest="probe_read",
                    help="eval-only diagnostic: linear probe (colour, number tens) on the READ vector at the last "
                         "question token, over N eval stories; also read-attention entropy for the pool")
    ap.add_argument("--train-down", action="store_true", dest="train_down",
                    help="learn `down` (PCA init) instead of keeping the fixed basis")
    ap.add_argument("--oracle-write", action="store_true", dest="oracle_write",
                    help="positive control: write ONLY the tokens of the fact sentences (stride 1), per stream")
    ap.add_argument("--oracle-support", action="store_true", dest="oracle_support",
                    help="oracle write = end-of-sentence states of the SUPPORTING facts (bAbI support indices)")
    ap.add_argument("--label-loss", type=int, default=1, dest="label_loss",
                    help="1: CE, KL and margin only on the answer LABEL tokens of the target sentence (26/10: with the "
                         "whole official sentence the format tokens dominate, the memory went extinct: theta -> 6e-4, "
                         "write norm -> 0.02, gap 0 by step 300)")
    ap.add_argument("--oracle-content", action="store_true", dest="oracle_content",
                    help="question-agnostic oracle: write the content tokens (names, locations, objects, directions) "
                         "of every fact sentence")
    ap.add_argument("--long-p", type=int, default=8192, dest="long_p")
    ap.add_argument("--long-q", type=int, default=2, dest="long_q",
                    help="at P >= --long-p keep only the first --long-q questions of the story (main first): 26/10, "
                         "4 questions at P 8192 ran out of memory on 21 percent of the steps")
    ap.add_argument("--teacher-notes", choices=["support", "all"], default="support", dest="teacher_notes",
                    help="teacher context: the supporting bAbI sentences (default) or all of them, then the window")
    ap.add_argument("--official", default="",
                    help="eval only on the OFFICIAL BABILong test: 'qa1,qa2:4k,8k' (bl_data/babilong_test/<len>/<task>.json)")
    ap.add_argument("--official-n", type=int, default=100, dest="official_n")
    ap.add_argument("--official-conds", default="win,mem,omem", dest="official_conds",
                    help="win (window only, graft off), mem (own memory), omem (memory of the PREVIOUS sample), "
                         "full (whole context, graft off; skipped above --full-max tokens), orc (oracle write of the "
                         "bAbI sentences located through the 0k file, diagnostic)")
    ap.add_argument("--full-max", type=int, default=33000, dest="full_max")
    ap.add_argument("--init", default=None, help="load a branch state_dict (same architecture flags)")
    ap.add_argument("--eval-only", action="store_true", dest="eval_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    TextHost.past_chunk, TextHost.past_mode = a.past_chunk, a.past_mode
    if a.past_chunk:
        print(f"past prefill: {a.past_mode} blocks of {a.past_chunk} tokens", flush=True)
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
    n_writes = (a.past // a.write_pool if a.write_pool > 1 else a.past // a.write_attnpool if a.write_attnpool > 1
                else -(-a.past // a.write_stride))
    def annotate_values(st, lays):
        """per variant: the position of each queried fact's VALUE token in the past (one per family)."""
        for v, L in enumerate(lays):
            vals = []
            for f in st["items"]:
                ans = re.sub(r"^(the|a|an) ", "", st["items"][f][v]["answer_text"].strip().lower())
                spans = [(s_, e_) for s_, e_ in G.facts_of(L, f) if e_ <= len(L["past"])]
                best = None
                for s_, e_ in spans:          # last fact sentence holding the answer text; its LAST answer token
                    ids_ = L["past"][s_:e_]
                    ends, txt = [], ""
                    for t_ in ids_:
                        txt += tok.decode([t_]); ends.append(len(txt))
                    k_ = txt.lower().rfind(ans)
                    if k_ >= 0:
                        last_char = k_ + len(ans) - 1
                        best = s_ + next(j for j, e in enumerate(ends) if e > last_char)
                        if f == "number":     # Qwen splits numbers into digits: keep every digit token
                            first = s_ + next(j for j, e in enumerate(ends) if e > k_)
                            vals.extend(range(first, best))
                if best is None and spans:    # answer not spelled out (count_start): the sentence's last token
                    best = spans[-1][1] - 1
                if best is not None:
                    vals.append(best)
                    L.setdefault("val_of", {})[f] = best
            L["vals"] = sorted(set(vals))

    if a.mem_type in ("kvfull", "hybrid"):
        import types
        import torch.nn as nn
        if a.mem_type == "kvfull":
            a.slots = a.slots or max(1, (2 * a.memory_hidden * a.d_mem + 2047) // 2048)   # 8 x 2048 = 16384 floats
            a.oracle_values = True
        else:
            a.slots = a.hybrid_slots

        class KVFull(nn.Module):
            """Equal-budget KV cache in the FULL hidden space: `slots` raw layer-24 states (no PCA), read by a
            learned attention (q, k, v: 2048 -> d_mem), then the same norm + up as the deep branch."""
            def __init__(self, D, dk, n, tau):
                super().__init__()
                self.slots, self.dim, self.dk = n, D, dk
                self.k_proj, self.q_proj, self.v_proj = (nn.Linear(D, dk, bias=False) for _ in range(3))
                self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

            def init_memory(self, B):
                return {"slots": torch.zeros(B, 0, self.dim, device=next(self.parameters()).device)}, {}

            def write(self, z):
                d0 = torch.zeros(z.shape[0], z.shape[1], device=z.device)
                return {"slots": z[:, -self.slots:]}, {}, (d0, d0, d0)

            def read(self, state, h):
                s_ = state["slots"]
                if s_.shape[1] == 0:
                    return torch.zeros(*h.shape[:-1], self.dk, device=h.device, dtype=h.dtype)
                l2 = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-6)  # noqa: E731
                att_ = torch.softmax(self.log_tau.exp() * torch.einsum("bnd,bkd->bnk", l2(self.q_proj(h)),
                                                                       l2(self.k_proj(s_))), -1)
                return torch.einsum("bnk,bkd->bnd", att_, self.v_proj(s_))

    if a.mem_type == "hybrid":
        branch.kv = KVFull(2048, a.d_mem, a.slots, a.tau_init).to(a.device)
        branch.slot_q = nn.Parameter(torch.randn(a.slots, a.d_mem, device=a.device) * 0.02)
        branch.slot_k = nn.Linear(2048, a.d_mem, bias=False).to(a.device)

        def _delta_h(self, h, state=None, S=None, *, token_offset=0, write_mask=None):
            """hybrid read: deep retrieve (on down(h)) + cache read (on raw h), summed, then the same norm + up."""
            if state is None:
                state, S = self.init_carry(h.shape[0])
            ds = {k_: v_ for k_, v_ in state.items() if k_ != "slots"}
            r = self.memory.retrieve(ds, self.memory.q_proj(self.down(h)))
            if "slots" in state:
                r = r + self.kv.read(state, h)
            if self.norm_r is not None:
                r = self.norm_r(r)
            return self.up(r), state, S
        branch.delta = types.MethodType(_delta_h, branch)
        print(f"hybrid memory: deep h{a.memory_hidden} + {a.slots} raw slots x 2048", flush=True)
    if a.mem_type == "kvfull":
        branch.memory = KVFull(2048, a.d_mem, a.slots, a.tau_init).to(a.device)
        branch.slot_q = nn.Parameter(torch.randn(a.slots, a.d_mem, device=a.device) * 0.02)
        branch.slot_k = nn.Linear(2048, a.d_mem, bias=False).to(a.device)

        def _delta(self, h, state=None, S=None, *, token_offset=0, write_mask=None):
            if state is None:
                state, S = self.memory.init_memory(h.shape[0])
            r = self.memory.read(state, h)
            if self.norm_r is not None:
                r = self.norm_r(r)
            return self.up(r), state, S
        branch.delta = types.MethodType(_delta, branch)
        print(f"kvfull memory: {a.slots} slots x 2048 floats (raw layer-24 states, no PCA)", flush=True)
    if a.mem_type == "pool":
        sys.path.insert(0, os.path.expanduser("~/work/iclr/baselines"))
        from bl import SlotMemory
        a.slots = a.slots or (2 * a.memory_hidden * a.d_mem) // a.d_mem
        a.pool = a.pool or math.ceil(n_writes / a.slots)
        branch.memory = SlotMemory(a.d_mem, a.slots, a.pool, a.tau_init).to(a.device)
        _ret = branch.memory.retrieve
        att_stats = {"H": [], "maxw": []}

        def _keys(m, s_):
            from bl import _l2
            k_ = m.k_proj(s_)
            if a.pool_center:
                k_ = k_ - k_.mean(1, keepdim=True)
            return _l2(k_)

        def _retrieve_centered(state, q, m=branch.memory):
            from bl import _l2
            s_ = state["slots"]
            if s_.shape[1] <= 1:
                return _ret(state, q)
            att_ = torch.softmax(m.log_tau.exp() * torch.einsum("bnd,bkd->bnk", _l2(q), _keys(m, s_)), -1)
            return torch.einsum("bnk,bkd->bnd", att_, m.v_proj(s_))
        _ret_used = _retrieve_centered if a.pool_center else _ret

        def _retrieve_logged(state, q, _ret=_ret_used, m=branch.memory):
            s_ = state["slots"]
            if s_.shape[1] > 1:
                with torch.no_grad():
                    from bl import _l2
                    att_ = torch.softmax(m.log_tau.exp() * torch.einsum("bnd,bkd->bnk", _l2(q), _keys(m, s_)), -1)
                    H_ = -(att_ * att_.clamp_min(1e-12).log()).sum(-1) / math.log(s_.shape[1])
                    att_stats["H"].append(float(H_.mean())); att_stats["maxw"].append(float(att_.max(-1).values.mean()))
            return _ret(state, q)
        branch.memory.retrieve = _retrieve_logged
        print(f"pool memory: {a.slots} cases x {a.d_mem} floats, P = {a.pool} (writes per stream {n_writes}; "
              f"the last {a.slots * a.pool} writes are kept)", flush=True)
    if a.proj_hidden and not a.proj_residual:
        import torch.nn as nn
        H, dm = a.proj_hidden, a.d_mem
        branch.down.requires_grad_(True)
        for name in ("k_proj", "v_proj", "q_proj"):
            setattr(branch.memory, name, nn.Sequential(nn.Linear(dm, H), nn.GELU(), nn.Linear(H, dm)))
        up = nn.Sequential(nn.Linear(dm, H), nn.GELU(), nn.Linear(H, 2048, bias=False))
        nn.init.zeros_(up[2].weight)                     # identity at init, as the linear up
        branch.up = up
        branch.to(a.device)
    if a.train_down:
        branch.down.requires_grad_(True)
    if a.write_attnpool > 1:
        import torch.nn as nn
        branch.attn_pool = nn.Linear(a.d_mem, a.attnpool_heads).to(a.device)
        nn.init.zeros_(branch.attn_pool.weight)
        nn.init.zeros_(branch.attn_pool.bias)
    def wrap_residual():
        import torch.nn as nn

        class ResMLP(nn.Module):
            def __init__(self, lin, din, H, dout):
                super().__init__()
                self.lin = lin
                self.mlp = nn.Sequential(nn.Linear(din, H), nn.GELU(), nn.Linear(H, dout, bias=False))
                nn.init.zeros_(self.mlp[2].weight)        # exactly the loaded linear map at start

            def forward(self, x):
                return self.lin(x) + self.mlp(x)
        class PostRes(nn.Module):
            """kvfull: y = lin(x) (2048 -> d_mem), then y + MLP(y) with a zero-init last layer."""
            def __init__(self, lin, d, H):
                super().__init__()
                self.lin = lin
                self.mlp = nn.Sequential(nn.Linear(d, H), nn.GELU(), nn.Linear(H, d, bias=False))
                nn.init.zeros_(self.mlp[2].weight)

            def forward(self, x):
                y = self.lin(x)
                return y + self.mlp(y)
        H, dm = a.proj_hidden, a.d_mem
        for name in ("k_proj", "v_proj", "q_proj"):
            if a.mem_type == "kvfull":
                setattr(branch.memory, name, PostRes(getattr(branch.memory, name), dm, H))
            else:
                setattr(branch.memory, name, ResMLP(getattr(branch.memory, name), dm, H, dm))
        branch.up = ResMLP(branch.up, dm, H, 2048)
        branch.down.requires_grad_(True)
        branch.to(a.device)
        print(f"proj-residual: linear maps from --init + zero-init MLPs (H {H}); down learned at lr x {a.down_lr_mult}",
              flush=True)
    pre_wrapped = False
    if a.proj_hidden and a.proj_residual and a.init:
        if any(".lin." in k_ for k_ in torch.load(a.init, map_location="cpu", weights_only=False)["branch"]):
            wrap_residual(); pre_wrapped = True          # checkpoint of a proj-residual run: wrap, then load
    if a.init:
        blob_ = torch.load(a.init, map_location="cpu", weights_only=False)
        missing = branch.load_state_dict(blob_["branch"], strict=False)
        print(f"init from {a.init} (step {blob_.get('step')}): {missing}", flush=True)
    if a.proj_hidden and a.proj_residual and not pre_wrapped:
        wrap_residual()
    keyreg = {"z": [], "q": None}
    if a.addr_nce > 0:
        import torch.nn as nn
        branch.addr_logtau = nn.Parameter(torch.tensor(math.log(10.0), device=a.device))
        _ret_n = branch.memory.retrieve

        def _ret_q(state, q, _r=_ret_n):
            if branch.training:
                keyreg["q"] = q
            return _r(state, q)
        branch.memory.retrieve = _ret_q
    if a.key_whiten:
        import torch.nn as nn

        class Whiten(nn.Module):
            """shared running ZCA: k and q both go through lin, then (x - mu) W, W = Sigma^-1/2 * sqrt(mean eig)."""
            def __init__(self, d, mom=0.01, eps=1e-3):
                super().__init__()
                self.register_buffer("mu", torch.zeros(d)); self.register_buffer("cov", torch.eye(d))
                self.register_buffer("W", torch.eye(d)); self.mom, self.eps = mom, eps

            @torch.no_grad()
            def observe(self, x):
                x = x.reshape(-1, x.shape[-1]).float()
                m = x.mean(0); c = ((x - m).T @ (x - m)) / max(1, x.shape[0] - 1)
                self.mu = torch.lerp(self.mu, m, self.mom); self.cov = torch.lerp(self.cov, c, self.mom)
                ev, U = torch.linalg.eigh(self.cov.double())
                ev = ev.clamp_min(self.eps * ev.max())
                self.W = (U @ torch.diag(ev.rsqrt()) @ U.T * ev.mean().sqrt()).float()   # new tensor: no in-place on the graph

            def apply(self, y):
                return (y - self.mu) @ self.W

        class WProj(nn.Module):
            def __init__(self, lin, wh, observe):
                super().__init__()
                self.lin, self.observe_ = lin, observe
                self.wh = [wh]                                  # shared, not a submodule twice

            def forward(self, x):
                y = self.lin(x)
                if self.observe_ and self.training:
                    self.wh[0].observe(y.detach())
                return self.wh[0].apply(y)
        branch.whiten = Whiten(a.d_mem).to(a.device)
        branch.memory.k_proj = WProj(branch.memory.k_proj, branch.whiten, True)
        branch.memory.q_proj = WProj(branch.memory.q_proj, branch.whiten, False)
        print("key-whiten: running ZCA on k and q (stats from the written keys)", flush=True)
    mix = {"p": 0.0}
    pool_stats = defaultdict(list)          # mean (k * weight) per token kind: >1 = favoured by the pooling
    branch.train()
    wrapper = graft(host.model, branch, layer=host.site)
    # extinction monitors: |delta| / |h| at the graft site during reads
    mon = {"up": [], "h": []}
    branch.up.register_forward_hook(lambda m, i, o: mon["up"].append(float(o.detach().float().norm(dim=-1).mean())))
    wrapper.layer.register_forward_hook(lambda m, i, o: mon["h"].append(
        float((o[0] if isinstance(o, tuple) else o).detach().float().norm(dim=-1).mean())))
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    ap_ids = {id(p) for p in branch.attn_pool.parameters()} if a.write_attnpool > 1 else set()
    hd_mod = branch.kv if a.mem_type == "hybrid" else branch.memory     # hybrid: the x10 head is the cache read
    hd_ids = {id(p) for p in hd_mod.parameters()} - ap_ids if a.head_lr_mult != 1.0 else set()
    dn_ids = {id(p) for p in branch.down.parameters()} if a.down_lr_mult != 1.0 else set()
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in ap_ids | hd_ids | dn_ids], "mult": 1.0},
                             {"params": [p for p in params if id(p) in dn_ids], "mult": a.down_lr_mult},
                             {"params": [p for p in params if id(p) in ap_ids], "mult": a.attnpool_lr_mult},
                             {"params": [p for p in params if id(p) in hd_ids], "mult": a.head_lr_mult}],
                            lr=a.lr, weight_decay=0.01)
    pname = {id(p): n for n, p in branch.named_parameters()}

    def grad_by_group():
        out_ = defaultdict(float)
        for p_ in params:
            if p_.grad is None:
                continue
            n_ = pname.get(id(p_), "?")
            key_ = next((k for k in ("q_proj", "k_proj", "v_proj", "log_tau", "attn_pool", "up", "down") if k in n_),
                        "mem_other" if n_.startswith("memory.") else "other")
            out_[key_] += float(p_.grad.detach().float().pow(2).sum())
        return {k: round(v ** 0.5, 5) for k, v in out_.items()}
    if a.mem_type == "kvfull":
        mem_numel = a.slots * 2048
        st_init = None
    elif a.mem_type == "hybrid":
        st0, _ = branch.init_carry(1)
        mem_numel = sum(v.numel() for v in st0.values()) + a.slots * 2048
    elif a.mem_type == "pool":
        mem_numel = a.slots * a.d_mem
        st_init = None
    else:
        st0, _ = branch.init_carry(1)
        mem_numel = sum(v.numel() for v in st0.values())
    enc = (lambda x: x) if a.mem_type == "kvfull" else (lambda x: branch.down(x))  # noqa: E731
    slotlike = a.mem_type in ("pool", "kvfull")

    content_ids = None
    if a.oracle_content:
        words = G.LOCS + G.PEOPLE + ["Bill", "Fred", "Jeff", "apple", "football", "milk", "north", "south", "east", "west"]
        content_ids = {tok.encode(" " + w, add_special_tokens=False)[0] for w in words}

    def orc_pos(L_):
        if a.oracle_support and "support_ends" in L_:
            return L_["support_ends"]
        if content_ids is not None:           # question-agnostic: the content tokens of EVERY fact sentence
            pos_ = [p for s_, e_, k_ in L_["bounds"] if k_ not in ("scen", "dist") for p in range(s_, e_)
                    if L_.get("past") is None or L_["past"][p] in content_ids]
            return pos_ or [L_["bounds"][-1][1] - 1]
        if a.oracle_sent_ends:     # end-of-sentence state of K fact sentences drawn at random (budget)
            ends = [e_ - 1 for s_, e_, k_ in L_["bounds"] if k_ not in ("scen", "dist")]
            return sorted(random.sample(ends, min(a.oracle_sent_ends, len(ends))))
        if a.oracle_values:
            return L_["vals"]
        return [p for s_, e_, k_ in L_["bounds"] if k_ not in ("scen", "dist") for p in range(s_, e_)]
    kv_tok = 36 * 2 * 2 * 128 * 2          # Qwen2.5-3B: layers x (K,V) x kv-heads x head-dim x bf16 bytes
    budget = {"state_numel": mem_numel, "state_kib_fp32": mem_numel * 4 / 1024,
              "kv_tokens_equiv_bf16": mem_numel * 2 / kv_tok, "hidden_tokens_equiv_bf16": mem_numel * 2 / (2048 * 2),
              "trainable_params": sum(p.numel() for p in params)}
    print("memory budget:", json.dumps(budget), flush=True)
    # identity at init: a written memory must leave the host's logits bit-identical while up == 0
    with torch.no_grad():
        st_ = G.make_story(10_000, "eval")
        L_ = G.layout(st_, tok, P=a.past, W=a.window, D=0)
        annotate_values(st_, L_)
        ids_ = torch.tensor([build_qa(host, L_[0]["window"], next(iter(st_["items"].values()))[0], answer=False)[0]], device=a.device)
        with disabled(wrapper):
            base_ = host.model(input_ids=ids_, use_cache=False).logits.float()
        s1_, S1_, _ = (lambda r: (r[0], r[1], r[2]))(
            branch.memory.write(enc(host.layer_states(torch.tensor([L_[0]["past"]], device=a.device)))[:, ::a.write_stride])
            if a.mem_type in ("pool", "kvfull") else
            write_stream(branch, branch.down(host.layer_states(torch.tensor([L_[0]["past"]], device=a.device)))[:, ::a.write_stride], a.chunk)[:2] + (None,))
        wrapper.carry, wrapper.token_offset = (s1_, S1_), 0
        after_ = host.model(input_ids=ids_, use_cache=False).logits.float()
        print(f"identity at init: {bool(torch.equal(base_, after_))} (max diff {float((after_ - base_).abs().max()):.3g})",
              flush=True)

    def write_mem(pasts, lays_=None):
        """list of past id lists -> (state, S) batched, with graph."""
        ids = torch.tensor(pasts, device=a.device)
        with disabled(wrapper):
            if ids.shape[1] > 8192:          # very long pasts: one stream at a time (prefill memory)
                h = torch.cat([host.layer_states(ids[i:i + 1]) for i in range(ids.shape[0])])
            else:
                h = host.layer_states(ids)
        orc_rows = list(range(len(pasts))) if a.oracle_write else []
        if a.oracle_mix_stream and mix["p"] > 0 and lays_ is not None and branch.training:
            orc_rows = [i for i in range(len(pasts)) if random.random() < mix["p"]]
            mix["n_orc"] = mix.get("n_orc", 0) + len(orc_rows)
        if a.mem_type == "hybrid" and orc_rows:
            parts = [hyb_oracle(h[i:i + 1], lays_[i]) if i in orc_rows else pooled_write(h[i:i + 1], None)
                     for i in range(len(pasts))]
            cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
            return cat([o[0] for o in parts]), cat([o[1] for o in parts]), parts[-1][2]
        if orc_rows and len(orc_rows) < len(pasts):
            # mixed batch: write each stream with its own rule, then stack the carries
            parts = []
            for i in range(len(pasts)):
                if i in orc_rows:
                    pos = orc_pos(lays_[i])
                    if slotlike:
                        parts.append(branch.memory.write(enc(h[i:i + 1, pos])))
                    else:
                        st_, S_, _, g_ = write_stream(branch, branch.down(h[i:i + 1, pos]), a.chunk)
                        parts.append((st_, S_, g_))
                else:
                    parts.append(pooled_write(h[i:i + 1], None))
            cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
            if slotlike:
                parts = [({"slots": eq_slots(o[0]["slots"], a.slots)}, {}, o[2]) for o in parts]
                return cat([o[0] for o in parts]), {}, parts[-1][2]
            return cat([o[0] for o in parts]), cat([o[1] for o in parts]), parts[-1][2]
        if orc_rows:
            outs = []
            for i, L_ in enumerate(lays_):
                pos = orc_pos(L_)
                zi = enc(h[i:i + 1, pos])
                if slotlike:
                    o_ = branch.memory.write(zi)
                    outs.append(({"slots": eq_slots(o_[0]["slots"], a.slots)}, {}, o_[2]))
                else:
                    if branch.training:
                        keyreg["z"].append(zi[0])
                    st_, S_, _, g_ = write_stream(branch, zi, a.chunk)
                    outs.append((st_, S_, g_))
            cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
            return cat([o[0] for o in outs]), cat([o[1] for o in outs]) if outs[0][1] else {}, outs[0][2]
        return pooled_write(h, lays_)

    def hyb_oracle(h1, L_):
        """hybrid oracle: deep writes all fact tokens (the bound), the cache keeps the value tokens."""
        pos_all = [p for s_, e_, k_ in L_["bounds"] if k_ not in ("scen", "dist") for p in range(s_, e_)]
        st_, S_, _, g_ = write_stream(branch, branch.down(h1[:, pos_all]), a.chunk)
        o_ = branch.kv.write(h1[:, L_["vals"]])
        st_ = dict(st_); st_["slots"] = eq_slots(o_[0]["slots"], a.slots)
        return st_, S_, g_

    def perceiver(h, mod):
        kk = branch.slot_k(h)
        kk = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
        qn = branch.slot_q / (branch.slot_q.norm(dim=-1, keepdim=True) + 1e-6)
        w_ = torch.softmax(10.0 * torch.einsum("kd,bnd->bkn", qn, kk), -1)
        return mod.write(torch.einsum("bkn,bnd->bkd", w_, h))

    def eq_slots(x, n):
        """pool batching: bring a [1, k, d] slot set to exactly n slots (k > n: keep the last n, FIFO as the
        memory itself; k < n: cyclic repeat, which leaves the attention read nearly unchanged)."""
        if x.shape[1] >= n:
            return x[:, -n:]
        reps = -(-n // x.shape[1])
        return x.repeat(1, reps, 1)[:, -n:]

    def pooled_write(h, lays_):
        if a.mem_type == "kvfull":
            kk = branch.slot_k(h)
            kk = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
            qn = branch.slot_q / (branch.slot_q.norm(dim=-1, keepdim=True) + 1e-6)
            w_ = torch.softmax(10.0 * torch.einsum("kd,bnd->bkn", qn, kk), -1)
            return branch.memory.write(torch.einsum("bkn,bnd->bkd", w_, h))
        if a.write_attnpool > 1:
            k_ = a.write_attnpool
            zz = branch.down(h)
            Bn, Nn, dn = zz.shape
            zz = zz[:, :Nn // k_ * k_].view(Bn, -1, k_, dn)
            w_ = torch.softmax(branch.attn_pool(zz), dim=2)                 # [B, nb, k, H]
            z = (w_.unsqueeze(-1) * zz.unsqueeze(3)).sum(2).mean(2)         # [B, nb, d]
            if mix["p"] > 0 and lays_ is not None and branch.training and not a.oracle_mix_stream:
                fm = torch.zeros(Bn, zz.shape[1] * k_, device=zz.device)
                for i, L_ in enumerate(lays_):
                    for s_, e_, kind in L_["bounds"]:
                        if kind not in ("scen", "dist") and s_ < fm.shape[1]:
                            fm[i, s_:min(e_, fm.shape[1])] = 1
                fm = fm.view(Bn, -1, k_)
                cnt = fm.sum(2, keepdim=True)
                zf = (zz * fm.unsqueeze(-1)).sum(2) / cnt.clamp_min(1)
                pick = ((cnt[..., 0] > 0) & (torch.rand(Bn, zz.shape[1], device=zz.device) < mix["p"])).unsqueeze(-1)
                z = torch.where(pick, zf, z)
            if lays_ is not None and branch.training:
                with torch.no_grad():
                    wk = (w_.mean(-1) * k_).reshape(Bn, -1)                     # per token
                    for i, L_ in enumerate(lays_):
                        for s_, e_, kind in L_["bounds"]:
                            if e_ > wk.shape[1]:
                                continue
                            cls = kind if kind in ("scen", "dist") else ("gist" if kind in G.GIST else "detail")
                            pool_stats[cls].append(float(wk[i, s_:e_].mean()))
        elif a.write_pool > 1:
            zz = branch.down(h)
            Bn, Nn, dn = zz.shape
            z = zz[:, :Nn // a.write_pool * a.write_pool].view(Bn, -1, a.write_pool, dn).mean(2)
        else:
            z = branch.down(h)[:, ::a.write_stride]
        if a.mem_type == "pool":
            return branch.memory.write(z)
        state, S, _, gates = write_stream(branch, z, a.chunk)
        if a.mem_type == "hybrid":
            state = dict(state); state["slots"] = perceiver(h, branch.kv)[0]["slots"]
        return state, S, gates

    def lr_scale(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    BLM = hasattr(G, "student_ids")

    @torch.no_grad()
    def evaluate(ngroups, tag):
        branch.eval()
        rows = []
        pasts_ = [int(x) for x in a.eval_pasts.split(",")] if a.eval_pasts else [a.past]
        for P_, D in [(P_, D) for P_ in pasts_ for D in [int(x) for x in a.eval_dists.split(",")]]:
            Dkey = D if len(pasts_) == 1 else f"P{P_}/{D}"
            prev = None
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t_all, t_wr = time.time(), 0.0
            for g in range(10_000, 10_000 + ngroups):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=P_, W=a.window, D=D)
                annotate_values(st, lays)
                torch.cuda.synchronize(); t0_ = time.time()
                state, S, _ = write_mem([l["past"] for l in lays], lays)
                torch.cuda.synchronize(); t_wr += time.time() - t0_
                if g == 10_000:
                    snumel = sum(v[0].numel() for v in state.values())
                win = lays[0]["window"]
                for f in st["items"]:
                    its = st["items"][f]
                    seqs = []
                    for it in (its if not BLM else []):
                        o = "\n".join(f"{LET[i]}. {t}" for i, t in enumerate(it["options"]))
                        seqs.append(host.t(SYS + "You are reading a long story in pieces.\n\nThe most recent part of the story:\n")
                                    + win + host.t(f"\n\nQuestion: {it['question']}\n{o}\nAnswer with the letter of the correct option." + END))
                    if not BLM:                 # letter MCQ (synthetic benches only)
                        ids, att = pad_batch(seqs * 2, host.pad)
                        lens = [len(s) for s in seqs] * 2
                        set_carry(wrapper, state, S, [0, 1, 2, 3, 1, 2, 3, 0])     # own, then other (v+1)
                        out = host.model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), use_cache=False).logits
                        lg = torch.stack([out[i, lens[i] - 1, host.lid].float().cpu() for i in range(8)])
                        with disabled(wrapper):
                            ids4, att4 = pad_batch(seqs, host.pad)
                            o4 = host.model(input_ids=ids4.to(a.device), attention_mask=att4.to(a.device), use_cache=False).logits
                            lw = torch.stack([o4[i, len(seqs[i]) - 1, host.lid].float().cpu() for i in range(4)])
                    # option likelihood in the TRAINED free-answer format: sum log p(option text | window, question)
                    fseqs = [build_qa(host, win, it, answer=False)[0] for it in its]
                    lk = {}
                    conds_ = [("lk_mem", [0, 1, 2, 3]), ("lk_other", [1, 2, 3, 0]), ("lk_win", None)]
                    if prev is not None:
                        conds_.append(("lk_ostory", [0, 1, 2, 3]))
                    for cond, rows_ in conds_:
                        seq_, qlen_, alen_, rr = [], [], [], []
                        for v in range(4):
                            for o_ in (its[v]["option_texts"] if BLM else [x[0].upper() + x[1:] for x in its[v]["options"]]):
                                at = host.t(o_)
                                seq_.append(fseqs[v] + at); qlen_.append(len(fseqs[v])); alen_.append(len(at))
                                rr.append(rows_[v] if rows_ else 0)
                        ids_, att_ = pad_batch(seq_, host.pad)
                        if rows_ is None:
                            with disabled(wrapper):
                                lp_ = answer_logp(host.model, ids_, att_, qlen_, alen_, a.device)
                        else:
                            if cond == "lk_ostory":
                                set_carry(wrapper, prev[0], prev[1], rr)
                            else:
                                set_carry(wrapper, state, S, rr)
                            lp_ = answer_logp(host.model, ids_, att_, qlen_, alen_, a.device)
                        sc = []
                        for j, (lpj, sq) in enumerate(zip(lp_, seq_)):
                            tgt = torch.tensor(sq[qlen_[j]:qlen_[j] + alen_[j]], device=lpj.device)
                            sc.append(float(lpj.gather(1, tgt[:, None]).sum()))
                        lk[cond] = torch.tensor(sc).view(4, 4)
                    # free answer, own memory, greedy 8 tokens (16 for the official answer sentences)
                    set_carry(wrapper, state, S, [0, 1, 2, 3])
                    gen = greedy(fseqs, 16 if BLM else 8)
                    if BLM:                     # no letter MCQ: the lg_* keys carry the option likelihood
                        lg = torch.cat([lk["lk_mem"], lk["lk_other"]])
                        lw = lk["lk_win"]
                    for v, it in enumerate(its):
                        rows.append({"tag": tag, "D": Dkey, "group": g, "family": f, "variant": v, "answer": it["answer"],
                                     "lg_mem": lg[v].tolist(), "lg_other": lg[4 + v].tolist(), "lg_win": lw[v].tolist(),
                                     "lk_mem": lk["lk_mem"][v].tolist(), "lk_other": lk["lk_other"][v].tolist(),
                                     "lk_win": lk["lk_win"][v].tolist(),
                                     **({"lk_ostory": lk["lk_ostory"][v].tolist()} if "lk_ostory" in lk else {}),
                                     "donor_answer": its[(v + 1) % 4]["answer"], "free": gen[v],
                                     "free_ok": int(G.judge(it, gen[v]) if BLM else ans_ok(it["answer_text"], gen[v]))})
                prev = (state, S)
            torch.cuda.synchronize()
            print("EVAL-COST " + json.dumps({"P": P_, "D": D, "groups": ngroups, "stories_per_group": len(lays),
                  "write_s_per_story": round(t_wr / (ngroups * len(lays)), 3),
                  "total_s_per_group": round((time.time() - t_all) / ngroups, 2),
                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                  "state_floats_per_story": snumel, "state_kib_fp32": round(snumel * 4 / 1024, 1)}), flush=True)
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

    @torch.no_grad()
    def official_eval():
        """OFFICIAL BABILong test, official prompt (context = the last W tokens of the input) and metric.
        The rest of the input (tokens[:-W]) is written into the memory in streaming (--past-chunk)."""
        sys.path.insert(0, os.path.join(HERE, "babilong_ext"))
        from metrics import compare_answers, TASK_LABELS
        branch.eval()
        conds = a.official_conds.split(",")
        tasks_, lens_ = a.official.split(":")
        orc_flag = a.oracle_write
        fo = open(a.out + "_official.jsonl", "a")
        summ = {}
        for Ln in lens_.split(","):
            for t in tasks_.split(","):
                ds = json.load(open(os.path.join(HERE, "bl_data", "babilong_test", Ln, f"{t}.json")))[:a.official_n]
                z0 = json.load(open(os.path.join(HERE, "bl_data", "babilong_test", "0k", f"{t}.json")))
                acc, prev, t_wr, t0_ = defaultdict(list), None, 0.0, time.time()
                for i, ex in enumerate(ds):
                    text = ex["input"].strip()
                    it = {"task": t, "question": ex["question"], "answer_text": ex["target"]}
                    enc_ = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                    ctx, offs = enc_["input_ids"], enc_["offset_mapping"]
                    win, past = ctx[-a.window:], ctx[:-a.window]
                    q = G.student_ids(tok, win, it)
                    rec = {"task": t, "length": Ln, "i": i, "target": ex["target"], "ntok": len(ctx), "n_past": len(past)}
                    outs = {}
                    if "win" in conds:
                        with disabled(wrapper):
                            outs["win"] = greedy([q], 16)[0]
                    if "full" in conds and len(ctx) + 600 <= a.full_max:
                        qf = G.student_ids(tok, ctx, it)
                        with disabled(wrapper):
                            x_ = torch.tensor([qf], device=a.device)
                            try:
                                go = host.model.generate(input_ids=x_, attention_mask=torch.ones_like(x_), max_new_tokens=16,
                                                         do_sample=False)
                                outs["full"] = tok.decode(go[0, x_.shape[1]:], skip_special_tokens=True)
                            except torch.OutOfMemoryError:
                                torch.cuda.empty_cache()
                                rec["full_oom"] = 1
                    if any(c in conds for c in ("mem", "omem", "orc")) and len(past) > 0:
                        torch.cuda.synchronize(); tw = time.time()
                        a.oracle_write = False
                        st_, S_, _ = write_mem([past], None)
                        a.oracle_write = orc_flag
                        torch.cuda.synchronize(); t_wr += time.time() - tw
                        rows_q = []
                        if "mem" in conds:
                            rows_q.append(("mem", st_, S_))
                        if "omem" in conds and prev is not None:
                            rows_q.append(("omem", prev[0], prev[1]))
                        if "orc" in conds:
                            # locate the bAbI sentences of the 0k version of this sample (same question/target)
                            zz = z0[i] if i < len(z0) else None
                            spans = []
                            if zz and zz["question"] == ex["question"] and zz["target"] == ex["target"]:
                                pos_c = 0
                                for f_ in [x.strip() + "." for x in zz["input"].split(".") if x.strip()]:
                                    k_ = text.find(f_, pos_c)
                                    if k_ < 0:
                                        spans = []
                                        break
                                    spans.append((k_, k_ + len(f_))); pos_c = k_ + len(f_)
                            if spans:
                                tpos = [j for j, (c0, c1) in enumerate(offs[:len(past)])
                                        if any(c0 < e_ and c1 > s_ for s_, e_ in spans)]
                                if tpos:
                                    fake = {"bounds": [(p_, p_ + 1, t) for p_ in tpos], "vals": tpos,
                                            "support_ends": tpos}
                                    a.oracle_write = True
                                    so_, So_, _ = write_mem([past], [fake])
                                    a.oracle_write = orc_flag
                                    rows_q.append(("orc", so_, So_))
                                    rec["orc_writes"] = len(tpos)
                        for name_, s1, S1 in rows_q:
                            set_carry(wrapper, s1, S1, [0])
                            outs[name_] = greedy([q], 16)[0]
                        prev = (st_, S_)
                    for c_, o_ in outs.items():
                        ok_ = bool(compare_answers(ex["target"], o_, ex["question"], TASK_LABELS[t]))
                        acc[c_].append(ok_)
                        rec[c_] = o_; rec[c_ + "_ok"] = int(ok_)
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                res = {c_: round(100 * sum(v_) / len(v_), 1) for c_, v_ in acc.items()} | \
                      {c_ + "_n": len(v_) for c_, v_ in acc.items()}
                res["write_s_per_sample"] = round(t_wr / max(1, len(ds)), 2)
                res["s_per_sample"] = round((time.time() - t0_) / max(1, len(ds)), 2)
                summ[f"{t}@{Ln}"] = res
                print("OFFICIAL " + json.dumps({"task": t, "length": Ln, **res}), flush=True)
        with open(a.out + "_official_summary.json", "w") as fs_:
            json.dump(summ, fs_, indent=1)

    def summarize(rows):
        res = {}
        by = defaultdict(dict)
        for r in rows:
            by[(r["D"], r["family"], r["group"])][r["variant"]] = r
        acc = defaultdict(lambda: defaultdict(list))
        for (D, f, g), vs in by.items():
            if len(vs) < 4:
                continue
            for key in ("lg_mem", "lg_other", "lg_win", "lk_mem", "lk_other", "lk_win", "lk_ostory"):
                if any(key not in vs[v] for v in range(4)):
                    continue
                L = torch.tensor([vs[v][key] for v in range(4)])
                C = L - L.mean(0, keepdim=True)
                acc[(D, f)][key].append(all(int(C[v].argmax()) == vs[v]["answer"] for v in range(4)))
                best = max(PERMS, key=lambda p: sum(float(L[v, p[v]]) for v in range(4)))
                tie = bool((L - L[0:1]).abs().max() < 1e-6)       # identical rows (window only): no information
                acc[(D, f)][key + "_perm"].append(all(best[v] == vs[v]["answer"] for v in range(4)) and not tie)
                acc[(D, f)][key + "_q"].append(sum(int(L[v].argmax()) == vs[v]["answer"] for v in range(4)) / 4)
            acc[(D, f)]["donor"].append(sum(int(torch.tensor(vs[v]["lg_other"]).argmax()) == vs[v]["donor_answer"]
                                            for v in range(4)) / 4)
            acc[(D, f)]["free"].append(sum(vs[v]["free_ok"] for v in range(4)) / 4)
        for (D, f), d in sorted(acc.items()):
            res[f"{f}@{D}"] = {k: round(100 * sum(v) / len(v), 1) for k, v in d.items()} | {"n": len(d["lg_mem"])}
        return res

    FLUSH["alpha"] = a.flush_alpha
    if a.flush_steps == "auto":
        FLUSH["auto"] = True
    else:
        FLUSH["n"] = int(a.flush_steps)
    if a.eta_override_rw is not None:
        branch.memory.eta_override = a.eta_override_rw
    if FLUSH["n"] or FLUSH["auto"] or a.eta_override_rw is not None:
        print(f"read-time dynamics: flush {a.flush_steps}, eta override {a.eta_override_rw}", flush=True)
    if a.mass_probe:
        # is the count decodable from the MASS of the write (number of writes, |W - W0|, |S|)?
        # within each group the 4 variants have 4 distinct n: a feature that ranks them in n order
        # would let the best-assignment metric score without reading any content.
        branch.eval()
        fam_ = a.mass_family
        feats, labs, rank_ok = defaultdict(list), [], defaultdict(int)
        s0, _ = branch.init_carry(1)
        with torch.no_grad():
            for g in range(10_000, 10_000 + a.mass_probe):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=a.past, W=a.window, D=300)
                annotate_values(st, lays)
                state, S, _ = write_mem([l["past"] for l in lays], lays)
                at_ = [st["items"][fam_][v]["answer_text"] for v in range(4)]
                nv = [float(x) if x.replace(".", "").isdigit() else float(st["items"][fam_][v]["answer"]) for v, x in enumerate(at_)]
                fv = defaultdict(list)
                for v in range(4):
                    dW = sum(float((state[k_][v] - s0[k_][0]).norm() ** 2) for k_ in s0) ** 0.5
                    nS = sum(float(S[k_][v].norm() ** 2) for k_ in S) ** 0.5 if S else 0.0
                    nw = len(orc_pos(lays[v])) if a.oracle_write else len(lays[v]["past"])
                    for k_, x_ in (("dW", dW), ("S", nS), ("n_writes", nw)):
                        fv[k_].append(x_); feats[k_].append(x_)
                labs += nv
                for k_, xs in fv.items():          # within-group spread of the mass feature
                    m_ = sum(xs) / 4
                    feats[k_ + "_spread"].append((max(xs) - min(xs)) / (abs(m_) + 1e-9))
                order_n = sorted(range(4), key=lambda i: nv[i])
                for k_, xs in fv.items():
                    if sorted(range(4), key=lambda i: xs[i]) == order_n or sorted(range(4), key=lambda i: -xs[i]) == order_n:
                        rank_ok[k_] += 1
        import numpy as np
        out_ = {"n_groups": a.mass_probe, "family": fam_, "chance_rank": round(2 / 24, 3)}
        for k_, xs in feats.items():
            if k_.endswith("_spread"):
                out_[k_ + "_rel_mean"] = round(float(np.mean(xs)), 4)
                continue
            xs_ = np.array(xs)
            c_ = float(np.corrcoef(xs_, np.array(labs))[0, 1]) if xs_.std() > 0 else 0.0
            out_[k_] = {"corr_with_n": round(c_, 3), "group_rank_exact": round(rank_ok[k_] / a.mass_probe, 3)}
        print("MASS-PROBE " + json.dumps(out_), flush=True)
        print("TRAIN-TEXT-AB-FINI", flush=True)
        return

    if a.key_read:
        branch.eval()
        mem_ = branch.memory
        cs = lambda x, y: F.cosine_similarity(x.float(), y.float(), dim=-1)  # noqa: E731
        stories = []
        with torch.no_grad():
            for g in range(50_000, 50_000 + a.key_read):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=a.past, W=a.window, D=300)
                annotate_values(st, lays)
                L0 = lays[0]
                h = host.layer_states(torch.tensor([L0["past"]], device=a.device))
                pos_ = orc_pos(L0)
                z = enc(h[:, pos_])
                st_, _S, _, _ = write_stream(branch, z, a.chunk)
                kr = mem_.k_proj(z)                       # raw key, retrieve applies the qk norm / feature map
                vv = mem_.v_proj(z)
                fam_idx = {f: pos_.index(L0["val_of"][f]) for f in G.FAMILIES
                           if L0.get("val_of", {}).get(f) in pos_}
                stories.append({"state": st_, "k": kr, "v": vv[0], "fam": fam_idx, "n": len(pos_)})
            N_ = len(stories)
            spec_f, own_f, oth_f = defaultdict(list), defaultdict(list), defaultdict(list)
            prof = defaultdict(list)                      # decile of write index -> specificity
            prof_own = defaultdict(list)
            rd_own, rd_oth = [], []
            for i, S_i in enumerate(stories):
                S_j = stories[(i + 1) % N_]
                r_own = mem_.retrieve(S_i["state"], S_i["k"])[0]
                r_oth = mem_.retrieve(S_j["state"], S_i["k"])[0]
                c_own, c_oth = cs(r_own, S_i["v"]), cs(r_oth, S_i["v"])
                n_ = S_i["n"]
                for t in range(n_):
                    b_ = min(9, int(10 * t / n_))
                    prof[b_].append(float(c_own[t] - c_oth[t])); prof_own[b_].append(float(c_own[t]))
                for f, t in S_i["fam"].items():
                    spec_f[f].append(float(c_own[t] - c_oth[t])); own_f[f].append(float(c_own[t]))
                    oth_f[f].append(float(c_oth[t]))
                rd_own.append(r_own); rd_oth.append(r_oth)
            ident, ident_oth = {}, {}
            for f in G.FAMILIES:
                ii = [i for i, S_i in enumerate(stories) if f in S_i["fam"]]
                if len(ii) < 2:
                    continue
                V = torch.stack([stories[i]["v"][stories[i]["fam"][f]] for i in ii]).float()
                Vn = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
                ok, ok_o = 0, 0
                for m_, i in enumerate(ii):
                    t = stories[i]["fam"][f]
                    for rd, acc in ((rd_own[i][t], "own"), (rd_oth[i][t], "oth")):
                        sc = Vn @ (rd.float() / (rd.float().norm() + 1e-9))
                        hit = int(int(sc.argmax()) == m_)
                        if acc == "own":
                            ok += hit
                        else:
                            ok_o += hit
                ident[f] = round(ok / len(ii), 3); ident_oth[f] = round(ok_o / len(ii), 3)
        mean = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
        print("KEY-READ " + json.dumps({
            "n_stories": N_, "n_writes_mean": round(sum(S["n"] for S in stories) / N_, 1),
            "chance_ident": round(1 / N_, 3),
            "spec_by_family (cos own - cos other)": {f: mean(v) for f, v in spec_f.items()},
            "cos_own_by_family": {f: mean(v) for f, v in own_f.items()},
            "cos_other_by_family": {f: mean(v) for f, v in oth_f.items()},
            "ident_own": ident, "ident_other": ident_oth,
            "spec_by_write_decile (0=first)": {b: mean(prof[b]) for b in sorted(prof)},
            "cos_own_by_write_decile": {b: mean(prof_own[b]) for b in sorted(prof_own)},
            "flush_steps_mean": mean(FLUSH["log"]) if FLUSH["log"] else 0}), flush=True)
        print("TRAIN-TEXT-AB-FINI", flush=True)
        return

    if a.addr_stats:
        branch.eval()
        cap = {}
        _ret0 = branch.memory.retrieve

        def _capret(state, q, _r=_ret0):
            out_ = _r(state, q)
            cap["q"], cap["r"] = q.detach(), out_.detach()
            return out_
        branch.memory.retrieve = _capret
        R_all, ranks, coss = [], defaultdict(list), []
        with torch.no_grad():
            for g in range(40_000, 40_000 + a.addr_stats):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=a.past, W=a.window, D=300)
                annotate_values(st, lays)
                L0 = lays[0]
                h = host.layer_states(torch.tensor([L0["past"]], device=a.device))
                # what was written, and where (token positions covered by each write)
                if a.oracle_write:
                    pos_ = orc_pos(L0); zw = enc(h[:, pos_]); cover = [[p_] for p_ in pos_]
                elif a.write_attnpool > 1:
                    k_ = a.write_attnpool; zz = branch.down(h); nb = zz.shape[1] // k_
                    zz = zz[:, :nb * k_].view(1, nb, k_, -1)
                    w_ = torch.softmax(branch.attn_pool(zz), dim=2)
                    zw = (w_.unsqueeze(-1) * zz.unsqueeze(3)).sum(2).mean(2)
                    cover = [list(range(b_ * k_, (b_ + 1) * k_)) for b_ in range(nb)]
                else:
                    zw = branch.down(h)[:, ::a.write_stride]; cover = [[p_] for p_ in range(0, h.shape[1], a.write_stride)]
                st_, S_, _, _ = write_stream(branch, zw, a.chunk)
                K = branch.memory.k_proj(zw)[0]
                Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-9)
                reads = []
                for f in G.FAMILIES:
                    it = st["items"][f][0]
                    ids_ = torch.tensor([build_qa(host, L0["window"], it, answer=False)[0]], device=a.device)
                    wrapper.carry, wrapper.token_offset = (st_, S_), 0
                    host.model(input_ids=ids_, use_cache=False)
                    r_ = cap["r"][0, -1].float(); q_ = cap["q"][0, -1].float()
                    reads.append(r_)
                    tgt = L0.get("val_of", {}).get(f)
                    idx = [i_ for i_, c_ in enumerate(cover) if tgt is not None and tgt in c_]
                    if idx:
                        sc = Kn @ (q_ / (q_.norm() + 1e-9))
                        rk = float((sc > sc[idx[0]]).sum()) / max(1, len(sc) - 1)
                        ranks[f].append(rk)
                Rm = torch.stack(reads)
                Rn = Rm / (Rm.norm(dim=-1, keepdim=True) + 1e-9)
                C = Rn @ Rn.T
                coss.append(float((C.sum() - C.diagonal().sum()) / (C.numel() - C.shape[0])))
                R_all.append(Rm.cpu())
        R3 = torch.stack(R_all)                                   # [story, family, d]
        tot = float(((R3 - R3.mean((0, 1), keepdim=True)) ** 2).sum(-1).mean())
        q_eff = float(((R3.mean(0, keepdim=True) - R3.mean((0, 1), keepdim=True)) ** 2).sum(-1).mean())
        s_eff = float(((R3.mean(1, keepdim=True) - R3.mean((0, 1), keepdim=True)) ** 2).sum(-1).mean())
        print("ADDR-STATS " + json.dumps({
            "n_stories": len(R_all), "n_writes": len(cover),
            "mean_cos_between_questions": round(sum(coss) / len(coss), 4),
            "var_share_question": round(q_eff / tot, 4), "var_share_story": round(s_eff / tot, 4),
            "target_key_rank_pct (0=best, 0.5=chance)": {f: round(sum(v) / len(v), 3) for f, v in ranks.items()}}),
            flush=True)
        print("TRAIN-TEXT-AB-FINI", flush=True)
        return

    if a.key_stats:
        branch.eval()
        res_ = defaultdict(list)
        with torch.no_grad():
            for g in range(30_000, 30_000 + a.key_stats):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=a.past, W=a.window, D=300)
                annotate_values(st, lays)
                L0 = lays[0]
                h = host.layer_states(torch.tensor([L0["past"]], device=a.device))
                z = enc(h[:, orc_pos(L0)])
                K = branch.memory.k_proj(z)[0].double()
                Kn = K / K.norm(dim=-1, keepdim=True)
                C = Kn @ Kn.T
                n_ = C.shape[0]
                off = (C.abs().sum() - n_) / max(1, n_ * (n_ - 1))
                sv = torch.linalg.svdvals(K) ** 2
                svc = torch.linalg.svdvals(K - K.mean(0, keepdim=True)) ** 2
                res_["n_writes"].append(n_)
                res_["key_pr"].append(float(sv.sum() ** 2 / (sv ** 2).sum()))
                res_["key_pr_centred"].append(float(svc.sum() ** 2 / (svc ** 2).sum()))
                res_["key_mean_abs_cos"].append(float(off))
                if a.mem_type == "deep":
                    st_, S_, _, g_ = write_stream(branch, z, a.chunk)
                    s0, _ = branch.init_carry(1)
                    res_["pr_w1"].append(pr(st_["w1"][0]))
                    res_["state_rel_change"].append(sum(float((st_[k_][0] - s0[k_][0]).norm()) for k_ in s0) /
                                                    max(1e-9, sum(float(s0[k_][0].norm()) for k_ in s0)))
                    res_["w1_shape"] = [list(st_["w1"][0].shape)]
        print("KEY-STATS " + json.dumps({k: (v[0] if k == "w1_shape" else round(sum(v) / len(v), 4))
                                          for k, v in res_.items()}), flush=True)
        print("TRAIN-TEXT-AB-FINI", flush=True)
        return

    if a.probe_read:
        branch.eval()
        got = {"r": None}
        mname_ = "read" if a.mem_type == "kvfull" else "retrieve"
        _r0 = getattr(branch.memory, mname_)

        def _cap(state, q, _r0=_r0):
            r_ = _r0(state, q)
            got["r"] = r_.detach()
            return r_
        setattr(branch.memory, mname_, _cap)
        feats_, labs_ = defaultdict(list), defaultdict(list)
        with torch.no_grad():
            for g in range(20_000, 20_000 + a.probe_read):
                st = G.make_story(g, "eval")
                lays = G.layout(st, tok, P=a.past, W=a.window, D=300)
                annotate_values(st, lays)
                state, S, _ = write_mem([l["past"] for l in lays], lays)
                for fam in ("colour", "number", "count_start"):
                    its = st["items"][fam]
                    seqs = [build_qa(host, lays[0]["window"], it, answer=False)[0] for it in its]
                    ids_, att_ = pad_batch(seqs, host.pad)
                    set_carry(wrapper, state, S, [0, 1, 2, 3])
                    host.model(input_ids=ids_.to(a.device), attention_mask=att_.to(a.device), use_cache=False)
                    r_ = got["r"]
                    for v, it in enumerate(its):
                        feats_[fam].append(r_[v, len(seqs[v]) - 1].float().cpu())
                        at_ = it["answer_text"].strip().lower()
                        labs_[fam].append(str(int(at_) // 10) if fam == "number" and at_.isdigit() else at_)
        for fam in feats_:
            cls_ = sorted(set(labs_[fam])); X = torch.stack(feats_[fam])
            y = torch.tensor([cls_.index(l_) for l_ in labs_[fam]]); ncls_ = len(cls_)
            maj_ = max(labs_[fam].count(c_) for c_ in cls_) / len(y)
            ntr = int(0.75 * len(y))
            X = (X - X[:ntr].mean(0)) / (X[:ntr].std(0) + 1e-6)
            W_ = torch.zeros(X.shape[1], ncls_, requires_grad=True); b_ = torch.zeros(ncls_, requires_grad=True)
            o_ = torch.optim.Adam([W_, b_], lr=1e-2)
            for _ in range(500):
                l_ = F.cross_entropy(X[:ntr] @ W_ + b_, y[:ntr]) + 1e-3 * W_.pow(2).sum()
                o_.zero_grad(); l_.backward(); o_.step()
            acc_ = float(((X[ntr:] @ W_ + b_).argmax(1) == y[ntr:]).float().mean())
            print(f"PROBE-READ {fam}: answer value from the read vector, held out {100 * acc_:.1f}% "
                  f"({ncls_} classes, majority {100 * maj_:.0f}%, n={len(y)})", flush=True)
        if a.mem_type == "pool":
            print(f"PROBE-READ read attention: entropy/log K {sum(att_stats['H']) / max(1, len(att_stats['H'])):.4f}, "
                  f"max weight {sum(att_stats['maxw']) / max(1, len(att_stats['maxw'])):.4f}, "
                  f"tau {float(branch.memory.log_tau.exp()):.3f}", flush=True)
        print("TRAIN-TEXT-AB-FINI", flush=True)
        return

    if a.official:
        official_eval()
        print("TRAIN-TEXT-AB-FINI", flush=True)
        return

    hist, log, t0 = [], [], time.time()
    kept = defaultdict(list)          # teacher kept (right) per family, for the KL mask
    fams = TRAIN_FAMS.get(a.arm)
    run = None
    if a.eval_only:
        a.steps = 0
    for step in range(a.steps + 1):
        if a.eval_every and step % a.eval_every == 0 and (step > 0 or a.steps <= 100):
            ng = a.final_eval_groups if step == a.steps else a.eval_groups
            rows = evaluate(ng, step)
            s = summarize(rows)
            hist.append({"step": step, "eval": s})
            with open(a.out + f"_eval{step}.jsonl", "w") as fo:
                for r in rows:
                    fo.write(json.dumps(r) + "\n")
            print(f"  == eval {step}: OPTION-LIKELIHOOD per question % mem / other / window || group by best "
                  f"assignment % mem / other variant / window / other STORY (chance 1/24 = 4.2) || letter-MCQ "
                  f"contrastive mem | free mem", flush=True)
            for k, v in s.items():
                print(f"     {k:22s} {v.get('lk_mem_q', -1):5.1f} / {v.get('lk_other_q', -1):5.1f} / {v.get('lk_win_q', -1):5.1f}"
                      f" || {v.get('lk_mem_perm', -1):5.1f} / {v.get('lk_other_perm', -1):5.1f} / {v.get('lk_win_perm', -1):5.1f}"
                      f" / {v.get('lk_ostory_perm', -1):5.1f}"
                      f" || {v['lg_mem']:5.1f} | {v['free']:5.1f}  n{v['n']}", flush=True)
        if step == a.steps:
            break
        for gr in opt.param_groups:
            gr["lr"] = a.lr * lr_scale(step) * gr["mult"]
        mon["up"].clear(); mon["h"].clear()
        mix["p"] = max(0.0, 1.0 - step / a.oracle_mix_steps) if a.oracle_mix_steps else 0.0
        g = 1_000_000 + a.seed * 100_000 + step
        st = G.make_story(g, "train")
        if a.past_choices:
            P_ = rng.choice([int(x) for x in a.past_choices.split(",")])
            D = rng.randrange(0, (600 if P_ <= 2048 else min(a.dmax, P_ - 2000)) + 1)
        else:
            P_ = a.past
            D = rng.randrange(0, a.dmax + 1)
        try:
            lays = G.layout(st, tok, P=P_, W=a.window, D=D)
        except ValueError as e_:
            print(f"  step {step}: layout refused ({e_}), skipped", flush=True)
            continue
        annotate_values(st, lays)
        v1, v2 = rng.sample(range(4), 2)
        win = lays[0]["window"]
        try:
            state, S, (al, et, th) = write_mem([lays[v1]["past"], lays[v2]["past"]], [lays[v1], lays[v2]])
            if a.arm in ("A", "B"):
                qf = list(st["items"]) if hasattr(G, "student_ids") else rng.sample(fams, min(a.kq, len(fams)))
                if hasattr(G, "student_ids") and P_ >= a.long_p:   # memory: fewer questions on the longest pasts
                    qf = qf[:a.long_q]
                stu, tea, meta = [], [], []
                for r, v in enumerate((v1, v2)):
                    for f in qf:
                        it = st["items"][f][v]
                        q, ans = build_qa(host, win, it)
                        fb_ = G.facts_of(lays[v], f)
                        if a.teacher_notes == "support" and "support" in it:   # supporting sentences only
                            fb_ = [fb_[i_] for i_ in it["support"] if i_ < len(fb_)]
                        facts = [x for s_, e_ in fb_ for x in lays[v]["past"][s_:e_]]
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
                        ok = (G.judge(it_, tok.decode(gt)) if hasattr(G, "judge") else ans_ok(it_["answer_text"], tok.decode(gt))) \
                            and len(gt) > 0
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
                def label_idx(an_, lab_):
                    """positions of the answer LABEL inside the target tokens (the only content-bearing ones in an
                    official answer sentence); all positions if not found."""
                    txt_, ends_ = "", []
                    for t_ in an_:
                        txt_ += tok.decode([t_]); ends_.append(len(txt_))
                    k_ = txt_.lower().rfind(lab_.lower())
                    if k_ < 0:
                        return list(range(len(an_)))
                    st_ = [0] + ends_[:-1]
                    return [j for j in range(len(an_)) if st_[j] < k_ + len(lab_) and ends_[j] > k_]
                for i, (q, an, r) in enumerate(stu):
                    tgt = torch.tensor(an, device=a.device)
                    if a.label_loss:
                        lab_ = st["items"][meta[i]][(v1, v2)[r]]["answer_text"]
                        jj = torch.tensor(label_idx(an, lab_), device=a.device)
                    else:
                        jj = torch.arange(len(an), device=a.device)
                    ce_own.append(F.nll_loss(slp[i][jj], tgt[jj]))
                    ce_oth.append(F.nll_loss(slp[n + i][jj], tgt[jj]))
                    if tok_ok[i]:
                        kls.append((tlp[i][jj].exp() * (tlp[i][jj] - slp[i][jj])).sum(-1).mean())
                ce_own, ce_oth = torch.stack(ce_own), torch.stack(ce_oth)
                l_ce = ce_own.mean()
                l_kl = torch.stack(kls).mean() if kls else torch.zeros((), device=a.device)
                cmask = torch.tensor([float(st["items"][meta[i]][(v1, v2)[r]]["answer_text"] !=
                                            st["items"][meta[i]][(v1, v2)[1 - r]]["answer_text"])
                                      for i, (_q, _an, r) in enumerate(stu)], device=a.device)
                # margin only where the two memories imply different answers (extra simulated questions may not)
                l_c = (F.relu(a.margin + ce_own - ce_oth) * cmask).sum() / cmask.sum().clamp_min(1.0)
                loss = l_ce + a.lam_kl * l_kl + a.lam_c * l_c
                info = {"ce": float(l_ce), "kl": float(l_kl), "con": float(l_c), "gap": float((ce_oth - ce_own).mean()),
                        "teach_ok": sum(tok_ok) / n}
                if a.addr_nce > 0 and keyreg["z"] and keyreg["q"] is not None:
                    nces, rks = [], []
                    lays_w = [lays[v1], lays[v2]]
                    for i, (q_ids, _an, r_) in enumerate(stu):
                        tgt_pos = lays_w[r_].get("val_of", {}).get(meta[i])
                        wpos = orc_pos(lays_w[r_])
                        if tgt_pos is None or tgt_pos not in wpos:
                            continue
                        K_ = branch.memory.k_proj(keyreg["z"][r_])
                        Kn_ = K_ / (K_.norm(dim=-1, keepdim=True) + 1e-6)
                        qv = keyreg["q"][i, len(q_ids) - 1]
                        lg_ = branch.addr_logtau.exp() * (Kn_ @ (qv / (qv.norm() + 1e-6)))
                        ti = wpos.index(tgt_pos)
                        nces.append(F.cross_entropy(lg_[None], torch.tensor([ti], device=a.device)))
                        with torch.no_grad():
                            rks.append(float((lg_ > lg_[ti]).sum()) / max(1, len(lg_) - 1))
                    if nces:
                        l_nce = torch.stack(nces).mean()
                        loss = loss + a.addr_nce * l_nce
                        info["nce"] = float(l_nce); info["tgt_rank"] = sum(rks) / len(rks)
                        info["nce_tau"] = float(branch.addr_logtau.exp())
                    keyreg["q"] = None
                if keyreg["z"]:
                    pens, prs = [], []
                    for z_ in keyreg["z"]:
                        K_ = branch.memory.k_proj(z_)
                        Kc = K_ - K_.mean(0, keepdim=True)
                        Kc = Kc / (Kc.norm(dim=0, keepdim=True) + 1e-6)
                        Cm = Kc.T @ Kc
                        pens.append(((Cm - torch.eye(Cm.shape[0], device=Cm.device)) ** 2).sum() / Cm.shape[0])
                        with torch.no_grad():
                            sv_ = torch.linalg.svdvals(K_.detach().double()) ** 2
                            prs.append(float(sv_.sum() ** 2 / (sv_ ** 2).sum()))
                    pen = torch.stack(pens).mean()
                    if a.key_decor > 0:
                        loss = loss + a.key_decor * pen
                    info["key_pen"] = float(pen); info["key_pr"] = sum(prs) / len(prs)
                    keyreg["z"].clear()
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
        gg = grad_by_group() if step % 10 == 0 else None
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
            gl = {"step": step, "P": P_, "D": D, "loss": float(loss), "loss_run": run, "gn": gn, **info,
                  "write_rel_norm": wn, "delta_over_h": dr, "mix_p": mix["p"],
                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                  "pool_w": {k_: round(sum(v_[-200:]) / len(v_[-200:]), 3) for k_, v_ in pool_stats.items()},
                  "kept": {f_: round(sum(v_) / len(v_), 3) for f_, v_ in kept.items()},
                  "eta_med": float(et.median()), "alpha_med": float(al.median()), "theta_med": float(th.median()),
                  "pr_w1": pr(state["w1"][0].detach()) if "w1" in state else None,
                  "grad": gg,
                  **({"att_H": round(sum(att_stats["H"][-50:]) / max(1, len(att_stats["H"][-50:])), 4),
                      "att_maxw": round(sum(att_stats["maxw"][-50:]) / max(1, len(att_stats["maxw"][-50:])), 4),
                      "tau": round(float(branch.memory.log_tau.exp()), 3)} if a.mem_type == "pool" else {}),
                  "s_per_step": (time.time() - t0) / (step + 1)}
            log.append(gl)
            print("  " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in gl.items() if k not in ("kept", "pool_w", "grad"))
                  + (f" pool_w {gl['pool_w']}" if gl["pool_w"] else "") + f" grad {gg}", flush=True)
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
