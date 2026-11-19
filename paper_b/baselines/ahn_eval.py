#!/usr/bin/env python3
"""AHN baseline (ByteDance Seed, arXiv 2510.07318) on our benchmarks, with the CONTENT CONTROL.

Host: Qwen2.5-3B-Instruct (text model; our graft runs on Qwen2.5-VL-3B-Instruct: host mismatch, see README).
Prompt: exactly our student prompt (gen_bench30.student_ids: SYS + PRE + context + question + END, Qwen chat
format; BABILong qa* records use the official BABILong prompt, as in our pipeline). Greedy, --gen new tokens.
Judge: gen_bench30.judge (BABILong qa*: official compare_answers).

Conditions (per record, ctx = context token ids, past = ctx[:-W], window = ctx[-W:]):
  win   base Qwen2.5-3B (AHN off, full attention) on  pre + ctx[-W:] + post
  full  base Qwen2.5-3B on pre + ctx + post (whole context)
  mem   AHN model on pre + ctx + post, attention sinks = the prompt prefix (pre, n_pre tokens), sliding window
        = W + n_post tokens, so at the end of the prefill the lossless KV is exactly pre + last W context tokens +
        question, and everything in between (= past) has been compressed into the AHN recurrent state.
  swa   CONTROL of the attention channel: same model, same sinks and window as mem, but the AHN output is zeroed
        (the recurrent state is still updated, never read). In a sliding window model the window tokens' hidden
        states carry information from up to 36 x W tokens back (one window per layer), so mem - swa isolates the
        compressed memory, mem - win mixes it with this channel.
  omem  CONTENT CONTROL: compressed state (and everything before the window) from the PREVIOUS record, window and
        question from this record. Implemented by SPLICING the input: pre_A + past_B + window_A + post_A, with
        past_B = ctx_B[:-W]. AHN is causal and the sinks are the shared prefix, so at the first window token the
        recurrent state of every layer is bit-for-bit the state AHN builds on record B's own prompt (same tokens,
        same positions up to |past_B|), and the window and question of A are then processed reading that state.
        This is the exact equivalent of exporting B's state and importing it before A's window; it also swaps the
        sliding-window channel (A's first window tokens attend to B's tail), so omem removes ALL information about
        A's past. B = previous record of the file, exactly as in our train_b30 jsonl_eval. bench30 files cycle through
        the 30 types, so B is a story of ANOTHER type; for BABILong (one file per task) B is the same task.
Output: <out>/<tag>_<variant>.jsonl (one line per record) and <out>/<tag>_<variant>_summary.json
(per condition: all, per type, held / not held for bench30; mean seconds per record per condition).

Usage (tower, env ahn):
  source env_ahn.sh
  CUDA_VISIBLE_DEVICES=1 python ahn_eval.py --data ../text/evalsets/b30_dev_4k_1ep_sp.jsonl \
    --variant GDN --conds win,mem,swa,omem,full --n 5 --out runs/smoke
  --data babilong:4k:qa1,qa2,...   reads <B30_BL_TEST>/4k/qaN.json (input, question, target)
"""
import argparse
import collections
import importlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.environ.get("AHN_HF", os.path.join(HERE, "hf"))      # the AHN modules and the Qwen2.5-3B host
os.environ.setdefault("HF_HUB_CACHE", HF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
VARIANTS = {"GDN": "ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B",
            "DN": "ByteDance-Seed/AHN-DN-for-Qwen-2.5-Instruct-3B",
            "Mamba2": "ByteDance-Seed/AHN-Mamba2-for-Qwen-2.5-Instruct-3B"}
BASE = "Qwen/Qwen2.5-3B-Instruct"
BL_DIR = os.environ.get("B30_BL_TEST", os.path.join(HERE, "..", "text", "bl_data", "babilong_test"))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl path, or babilong:<len>:<task,task,...>")
    ap.add_argument("--variant", default="GDN", choices=list(VARIANTS))
    ap.add_argument("--conds", default="win,mem,swa,omem,full")
    ap.add_argument("--window", type=int, default=256, help="W context tokens kept lossless (ours: 256). "
                    "AHN native LongBench setting: --window 8064 --sinks 128 (covers all of a 4k input)")
    ap.add_argument("--sinks", default="prefix", help="'prefix' = the prompt prefix tokens (matched to our student, "
                    "which sees SYS+PRE in clear), or an int (AHN native: 128). With an int < n_pre part of the "
                    "instruction is compressed; with an int > n_pre the first context tokens stay lossless")
    ap.add_argument("--n", type=int, default=0, help="first n records per file / task (0 = all)")
    ap.add_argument("--every", type=int, default=1, help="keep one record in k (omem then uses the previous KEPT "
                    "record). bench30 files cycle through the 30 types: use k prime with 30 (7, 11) to cover them all")
    ap.add_argument("--gen", type=int, default=16, help="max new tokens (ours: 16)")
    ap.add_argument("--full-max", type=int, default=32000, dest="full_max", help="skip full if len(ctx)+600 > this")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--min-free-gb", type=float, default=12.0, dest="min_free_gb")
    ap.add_argument("--mem-frac", type=float, default=0.45, dest="mem_frac",
                    help="hard cap of this process on the card (fraction of total memory): we OOM, not the neighbour")
    ap.add_argument("--state-report", action="store_true", help="measure the AHN recurrent state on the first record")
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------ data
def load_records(spec, n, every=1):
    """yields (tag, list of (rec, ctx_ids or text, it))"""
    if spec.startswith("babilong:"):
        _, ln, tasks = spec.split(":")
        for t in tasks.split(","):
            ds = json.load(open(os.path.join(BL_DIR, ln, f"{t}.json")))
            ds = ds[::every]
            ds = ds[:n] if n else ds
            out = []
            for i, ex in enumerate(ds):
                it = {"task": t, "question": ex["question"], "answer_text": ex["target"]}
                rec = {"id": f"{t}-{ln}-{i}", "type": t, "question": ex["question"], "answer": ex["target"]}
                out.append((rec, ex["input"].strip(), it))
            yield f"bl{ln}_{t}", out
    else:
        recs = [json.loads(x) for x in open(spec)]
        recs = recs[::every]
        recs = recs[:n] if n else recs
        out = []
        for rec in recs:
            it = {"task": rec.get("type", "free"), "question": rec["question"], "answer_text": rec["answer"],
                  "accept": rec.get("accept"), "yn": rec.get("yn", False), "allkeys": rec.get("allkeys", False),
                  "universe": rec.get("universe")}
            out.append((rec, rec["ids"] if "ids" in rec else rec["input"], it))
        yield os.path.splitext(os.path.basename(spec))[0], out


# ------------------------------------------------------------------------------------------------ model
def load_models(variant):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as HFQ
    from ahn.transformer.qwen2_ahn import register_customized_qwen2
    import ahn.transformer.qwen2_ahn.qwen2_ahn as QA
    import ahn.utils as AU

    register_customized_qwen2()
    ahn_path = snapshot_download(VARIANTS[variant])
    base_path = snapshot_download(BASE)
    tok = AutoTokenizer.from_pretrained(base_path)
    cfg = AutoConfig.from_pretrained(ahn_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, config=cfg, torch_dtype=torch.bfloat16)
    sd = load_file(os.path.join(ahn_path, "model.safetensors"), device="cpu")
    msd = model.state_dict()
    missing = [k for k in sd if k not in msd]
    assert not missing, f"AHN keys absent from the model: {missing[:5]}"
    model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype != torch.bfloat16 else v for k, v in sd.items()},
                          strict=False)
    n_ahn = sum(v.numel() for v in sd.values())
    model = model.cuda().eval()
    for gc_ in (model.generation_config,):
        gc_.do_sample, gc_.temperature, gc_.top_p, gc_.top_k = False, None, None, None

    # base model (plain transformers Qwen2) SHARING the frozen weights of the AHN model (no second copy)
    bcfg = HFQ.Qwen2Config.from_pretrained(base_path)
    bcfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        base = HFQ.Qwen2ForCausalLM(bcfg)
    bsd = {k: v for k, v in model.state_dict().items() if ".ahn." not in k}
    res = base.load_state_dict(bsd, strict=False, assign=True)
    assert not [k for k in res.missing_keys if "lm_head" not in k], res.missing_keys[:5]
    base.tie_weights()
    base.model.rotary_emb = HFQ.Qwen2RotaryEmbedding(config=bcfg, device="cuda")
    base = base.eval()
    base.generation_config.do_sample = False
    base.generation_config.temperature = None
    base.generation_config.top_p = None
    base.generation_config.top_k = None

    # ---- patches (inference only, behaviour documented in the README) ----
    # (1) decode path: AHN's attention drops the oldest window token at index `num_attn_sinks`, hard-coded default
    #     128 in forward_inference, while the layer uses config.num_attn_sinks. Pass the config value.
    orig_fi = QA.Qwen2MemFlexAttn.forward_inference

    def forward_inference(self, *a, **k):
        k.setdefault("num_attn_sinks", getattr(self.config, "num_attn_sinks", 0))
        return orig_fi(self, *a, **k)
    QA.Qwen2MemFlexAttn.forward_inference = forward_inference

    # (2) zero the AHN read for the swa control (state still written)
    orig_ahn = AU.BaseAHN.forward
    FLAGS = {"zero": False}

    def ahn_forward(self, *a, **k):
        o, pkv = orig_ahn(self, *a, **k)
        return (o * 0 if FLAGS["zero"] else o), pkv
    AU.BaseAHN.forward = ahn_forward
    QA.BaseAHN.forward = ahn_forward
    return tok, model, base, n_ahn, FLAGS, ahn_path


def eos_ids(tok):
    return [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if i is not None]


@torch.no_grad()
def gen_base(base, tok, ids, ngen):
    x = torch.tensor([ids], device="cuda")
    out = base.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=ngen, do_sample=False,
                        eos_token_id=eos_ids(tok), pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def gen_ahn(model, tok, ids, ngen, n_sinks, window):
    model.config.num_attn_sinks = int(n_sinks)
    model.config.sliding_window = int(window)
    x = torch.tensor([ids], device="cuda")
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=ngen, do_sample=False,
                         num_beams=1, eos_token_id=eos_ids(tok), pad_token_id=tok.eos_token_id, use_cache=True)
    return tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def state_report(model, ids, n_sinks, window):
    """one prefill with use_cache; sizes of the AHN recurrent state (all layers) and of the lossless KV."""
    model.config.num_attn_sinks = int(n_sinks)
    model.config.sliding_window = int(window)
    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, use_cache=True)
    mc, kv = out.mem_past_key_values, out.past_key_values
    numel = nbytes = 0
    shapes = collections.Counter()
    for li in range(len(mc)):
        st = mc[li]
        for key, t in st.items():
            if torch.is_tensor(t):
                numel += t.numel()
                nbytes += t.numel() * t.element_size()
                shapes[f"{key}:{tuple(t.shape)}:{t.dtype}"] += 1
            elif isinstance(t, (tuple, list)):
                for u in t:
                    if torch.is_tensor(u):
                        numel += u.numel()
                        nbytes += u.numel() * u.element_size()
                        shapes[f"{key}:{tuple(u.shape)}:{u.dtype}"] += 1
    kvb = kvt = 0
    if kv is not None:
        legacy = kv.to_legacy_cache() if hasattr(kv, "to_legacy_cache") else kv
        for k_, v_ in legacy:
            kvb += (k_.numel() + v_.numel()) * k_.element_size()
            kvt = k_.shape[-2]
    return {"ahn_state_numel": numel, "ahn_state_bytes": nbytes, "ahn_state_tensors": dict(shapes),
            "kv_tokens_kept": kvt, "kv_bytes_kept": kvb, "prompt_tokens": len(ids)}


def mcnemar(rows, c1, c2):
    """exact two-sided McNemar on records holding both conditions: b = c1 right & c2 wrong, c = the reverse"""
    import math
    pr = [(r[c1 + "_ok"], r[c2 + "_ok"]) for r in rows if c1 + "_ok" in r and c2 + "_ok" in r]
    b = sum(1 for x, y in pr if x and not y)
    c = sum(1 for x, y in pr if y and not x)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)
    return {"n": len(pr), "b": b, "c": c, "diff_pts": round(100 * (b - c) / max(1, len(pr)), 1), "p": round(p, 5)}


# ------------------------------------------------------------------------------------------------ main
def main():
    a = parse()
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
    if free / 2**30 < a.min_free_gb:
        sys.exit(f"not enough free GPU memory ({free / 2**30:.1f} GiB < {a.min_free_gb}); not starting")
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    G = importlib.import_module(os.environ.get("TEXT_GEN", "gen_bench30"))
    tok, model, base, n_ahn, FLAGS, ahn_path = load_models(a.variant)
    print(f"AHN {a.variant}: {n_ahn:,} trained params ({ahn_path})", flush=True)
    conds = a.conds.split(",")
    os.makedirs(a.out, exist_ok=True)
    W = a.window
    for tag, items in load_records(a.data, a.n, a.every):
        fo = open(os.path.join(a.out, f"{tag}_{a.variant}.jsonl"), "w")
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        tim = collections.defaultdict(list)
        prev_past = None
        did_state = False
        rows = []
        for i, (rec, src, it) in enumerate(items):
            ctx = list(src) if not isinstance(src, str) else tok(src, add_special_tokens=False)["input_ids"]
            win, past = ctx[-W:], ctx[:-W]
            skel = G.student_ids(tok, [-1], it)
            n_pre = skel.index(-1)
            pre, post = skel[:n_pre], skel[n_pre + 1:]
            n_post = len(post)
            sinks = n_pre if a.sinks == "prefix" else int(a.sinks)
            ahn_win = W + n_post   # lossless at the end of the prefill: sinks + last W context tokens + question
            outs = {}

            def run(name, fn):
                torch.cuda.synchronize()
                t0 = time.time()
                try:
                    outs[name] = fn()
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM {name} rec {i} ntok {len(ctx)}", flush=True)
                    return
                torch.cuda.synchronize()
                tim[name].append(time.time() - t0)

            if a.state_report and not did_state and len(past) > 0:
                rep = state_report(model, pre + ctx + post, sinks, ahn_win)
                rep.update(variant=a.variant, trained_params=n_ahn, window=W, sinks=sinks)
                json.dump(rep, open(os.path.join(a.out, f"state_{a.variant}.json"), "w"), indent=1)
                print("STATE " + json.dumps(rep), flush=True)
                did_state = True
                torch.cuda.empty_cache()
            if "win" in conds:
                run("win", lambda: gen_base(base, tok, pre + win + post, a.gen))
            if "full" in conds and len(ctx) + 600 <= a.full_max:
                run("full", lambda: gen_base(base, tok, pre + ctx + post, a.gen))
            if len(past) > 0:
                if "mem" in conds:
                    run("mem", lambda: gen_ahn(model, tok, pre + ctx + post, a.gen, sinks, ahn_win))
                if "swa" in conds:
                    FLAGS["zero"] = True
                    try:
                        run("swa", lambda: gen_ahn(model, tok, pre + ctx + post, a.gen, sinks, ahn_win))
                    finally:
                        FLAGS["zero"] = False
                if "omem" in conds and prev_past is not None:
                    run("omem", lambda: gen_ahn(model, tok, pre + prev_past + win + post, a.gen, sinks, ahn_win))
                prev_past = past
            out_rec = {k: rec[k] for k in rec if k not in ("ids", "input")}
            out_rec.update(ntok=len(ctx), n_pre=n_pre, n_post=n_post, sinks=sinks, ahn_window=ahn_win, W=W)
            held = rec.get("held")
            for c, o in outs.items():
                ok = bool(G.judge(it, o))
                out_rec[c] = o
                out_rec[c + "_ok"] = int(ok)
                out_rec[c + "_s"] = round(tim[c][-1], 3) if tim[c] else None
                acc["_all"][c].append(ok)
                acc[str(rec.get("type", "free"))][c].append(ok)
                if held is not None:
                    acc["_held" if held else "_notheld"][c].append(ok)
            rows.append(out_rec)
            fo.write(json.dumps(out_rec) + "\n")
            fo.flush()
            print(f"  {tag} {i + 1}/{len(items)} " + " ".join(f"{c}={outs[c]!r}:{out_rec[c + '_ok']}" for c in outs)
                  + f" | gold {rec.get('answer')!r}", flush=True)
        summ = {t: {c: round(100 * sum(v) / len(v), 1) for c, v in d.items()} | {"n": max(len(v) for v in d.values())}
                for t, d in sorted(acc.items())}
        summ["_s_per_rec"] = {c: round(sum(v) / len(v), 3) for c, v in tim.items()}
        summ["_s_per_rec_excl_first"] = {c: round(sum(v[1:]) / max(1, len(v) - 1), 3) for c, v in tim.items()}
        summ["_mcnemar"] = {f"{x}_vs_{y}": mcnemar(rows, x, y) for x, y in
                            (("mem", "omem"), ("mem", "swa"), ("mem", "win"), ("swa", "win"), ("full", "mem"))}
        summ["_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        summ["_config"] = {"variant": a.variant, "window": W, "sinks": a.sinks, "gen": a.gen, "trained_params": n_ahn,
                           "host": BASE, "data": a.data}
        json.dump(summ, open(os.path.join(a.out, f"{tag}_{a.variant}_summary.json"), "w"), indent=1)
        print("SUMMARY " + tag + " " + json.dumps({"_all": summ.get("_all"), "s": summ["_s_per_rec"]}), flush=True)


if __name__ == "__main__":
    main()
