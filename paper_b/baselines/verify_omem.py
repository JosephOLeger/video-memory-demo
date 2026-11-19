#!/usr/bin/env python3
"""Check of the omem splice: the AHN recurrent state (all 36 layers) after prefilling
  (i) record B's own prompt  pre + ctx_B + post_B           (compressed region = past_B)
  (ii) the omem splice      pre + past_B + win_A + post_A   (compressed region = past_B)
must be the same tensor up to kernel numerics; and (iii) record A's own prompt must differ.
Usage: source env_ahn.sh; CUDA_VISIBLE_DEVICES=1 python verify_omem.py [--variant GDN]"""
import argparse
import importlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ahn_eval as E  # noqa: E402


@torch.no_grad()
def states(model, ids, sinks, window):
    model.config.num_attn_sinks, model.config.sliding_window = int(sinks), int(window)
    out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
    mc = out.mem_past_key_values
    return [mc[i]["recurrent_state"].float().clone() for i in range(len(mc))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="GDN")
    ap.add_argument("--data", default="../text/evalsets/b30_dev_4k_1ep_sp.jsonl")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.45)
    G = importlib.import_module(os.environ.get("TEXT_GEN", "gen_bench30"))
    tok, model, base, n_ahn, FLAGS, _ = E.load_models(a.variant)
    recs = [json.loads(x) for _, x in zip(range(2), open(a.data))]
    W = 256
    parts = []
    for r in recs:
        it = {"task": r["type"], "question": r["question"], "answer_text": r["answer"], "accept": r.get("accept"),
              "yn": r.get("yn"), "allkeys": r.get("allkeys"), "universe": r.get("universe")}
        sk = G.student_ids(tok, [-1], it)
        n_pre = sk.index(-1)
        parts.append((sk[:n_pre], r["ids"], sk[n_pre + 1:]))
    (preB, ctxB, postB), (preA, ctxA, postA) = parts
    assert preA == preB
    n_pre = len(preA)
    sB = states(model, preB + ctxB + postB, n_pre, W + len(postB))
    sS = states(model, preA + ctxB[:-W] + ctxA[-W:] + postA, n_pre, W + len(postA))
    sA = states(model, preA + ctxA + postA, n_pre, W + len(postA))
    sB2 = states(model, preB + ctxB + postB, n_pre, W + len(postB))
    # same compressed tokens, different suffix length only (B's past + B's window + A's question)
    sQ = states(model, preB + ctxB + postA, n_pre, W + len(postA))
    per = lambda x, y: [round(float((u - v).norm() / v.norm().clamp_min(1e-12)), 4) for u, v in zip(x, y)]
    print("PER_LAYER splice_vs_B", per(sS, sB))
    print("PER_LAYER B_rerun_vs_B", per(sB2, sB))
    print("PER_LAYER Bwin_Aquestion_vs_B", per(sQ, sB))
    print("PER_LAYER A_vs_B", per(sA, sB))
    rel = lambda x, y: max(float((u - v).norm() / v.norm().clamp_min(1e-12)) for u, v in zip(x, y))
    res = {"variant": a.variant, "max_rel_diff_splice_vs_B": rel(sS, sB), "max_rel_diff_A_vs_B": rel(sA, sB),
           "layers": len(sB), "state_shape": list(sB[0].shape)}
    print("VERIFY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
