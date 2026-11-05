"""Non-learned sanity baselines on the OFFICIAL BABILong test (26/10): does averaging the facts solve it?

The bAbI sentences of each official sample are located through the 0k file (same question and target,
sentences found in order in the long input; samples where this fails are skipped and counted).
Official prompt and metric; the context of the prompt is [X] + the last W tokens of the input:
    win        X = nothing (window only)
    mean1      X = ONE soft token: the mean input embedding of all fact tokens (perfect selection, averaging)
    meanS      X = one soft token per fact sentence (mean embedding of the sentence)
    facts      X = the fact sentences as text (perfect selection, no compression: upper bound of a memory)
Greedy 16 tokens, graft off, host frozen.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "babilong_ext"))
import gen_babilong as G  # noqa: E402
from metrics import compare_answers, TASK_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8")
    ap.add_argument("--lengths", default="4k")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--conds", default="win,mean1,meanS,facts")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    E = model.get_input_embeddings()
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    @torch.no_grad()
    def greedy_embeds(emb, n=16):
        out = []
        for _ in range(n):
            L = emb.shape[1]
            pos = torch.arange(L, device="cuda").view(1, 1, L).expand(3, 1, L)
            lg = model(inputs_embeds=emb, position_ids=pos, use_cache=False).logits[0, -1]
            t = int(lg.argmax())
            if t in stops:
                break
            out.append(t)
            emb = torch.cat([emb, E(torch.tensor([[t]], device="cuda"))], 1)
        return tok.decode(out)

    fo = open(a.out, "w")
    summ = {}
    for Ln in a.lengths.split(","):
        for t in a.tasks.split(","):
            ds = json.load(open(os.path.join(HERE, "bl_data", "babilong_test", Ln, f"{t}.json")))[:a.n]
            z0 = json.load(open(os.path.join(HERE, "bl_data", "babilong_test", "0k", f"{t}.json")))
            acc, skipped = defaultdict(list), 0
            for i, ex in enumerate(ds):
                text = ex["input"].strip()
                zz = z0[i] if i < len(z0) else None
                spans, pos_c = [], 0
                if zz and zz["question"] == ex["question"] and zz["target"] == ex["target"]:
                    for f_ in [x.strip() + "." for x in zz["input"].split(".") if x.strip()]:
                        k_ = text.find(f_, pos_c)
                        if k_ < 0:
                            spans = []
                            break
                        spans.append((k_, k_ + len(f_)))
                        pos_c = k_ + len(f_)
                if not spans:
                    skipped += 1
                    continue
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                win = ids[-a.W:]
                sent_tok = [[j for j, (c0, c1) in enumerate(offs) if c0 < e and c1 > s] for s, e in spans]
                it = {"task": t, "question": ex["question"], "answer_text": ex["target"]}
                pre, post = G.prompt_parts(t, ex["question"])
                pre_ids, post_ids = tok.encode(pre, add_special_tokens=False), tok.encode(post, add_special_tokens=False)
                rec = {"task": t, "length": Ln, "i": i, "target": ex["target"], "n_facts": len(spans)}
                with torch.no_grad():
                    ids_t = torch.tensor(ids, device="cuda")
                    for c in a.conds.split(","):
                        if c == "win":
                            X = E(torch.tensor([pre_ids + win + post_ids], device="cuda"))
                        elif c == "facts":
                            fids = [x for st in sent_tok for x in ids[st[0]:st[-1] + 1]]
                            X = E(torch.tensor([pre_ids + fids + win + post_ids], device="cuda"))
                        else:
                            fe = [E(ids_t[st]).float() for st in sent_tok]
                            soft = (torch.cat(fe).mean(0, keepdim=True) if c == "mean1"
                                    else torch.stack([x.mean(0) for x in fe])).to(torch.bfloat16)
                            X = torch.cat([E(torch.tensor([pre_ids], device="cuda"))[0], soft,
                                           E(torch.tensor([win + post_ids], device="cuda"))[0]])[None]
                        o = greedy_embeds(X)
                        ok = bool(compare_answers(ex["target"], o, ex["question"], TASK_LABELS[t]))
                        acc[c].append(ok)
                        rec[c] = o
                        rec[c + "_ok"] = int(ok)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
            res = {c: round(100 * sum(v) / max(1, len(v)), 1) for c, v in acc.items()}
            res["n"] = len(acc.get("win", [])) or max((len(v) for v in acc.values()), default=0)
            res["skipped_unaligned"] = skipped
            summ[f"{t}@{Ln}"] = res
            print("MEAN-BASELINE " + json.dumps({"task": t, "length": Ln, **res}), flush=True)
    json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    print("MEAN-BASELINE-FINI", flush=True)


if __name__ == "__main__":
    main()
