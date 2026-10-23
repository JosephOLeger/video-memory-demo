"""Untrained memory ladder on synthetic stories (text), EVAL split. Port of ladder3.py (video).

Host: Qwen2.5-VL-3B-Instruct, text input only (same weights as the video graft), or any
causal LM via --model. The host sees
    [notes = memory (B tokens)] [most recent part = window (W tokens)] [question + options]
and the letter logits are read at the first answer position.

Memories (budget B tokens unless said otherwise):
    full     the whole past + window as one text (attention ceiling, P + W tokens)
    oracle   the fact sentences only, in order
    other    the fact sentences of ANOTHER variant of the same group (donor)
    fifo@D   the last B tokens of the past
    key@D    one sentence every k sentences over the past, random phase per (group, D)
    mean@D   past cut in P/B segments of B tokens, input embeddings averaged per position
    lex@D    sorted set of the past's content words that never occur in the filler
             (the copy shortcut: keywords without order or binding)
    empty    window only;  blind  question only
D = distance (tokens) from the end of the last fact sentence to the window.

    CUDA_VISIBLE_DEVICES=3 python ladder_text.py --families count_end --groups 2 --out runs/smoke.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_story as G  # noqa: E402

GEN_MD5 = hashlib.md5(open(G.__file__, "rb").read()).hexdigest()[:8]
STOP = set("""a an the and or of to in on at by for with was were is are be been had has have it its
that this then later after while some point much who what which there their they he she his her
""".split())


class Host:
    def __init__(self, model="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda"):
        from transformers import AutoTokenizer
        self.device = device
        if "VL" in model:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        else:
            from transformers import AutoModelForCausalLM as M
        self.model = M.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lid = [self.tok.encode(c, add_special_tokens=False)[0] for c in G.LET]
        self.emb = self.model.get_input_embeddings()
        self._soft = None   # (mask [L] bool, values [n, d])
        self.ph = self.tok.encode("#", add_special_tokens=False)[0]

        def hook(mod, inp, out):
            if self._soft is None:
                return out
            m, val = self._soft
            out = out.clone()
            out[0, m] = val.to(out.dtype)
            return out
        self.emb.register_forward_hook(hook)
        self.head, self.mid, self.tail = None, None, None

    def _t(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def build(self, mem, win, prompt, full=None):
        """mem/win/full: token id lists (or None). mem may be ('soft', n) for soft tokens."""
        parts = []
        if full is not None:
            parts += [self._t("Here is a long story.\n\nThe story:\n"), full, self._t("\n\n")]
        else:
            if mem is not None or win is not None:
                parts += [self._t("You are reading a long story in pieces.\n\n")]
            if mem is not None:
                parts += [self._t("Notes kept from earlier parts of the story:\n"), mem, self._t("\n\n")]
            if win is not None:
                parts += [self._t("The most recent part of the story:\n"), win, self._t("\n\n")]
        parts += [self._t(prompt)]
        user = [x for p in parts for x in p]
        pre = self._t("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
        post = self._t("<|im_end|>\n<|im_start|>assistant\n")
        return pre + user + post

    @torch.no_grad()
    def score(self, ids, soft=None):
        x = torch.tensor([ids], device=self.device)
        if soft is not None:
            mask = (x[0] == -1)
            x = x.clone()
            x[0, mask] = self.ph
            self._soft = (mask, soft.to(self.device))
        try:
            lg = self.model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            self._soft = None
        return lg[self.lid].cpu(), len(ids)


def prompt_of(it):
    o = "\n".join(f"{G.LET[i]}. {t}" for i, t in enumerate(it["options"]))
    return f"Question: {it['question']}\n{o}\nAnswer with the letter of the correct option."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(G.FAMILIES))
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gstart", type=int, default=10_000)
    ap.add_argument("--past", type=int, default=3072)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--dists", default="0,1536", help="D values (tokens)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--early", type=int, default=40)
    ap.add_argument("--cert", action="store_true",
                    help="certification mode: stop a family after 15 groups if ceiling<40 or floor~ceiling")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    P, W, B = a.past, a.window, a.budget
    dists = [int(x) for x in a.dists.split(",")]
    host = Host(a.model)
    tok = host.tok
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["family"], r["group"], r["variant"], r["cond"]))
    fo = open(a.out, "a")
    t0, nf = time.time(), 0
    grp_hist = defaultdict(list)
    stopped, dead = set(), set()

    def emit(it, cond, lg, ntok, mtok, extra=None):
        nonlocal nf
        pred = int(lg.argmax())
        r = {"family": it["family"], "kind": it["kind"], "group": it["group"], "variant": it["variant"],
             "answer": it["answer"], "cond": cond, "pred": pred, "correct": int(pred == it["answer"]),
             "lg": [round(x, 3) for x in lg.tolist()], "ntok": ntok, "mem_tok": mtok, "gen": GEN_MD5}
        if extra:
            r.update(extra)
        fo.write(json.dumps(r) + "\n")
        nf += 1
        return r["correct"]

    for g in range(a.gstart, a.gstart + a.groups):
        for fam in a.families.split(","):
            if fam in dead:
                continue
            items = G.make_group(fam, g, "eval")
            ok_grp = defaultdict(list)

            def need(it, c):
                if only is not None and c.split("@")[0] not in only and c not in only:
                    return False
                if (fam, c) in stopped:
                    return False
                return (fam, it["group"], it["variant"], c) not in done

            for D in dists:
                lays = G.layout(items, tok, P=P, W=W, D=D)
                # leak gate: identical window and length across variants
                assert all(l["window"] == lays[0]["window"] and len(l["past"]) == P for l in lays)
                win = lays[0]["window"]
                for v, it in enumerate(items):
                    L = lays[v]
                    p = prompt_of(it)
                    Dx = L["D"]
                    if D == dists[0]:
                        orc = [x for s in L["slot_ids"] for x in s]
                        if need(it, "oracle"):
                            lg, nt = host.score(host.build(orc, win, p))
                            ok_grp["oracle"].append(emit(it, "oracle", lg, nt, len(orc)))
                        if need(it, "other"):
                            dv = (v + 1) % 4
                            don = [x for s in lays[dv]["slot_ids"] for x in s]
                            lg, nt = host.score(host.build(don, win, p))
                            ok_grp["other"].append(emit(it, "other", lg, nt, len(don), {
                                "donor_answer": items[dv]["answer"],
                                "donor_hit": int(int(lg.argmax()) == items[dv]["answer"])}))
                        if need(it, "empty"):
                            lg, nt = host.score(host.build(None, win, p))
                            ok_grp["empty"].append(emit(it, "empty", lg, nt, 0))
                        if need(it, "blind"):
                            lg, nt = host.score(host.build(None, None, p))
                            ok_grp["blind"].append(emit(it, "blind", lg, nt, 0))
                    ex = {"D": Dx, "Dnom": D, "span": L["span_len"]}
                    c = f"full@{D}"
                    if need(it, c):
                        lg, nt = host.score(host.build(None, None, p, full=L["past"] + win))
                        ok_grp[c].append(emit(it, c, lg, nt, P, ex))
                    c = f"fifo@{D}"
                    if need(it, c):
                        mem = L["past"][-B:]
                        a0 = P - B
                        cov = sum(1 for s, e in L["fact_spans"] if s >= a0) / len(L["fact_spans"])
                        lg, nt = host.score(host.build(mem, win, p))
                        ok_grp[c].append(emit(it, c, lg, nt, B, dict(ex, cov=cov)))
                    c = f"key@{D}"
                    if need(it, c):
                        sb = L["sent_bounds"]
                        k = max(1, -(-P // B))
                        ph = random.Random(f"phase-{fam}-{g}-{D}").randrange(k)
                        idx = list(range(len(sb) - 1 - ph, -1, -k))
                        mem, picked = [], []
                        for i in idx:   # newest first, until budget
                            s, e, isf = sb[i]
                            if len(mem) + (e - s) > B:
                                break
                            mem = L["past"][s:e] + mem
                            picked.append(isf)
                        cov = sum(picked) / len(L["fact_spans"])
                        lg, nt = host.score(host.build(mem, win, p))
                        ok_grp[c].append(emit(it, c, lg, nt, len(mem), dict(ex, cov=cov, phase=ph)))
                    c = f"mean@{D}"
                    if need(it, c):
                        e = host.emb.weight
                        ids = torch.tensor(L["past"], device=e.device)
                        nseg = P // B
                        soft = e[ids[:nseg * B]].float().view(nseg, B, -1).mean(0)
                        lg, nt = host.score(host.build([-1] * B, win, p), soft=soft)
                        ok_grp[c].append(emit(it, c, lg, nt, B, ex))
                    c = f"lex@{D}"
                    if need(it, c):
                        txt = tok.decode(L["past"])
                        words = sorted(set(w for w in re.findall(r"[A-Za-z0-9\-']+", txt)
                                           if w.lower() not in G.FILLER_VOCAB and w.lower() not in STOP))
                        mem = tok.encode(" " + ", ".join(words), add_special_tokens=False)[:B]
                        lg, nt = host.score(host.build(mem, win, p))
                        ok_grp[c].append(emit(it, c, lg, nt, len(mem), ex))
                fo.flush()
            # group-level bookkeeping + early stopping
            for c, vv in ok_grp.items():
                if len(vv) == len(items):
                    grp_hist[(fam, c)].append(int(all(vv)))
            for (f, c), h in list(grp_hist.items()):
                if f != fam or (f, c) in stopped:
                    continue
                if len(h) >= a.early and c.split("@")[0] in ("oracle", "other", "empty", "blind") or (
                        len(h) >= a.early and c.startswith("fifo") and c != "fifo@0" and sum(h) == 0):
                    m = 100 * sum(h) / len(h)
                    if m >= 95 or m == 0:
                        stopped.add((f, c))
                        print(f"  early-stop {f} {c}: group acc {m:.0f} over {len(h)} groups", flush=True)
            if a.cert:
                h_or = grp_hist.get((fam, "oracle"), [])
                if len(h_or) == 15:
                    ceil = 100 * sum(h_or) / 15
                    floor = max(100 * sum(grp_hist.get((fam, c), [0])) / max(1, len(grp_hist.get((fam, c), [1])))
                                for c in ("blind", "empty", "other"))
                    if ceil < 40 or floor >= ceil - 10:
                        dead.add(fam)
                        print(f"  CERT-ABANDON {fam}: oracle group {ceil:.0f}, floor {floor:.0f}", flush=True)
            print(f"{fam} g{g} done [{nf} rows, {(time.time() - t0) / 60:.1f} min]", flush=True)
    print("LADDER-TEXT-FINI", flush=True)


if __name__ == "__main__":
    main()
