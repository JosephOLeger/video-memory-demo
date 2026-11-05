"""v2 ladder: joint stories (8 queried facts per story), same-type distractors, D up to ~6000 tokens.

One context serves the 8 questions of a story through a KV cache of the shared prefix (cropped back
after each question). Memories at budget B tokens:
    full@D      whole past + window (attention ceiling, P + W tokens)
    ep          the whole FACT episode (facts + in-episode distractors + scenery), ~1.2k tokens > B
    oracle      the family's own fact sentences
    other       the family's fact sentences from ANOTHER variant (donor)
    sel_best    passage cache with perfect selection: own fact sentences + random other event
                sentences up to B, in story order (upper bound of a selector that knew the question)
    sel_rand@D  random event sentences (facts or distractors, never scenery) up to B, story order
    evfifo@D    the most recent event sentences up to B ("keep whatever is not scenery")
    fifo@D      last B tokens;  key@D  one sentence every k;  mean@D  per-position mean of P/B segments
    lex@D       sorted set of content words absent from the scenery vocabulary (no order, no binding)
    empty       window only;  blind  question only

    CUDA_VISIBLE_DEVICES=3 python ladder_text2.py --groups 1 --out runs2/smoke.jsonl
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

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_story as G1  # noqa: E402
import importlib  # noqa: E402
G = importlib.import_module(os.environ.get("TEXT_GEN", "gen_story2"))  # gen_story3: cumulative families
from ladder_text import Host, STOP  # noqa: E402

GEN_MD5 = hashlib.md5(open(G.__file__, "rb").read()).hexdigest()[:8]
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"


def prompt_of(it):
    o = "\n".join(f"{G.LET[i]}. {t}" for i, t in enumerate(it["options"]))
    return f"Question: {it['question']}\n{o}\nAnswer with the letter of the correct option."


class Host2(Host):
    def ctx_ids(self, mem, win, full=None):
        t = self._t
        parts = [t(SYS)]
        if full is not None:
            parts += [t("Here is a long story.\n\nThe story:\n"), full, t("\n\n")]
        else:
            if mem is not None or win is not None:
                parts += [t("You are reading a long story in pieces.\n\n")]
            if mem is not None:
                parts += [t("Notes kept from earlier parts of the story:\n"), mem, t("\n\n")]
            if win is not None:
                parts += [t("The most recent part of the story:\n"), win, t("\n\n")]
        return [x for p in parts for x in p]

    @torch.no_grad()
    def score_many(self, ctx, prompts, soft=None):
        """ctx ids (may contain -1 for soft slots) shared by several prompts. -> [letter logits]"""
        from transformers import DynamicCache
        x = torch.tensor([ctx], device=self.device)
        if soft is not None:
            mask = x[0] == -1
            x = x.clone(); x[0, mask] = self.ph
            self._soft = (mask, soft.to(self.device))
        cache = DynamicCache()
        try:
            self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        finally:
            self._soft = None
        n = x.shape[1]
        out = []

        def run(ids, keep):
            s = torch.tensor([ids], device=self.device)
            pos = torch.arange(n, n + s.shape[1], device=self.device)
            lg = self.model(input_ids=s, past_key_values=cache, use_cache=True, cache_position=pos,
                            attention_mask=torch.ones(1, n + s.shape[1], device=self.device, dtype=torch.long),
                            logits_to_keep=keep).logits[0].float()
            cache.crop(n)
            return lg
        for p in prompts:
            if isinstance(p, str):                   # letter MCQ only (legacy)
                out.append(run(self._t(p) + self._t(END), 1)[-1][self.lid].cpu())
                continue
            # item: letter MCQ logits [4] + option log-likelihood in the free-answer format [4]
            # (same prompt as the trained student: "Answer with a few words.", option capitalised)
            lg4 = run(self._t(prompt_of(p)) + self._t(END), 1)[-1][self.lid].cpu()
            q = self._t(f"Question: {p['question']}\nAnswer with a few words." + END)
            lk = []
            for o in p["options"]:
                at = self._t(o[0].upper() + o[1:])
                lg = run(q + at, len(at) + 1)[:-1]               # positions predicting the option tokens
                lk.append(float(torch.log_softmax(lg, -1).gather(1, torch.tensor(at, device=lg.device)[:, None]).sum()))
            out.append(torch.cat([lg4, torch.tensor(lk)]))
        return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gstart", type=int, default=10_000)
    ap.add_argument("--past", type=int, default=8192)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--dists", default="0,3000,6000")
    ap.add_argument("--only", default=None)
    ap.add_argument("--check-cache", action="store_true", help="compare cached vs uncached logits and exit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    P, W, B = a.past, a.window, a.budget
    dists = [int(x) for x in a.dists.split(",")]
    host = Host2()
    tok = host.tok
    fams = G.FAMILIES
    if a.check_cache:
        st = G.make_story(a.gstart)
        L = G.layout(st, tok, P=P, W=W, D=dists[0])[0]
        its = [st["items"][f][0] for f in fams]
        for name, ctx in (("full", host.ctx_ids(None, None, full=L["past"] + L["window"])),
                          ("fifo", host.ctx_ids(L["past"][-B:], L["window"]))):
            cached, _ = host.score_many(ctx, [prompt_of(it) for it in its])
            for it, c in zip(its, cached):
                lg, _ = host.score(ctx + host._t(prompt_of(it)) + host._t(END))
                print(name, it["family"], [round(float(x), 2) for x in c], [round(float(x), 2) for x in lg],
                      "maxdiff %.3f" % float((c - lg).abs().max()))
        return
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["group"], r["variant"], r["cond"]))
    fo = open(a.out, "a")
    t0, nf = time.time(), 0

    def need(g, v, c):
        if only is not None and c.split("@")[0] not in only:
            return False
        return (g, v, c) not in done

    def emit(its, cond, lgs, ntok, mtok, extra=None):
        nonlocal nf
        for it, lg_all in zip(its, lgs):
            lg, lk = lg_all[:4], (lg_all[4:] if lg_all.numel() > 4 else None)
            pred = int(lg.argmax())
            r = {"family": it["family"], "kind": it["kind"], "group": it["group"], "variant": it["variant"],
                 "answer": it["answer"], "cond": cond, "pred": pred, "correct": int(pred == it["answer"]),
                 "lg": [round(x, 3) for x in lg.tolist()], "ntok": ntok, "mem_tok": mtok, "gen": GEN_MD5}
            if lk is not None:
                r["lk"] = [round(x, 3) for x in lk.tolist()]
            if extra:
                r.update(extra(it) if callable(extra) else extra)
            fo.write(json.dumps(r) + "\n")
            nf += 1

    def sel(L, rng, keep_fam=None):
        """event sentences up to B: own family's facts first if keep_fam, then random others; story order."""
        ev = [(s, e, k) for s, e, k in L["bounds"] if k != "scen"]
        chosen, tot = [], 0
        if keep_fam:
            for s, e, k in ev:
                if k == keep_fam:
                    chosen.append((s, e)); tot += e - s
        rest = [(s, e) for s, e, k in ev if (s, e) not in chosen]
        rng.shuffle(rest)
        for s, e in rest:
            if tot + e - s <= B:
                chosen.append((s, e)); tot += e - s
        chosen.sort()
        return [x for s, e in chosen for x in L["past"][s:e]], chosen

    for g in range(a.gstart, a.gstart + a.groups):
        st = G.make_story(g, "eval")
        for D in dists:
            lays = G.layout(st, tok, P=P, W=W, D=D)
            assert all(l["window"] == lays[0]["window"] and len(l["past"]) == P for l in lays)
            win = lays[0]["window"]
            for v in range(4):
                L = lays[v]
                its = [st["items"][f][v] for f in fams]
                prs = [prompt_of(it) for it in its]
                ex = {"D": L["D"], "Dnom": D}
                if D == dists[0]:
                    if need(g, v, "empty"):
                        lgs, n = host.score_many(host.ctx_ids(None, win), its)
                        emit(its, "empty", lgs, n, 0)
                    if need(g, v, "blind"):
                        lgs, n = host.score_many(host.ctx_ids(None, None), its)
                        emit(its, "blind", lgs, n, 0)
                    if need(g, v, "ep"):
                        a0, b0 = L["ep"]
                        lgs, n = host.score_many(host.ctx_ids(L["past"][a0:b0], win), its)
                        emit(its, "ep", lgs, n, b0 - a0)
                    for c in ("oracle", "other", "sel_best"):
                        if not need(g, v, c):
                            continue
                        for it, pr in zip(its, prs):
                            f = it["family"]
                            if c == "oracle":
                                mem = [x for s, e in G.facts_of(L, f) for x in L["past"][s:e]]
                                extra = None
                            elif c == "other":
                                dv = (v + 1) % 4
                                Ld = lays[dv]
                                mem = [x for s, e in G.facts_of(Ld, f) for x in Ld["past"][s:e]]
                                dans = st["items"][f][dv]["answer"]
                                extra = None
                            else:
                                mem, _ = sel(L, random.Random(f"best-{g}-{f}"), keep_fam=f)
                            lgs, n = host.score_many(host.ctx_ids(mem, win), [it])
                            if c == "other":
                                emit([it], c, lgs, n, len(mem), {"donor_answer": dans,
                                                                  "donor_hit": int(int(lgs[0].argmax()) == dans)})
                            else:
                                emit([it], c, lgs, n, len(mem))
                c = f"full@{D}"
                if need(g, v, c):
                    lgs, n = host.score_many(host.ctx_ids(None, None, full=L["past"] + win), its)
                    emit(its, c, lgs, n, P, ex)
                c = f"fifo@{D}"
                if need(g, v, c):
                    mem = L["past"][-B:]
                    cov = lambda it: {"cov": sum(e - s for s, e in G.facts_of(L, it["family"]) if s >= P - B) /  # noqa
                                      max(1, sum(e - s for s, e in G.facts_of(L, it["family"])))}
                    lgs, n = host.score_many(host.ctx_ids(mem, win), its)
                    emit(its, c, lgs, n, B, lambda it: dict(ex, **cov(it)))
                c = f"evfifo@{D}"
                if need(g, v, c):
                    ev = [(s, e) for s, e, k in L["bounds"] if k != "scen"]
                    chosen, tot = [], 0
                    for s, e in reversed(ev):
                        if tot + e - s > B:
                            break
                        chosen.insert(0, (s, e)); tot += e - s
                    mem = [x for s, e in chosen for x in L["past"][s:e]]
                    lgs, n = host.score_many(host.ctx_ids(mem, win), its)
                    emit(its, c, lgs, n, len(mem), ex)
                c = f"sel_rand@{D}"
                if need(g, v, c):
                    mem, ch = sel(L, random.Random(f"rand-{g}-{D}"))
                    chs = set(ch)
                    covf = lambda it: {"cov": sum(1 for x in G.facts_of(L, it["family"]) if x in chs) /  # noqa
                                       len(G.facts_of(L, it["family"]))}
                    lgs, n = host.score_many(host.ctx_ids(mem, win), its)
                    emit(its, c, lgs, n, len(mem), lambda it: dict(ex, **covf(it)))
                c = f"key@{D}"
                if need(g, v, c):
                    sb = L["bounds"]
                    k = max(1, -(-P // B))
                    ph = random.Random(f"phase-{g}-{D}").randrange(k)
                    mem, chosen = [], set()
                    for i in range(len(sb) - 1 - ph, -1, -k):
                        s, e, _ = sb[i]
                        if len(mem) + (e - s) > B:
                            break
                        mem = L["past"][s:e] + mem
                        chosen.add((s, e))
                    covf = lambda it: {"cov": sum(1 for x in G.facts_of(L, it["family"]) if x in chosen) /  # noqa
                                       len(G.facts_of(L, it["family"]))}
                    lgs, n = host.score_many(host.ctx_ids(mem, win), its)
                    emit(its, c, lgs, n, len(mem), lambda it: dict(ex, **covf(it)))
                c = f"mean@{D}"
                if need(g, v, c):
                    e_ = host.emb.weight
                    ids = torch.tensor(L["past"], device=e_.device)
                    nseg = P // B
                    soft = e_[ids[-nseg * B:]].float().view(nseg, B, -1).mean(0)
                    lgs, n = host.score_many(host.ctx_ids([-1] * B, win), its, soft=soft)
                    emit(its, c, lgs, n, B, ex)
                c = f"lex@{D}"
                if need(g, v, c):
                    txt = tok.decode(L["past"])
                    words = sorted(set(w for w in re.findall(r"[A-Za-z0-9\-']+", txt)
                                       if w.lower() not in G1.FILLER_VOCAB and w.lower() not in STOP))
                    mem = tok.encode(" " + ", ".join(words), add_special_tokens=False)[:B]
                    lgs, n = host.score_many(host.ctx_ids(mem, win), its)
                    emit(its, c, lgs, n, len(mem), dict(ex, lex_words=len(words)))
            fo.flush()
        print(f"g{g} done [{nf} rows, {(time.time() - t0) / 60:.1f} min]", flush=True)
    print("LADDER-TEXT2-FINI", flush=True)


if __name__ == "__main__":
    main()
