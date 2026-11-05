"""03/11 write-selection diagnostic (train_b30_sel.py): WHICH past tokens to write, at the oracle's budget.
Pure python, no model: the rules rand / reref / nll / nllsent pick positions of the past, train_b30_sel.py writes
them exactly as the sup / orc oracle writes its tokens (one by one, layer-24 states of the full prefill)."""
import math
import random
import zlib

# reref only: on top of G.STOP, frequent words that would tag everything
EXTRA_STOP = set("""i'm it's he's she's don't didn't not no yes so but if as do did does have has had will would can could
just also very what who where when which how why up down out over than too here now all any one two us we our your
its were what's that's then there's off back still about get got go went put""".split())
SENT_END = (".", "!", "?", "\n")


class Words:
    """token id -> normalised word (decoded, stripped, lower case), cached across records."""

    def __init__(self, tok):
        self.tok, self.c = tok, {}

    def __call__(self, t):
        w = self.c.get(t)
        if w is None:
            w = self.c[t] = self.tok.decode([t])
        return w

    def norm(self, t):
        return self(t).strip().lower()


def candidates(n, mask=None):
    """positions 1..n-1 (the first token has no host prediction), restricted to mask (content tokens) if given."""
    return [p for p in range(1, n) if mask is None or mask[p]]


def sel_rand(cands, k, key):
    """uniform draw of k candidates, seed fixed per record (crc32 of the record key)."""
    rng = random.Random(zlib.crc32(str(key).encode()))
    return sorted(rng.sample(cands, min(k, len(cands))))


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def reref_counts(words, ctx, cands, stop):
    """retroactive tagging: for each candidate p, the number of LATER positions of ctx (rest of the past + window)
    holding the same normalised word; stopwords and punctuation never count."""
    cs, seen, out = set(cands), {}, {}
    for q in range(len(ctx) - 1, -1, -1):
        w = words.norm(ctx[q])
        good = bool(w) and any(ch.isalnum() for ch in w) and w not in stop
        if q in cs:
            out[q] = seen.get(w, 0) if good else 0
        if good:
            seen[w] = seen.get(w, 0) + 1
    return out


def sel_reref(words, ctx, cands, k, stop, nll=None):
    """k candidates with the most later mentions; ties by host NLL (if given) else the later position first.
    Below k tagged tokens the rest is filled in the same order (count 0)."""
    cnt = reref_counts(words, ctx, cands, stop)
    tie = (lambda p: nll[p] if _ok(nll[p]) else -1e9) if nll is not None else (lambda p: p)
    rk = sorted(cands, key=lambda p: (-cnt[p], -tie(p)))
    return sorted(rk[:k]), sum(c > 0 for c in cnt.values()), cnt


def word_groups(words, ids):
    """word index per position: a word starts at a token with leading whitespace or a non alphanumeric token."""
    g, cur = [], -1
    for j, t in enumerate(ids):
        s = words(t)
        if j == 0 or not s or s[0].isspace() or not s[0].isalnum():
            cur += 1
        g.append(cur)
    return g


def sel_nll(nll, cands, k, groups=None):
    """k candidates of highest host NLL; with groups, a token scores the max NLL of its word (word level)."""
    sc = {p: nll[p] if _ok(nll[p]) else -1e9 for p in cands}
    if groups is not None:
        wm = {}
        for p in range(1, len(nll)):
            if _ok(nll[p]):
                wm[groups[p]] = max(wm.get(groups[p], -1e9), nll[p])
        sc = {p: wm.get(groups[p], -1e9) for p in cands}
    rk = sorted(cands, key=lambda p: (-sc[p], -p))
    return sorted(rk[:k])


def sentences(words, ids):
    """[(s, e)] sentence spans: a sentence ends after a token holding . ! ? or a newline."""
    out, s0 = [], 0
    for j, t in enumerate(ids):
        if any(c in words(t) for c in SENT_END):
            out.append((s0, j + 1)); s0 = j + 1
    if s0 < len(ids):
        out.append((s0, len(ids)))
    return out


def sel_nllsent(words, past, nll, cands, k):
    """whole sentences by the mean host NLL of their candidate tokens, their candidates written until the budget;
    the last sentence is cut to its highest NLL tokens so that exactly k tokens are written."""
    cs = set(cands)
    sents = []
    for s, e in sentences(words, past):
        cp = [p for p in range(s, e) if p in cs and _ok(nll[p])]
        if cp:
            sents.append((sum(nll[p] for p in cp) / len(cp), cp))
    out = []
    for _, cp in sorted(sents, key=lambda x: -x[0]):
        rem = k - len(out)
        if rem <= 0:
            break
        out += cp if len(cp) <= rem else sorted(cp, key=lambda p: -nll[p])[:rem]
    return sorted(out)


def overlap(sel, ref):
    s, r = set(sel), set(ref)
    i = len(s & r)
    return {"n": len(s), "p": round(i / max(1, len(s)), 4), "r": round(i / max(1, len(r)), 4)}


def sel_sent(words, past, cands, k, score=None, key=None):
    """03/11 --sel-unit sent: whole sentences (all their candidate tokens) until k tokens; sentences ranked by the
    mean token score (reref count, NLL), or in random order (score None, seed from key); the last sentence is cut
    to its best tokens (random ones without score) so that exactly k tokens are written."""
    cs = set(cands)
    sents = [[p for p in range(s, e) if p in cs] for s, e in sentences(words, past)]
    sents = [cp for cp in sents if cp]
    if score is None:
        rng = random.Random(zlib.crc32(str(key).encode()))
        rng.shuffle(sents)
        tsc = {p: rng.random() for cp in sents for p in cp}
    else:
        tsc = {p: score[p] if _ok(score[p]) else -1e9 for cp in sents for p in cp}
        sents = sorted(sents, key=lambda cp: (-sum(tsc[p] for p in cp) / len(cp), -cp[-1]))
    out = []
    for cp in sents:
        rem = k - len(out)
        if rem <= 0:
            break
        out += cp if len(cp) <= rem else sorted(cp, key=lambda p: -tsc[p])[:rem]
    return sorted(out)
