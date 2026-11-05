"""BABILong-like TRAINING stories (26/10): bAbI facts (TRAIN split) hidden in PG19 books (TRAIN split).

Same API as gen_story3 (make_story, layout -> past/window/bounds, facts_of, FAMILIES, TRAIN_FAMS), so the
trainer can use it through TEXT_GEN=gen_babilong. One bAbI sample (facts + question) per story; the task
cycles with the group index (balanced). Families = tasks.

Insertion (as in booydar/babilong, babilong_utils.NoiseInjectionDataset): background = consecutive
sentences of a random PG19 book from a random offset; the facts are put, in order, at sentence boundaries
drawn uniformly with replacement and sorted. Differences, all deliberate:
  * we KNOW the fact positions (bounds), for the oracle write and the teacher notes;
  * every fact lies in the PAST (before the window of W tokens), at least D tokens before it: the window is
    then identical across variants and the answer is never readable from the window;
  * regex sentence splitter (no nltk on the tower), whitespace collapsed.

Counterfactual variants (4 per story, same background, same boundaries, only the facts differ):
  qa1-qa4  location relabelling: the answer location A is swapped with B_v (v = 1..3) everywhere in the
           facts (a bijection, so the story stays consistent); locations named in the question are fixed.
           PRESENCE-BALANCED: the B_v are locations that occur in the facts, so all 4 options occur in every
           variant and "which locations were mentioned" does not give the answer (qa4 exception, see
           loc_variants). Every location is one Qwen token: same token counts.
  qa7      counting: the actor of a few possession sentences (got/dropped/gave...) is reassigned among the
           bAbI people, minimal edit, re-simulated for validity, so that the queried person carries
           none / one / two / three objects in the 4 variants. Names are one token: same token counts.

Splits: bAbI en-10k train, story id % 10 != 0 -> "train", == 0 -> "eval" (dev); samples equal to any bAbI
TEST sample (en and en-10k test files, which the official BABILong test is built from) are removed.
PG19: 600 books of train shard 0 for "train", the validation split for "eval". The official BABILong test is NEVER used
here (evaluation only, see train_bl.py --official).
"""
from __future__ import annotations

import functools
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BL = os.path.join(HERE, "bl_data")
BABI = os.path.join(BL, "babi_raw", "tasks_1-20_v1-2")
PG = {"train": os.path.join(BL, "pg19_raw", "pg19_train.jsonl"),          # first 600 books (> 20 k chars) of the
      "eval": os.path.join(BL, "pg19_raw", "pg19_validation.jsonl")}     # emozilla/pg19 train shard 0; validation
SALT = "babilong-like-v1"
TASKS = os.environ.get("BL_TASKS", "qa1,qa2,qa3,qa4,qa7").split(",")
FAMILIES = TASKS
GIST = TASKS
DETAIL = []
TRAIN_FAMS = {"A": TASKS, "B": TASKS}
HELD_OUT = None
FILES = {"qa1": "qa1_single-supporting-fact", "qa2": "qa2_two-supporting-facts", "qa3": "qa3_three-supporting-facts",
         "qa4": "qa4_two-arg-relations", "qa5": "qa5_three-arg-relations", "qa6": "qa6_yes-no-questions",
         "qa7": "qa7_counting", "qa8": "qa8_lists-sets"}
LOCS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
PEOPLE = ["Mary", "Sandra", "John", "Daniel"]
COUNT_W = ["none", "one", "two", "three"]
MAX_FACT_FRAC = 0.4          # layout refuses facts above this share of the past (the trainer skips the step)
MAX_FACTS = 100              # longer bAbI samples (9 % of qa3, none elsewhere) are not drawn: they fit P = 2048
MAX_SENT_TOK = 160           # very long background "sentences" are cut in pieces (insertion granularity)


# ----------------------------------------------------------------------------------------------- bAbI
def parse_babi(path):
    """bAbI file -> one sample per question: facts before it (questions dropped), question, answer, support."""
    out, sid, lines = [], -1, []
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        n, rest = line.split(" ", 1)
        if int(n) == 1:
            sid += 1
            lines = []
        if "\t" in rest:
            q, ans, sup = rest.split("\t")
            nums = [k for k, _ in lines]
            out.append(dict(sid=sid, facts=[t for _, t in lines], question=q.strip(), answer=ans.strip(),
                            support=[nums.index(int(s)) for s in sup.split()]))
        else:
            lines.append((int(n), rest.strip()))
    return out


@functools.lru_cache(maxsize=None)
def test_keys():
    keys = set()
    for sub in ("en", "en-10k"):
        for t, f in FILES.items():
            p = os.path.join(BABI, sub, f + "_test.txt")
            if os.path.exists(p):
                for s in parse_babi(p):
                    keys.add((t, tuple(s["facts"]), s["question"]))
    return keys


@functools.lru_cache(maxsize=None)
def pool(task, split):
    raw = parse_babi(os.path.join(BABI, "en-10k", FILES[task] + "_train.txt"))
    tk = test_keys()
    out = [dict(s, task=task) for s in raw
           if (s["sid"] % 10 == 0) == (split == "eval") and (task, tuple(s["facts"]), s["question"]) not in tk]
    return out


def dedupe_report():
    rep = {}
    for t in FILES:
        raw = parse_babi(os.path.join(BABI, "en-10k", FILES[t] + "_train.txt"))
        tk = test_keys()
        rep[t] = dict(n_train_samples=len(raw), equal_to_a_test_sample=sum((t, tuple(s["facts"]), s["question"]) in tk
                                                                           for s in raw))
    return rep


# ------------------------------------------------------------------------------------- counterfactuals
def _remap(s, mp):
    """word-level relabelling of locations by the bijection mp (dict, missing keys unchanged)."""
    return re.sub(r"\b(" + "|".join(LOCS) + r")\b", lambda m: mp.get(m.group(1), m.group(1)), s)


def loc_variants(rng, ex):
    """PRESENCE-BALANCED relabelling: the 4 options are locations that all occur in the facts of every variant
    (transpositions A <-> B_v among the PRESENT locations), so knowing which locations were mentioned does
    not give the answer. qa4 (2 facts, 2 non-question locations) cannot: there the options are the two present
    locations {A, C} and two absent ones {X, Y}, variants (A,C) (C,A) (X,Y) (Y,X): a presence-only memory is
    at 50 % per question and 25 % per group (reported as the qa4 presence floor)."""
    A = ex["answer"]
    if A not in LOCS:
        return None
    qlocs = {l for l in LOCS if re.search(rf"\b{l}\b", ex["question"])}
    present = {l for l in LOCS if any(re.search(rf"\b{l}\b", f) for f in ex["facts"])}
    cand = [l for l in LOCS if l != A and l not in qlocs and l in present]
    if len(cand) >= 3:
        Bs = rng.sample(cand, 3)
        maps = [{}] + [{A: B, B: A} for B in Bs]
        answers = [A] + Bs
    elif ex.get("task") == "qa4" and len(cand) >= 1:
        C = rng.choice(cand)
        absent = [l for l in LOCS if l not in present and l not in qlocs]
        if len(absent) < 2:
            return None
        X, Y = rng.sample(absent, 2)
        maps = [{}, {A: C, C: A}, {A: X, X: A, C: Y, Y: C}, {A: Y, Y: A, C: X, X: C}]
        answers = [A, C, X, Y]
    else:
        return None
    return [[_remap(f, mp) for f in ex["facts"]] for mp in maps], answers


GET = re.compile(r"^(\w+) (got|took|picked up|grabbed) the (\w+)( there)?\.$")
DROP = re.compile(r"^(\w+) (dropped|discarded|put down|left) the (\w+)( there)?\.$")
GIVE = re.compile(r"^(\w+) (gave|handed|passed) the (\w+) to (\w+)\.$")


def simulate(facts):
    """holdings per person; None if a fact is impossible (drop / give of an object not held, get of a held one)."""
    h = {}
    holder = {}
    for f in facts:
        m = GIVE.match(f)
        if m:
            a, _, o, b = m.groups()
            if holder.get(o) != a or a == b:
                return None
            h[a].discard(o); h.setdefault(b, set()).add(o); holder[o] = b
            continue
        m = GET.match(f)
        if m:
            a, _, o, _ = m.groups()
            if o in holder:
                return None
            h.setdefault(a, set()).add(o); holder[o] = a
            continue
        m = DROP.match(f)
        if m:
            a, _, o, _ = m.groups()
            if holder.get(o) != a:
                return None
            h[a].discard(o); del holder[o]
    return h


def _set_actor(f, new):
    m = GIVE.match(f)
    if m:
        return f"{new} {m.group(2)} the {m.group(3)} to {m.group(4)}." if new != m.group(4) else None
    return re.sub(r"^\w+", new, f)


def count_variants(rng, ex, tries=4000):
    """qa7: minimal actor reassignments of possession sentences so that the queried person carries
    none / one / two / three objects (one variant each; variant 0 is the original story)."""
    who = re.match(r"How many objects is (\w+) carrying\?", ex["question"]).group(1)
    base = simulate(ex["facts"])
    if base is None:
        return None
    c0 = len(base.get(who, ()))
    if c0 > 3 or COUNT_W[c0] != ex["answer"]:
        return None
    poss = [i for i, f in enumerate(ex["facts"]) if GET.match(f) or DROP.match(f) or GIVE.match(f)]
    if not poss:
        return None
    best = {c0: (0, list(ex["facts"]))}
    for t in range(tries):
        k = 1 + min(len(poss) - 1, int(rng.expovariate(1.0)))
        fs = list(ex["facts"])
        for i in rng.sample(poss, k):
            new = rng.choice([p for p in PEOPLE if p != fs[i].split()[0]])
            nf = _set_actor(fs[i], new)
            if nf is None:
                break
            fs[i] = nf
        else:
            h = simulate(fs)
            if h is None:
                continue
            c = len(h.get(who, ()))
            if c <= 3 and (c not in best or best[c][0] > k):
                best[c] = (k, fs)
        if len(best) == 4 and t > 200:
            break
    if len(best) < 4:
        return None
    order = [c0] + [c for c in range(4) if c != c0]
    return [best[c][1] for c in order], [COUNT_W[c] for c in order]


def count_support(facts, who):
    """qa7 supporting facts of a (possibly edited) variant: the possession sentences involving the person."""
    out = []
    for i, f in enumerate(facts):
        m = GIVE.match(f)
        if (m and who in (m.group(1), m.group(4))) or ((GET.match(f) or DROP.match(f)) and f.split()[0] == who):
            out.append(i)
    return out


# ----------------------------------------------------------------------------------------- answers/prompts
def target_text(task, ex_question, label):
    """the official answer FORMAT of each task (BABILong prompts.py examples / post_prompt)."""
    q = ex_question
    if task == "qa1":
        p = re.match(r"Where is (\w+)\?", q).group(1)
        return f"The most recent location of {p} is {label}."
    if task == "qa2":
        o = re.match(r"Where is the (\w+)\?", q).group(1)
        return f"The {o} is in the {label}."
    if task == "qa3":
        m = re.match(r"Where was the (\w+) before the (\w+)\?", q)
        return f"Before the {m.group(2)} the {m.group(1)} was in the {label}."
    return label


@functools.lru_cache(maxsize=None)
def _official():
    import sys
    sys.path.insert(0, os.path.join(HERE, "babilong_ext"))
    from prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE  # noqa: E402
    from metrics import compare_answers, TASK_LABELS  # noqa: E402
    return DEFAULT_PROMPTS, DEFAULT_TEMPLATE, compare_answers, TASK_LABELS


SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"


def prompt_parts(task, question):
    """official prompt split around the context: (prefix text, suffix text), chat template included.
    get_formatted_input(context, question, ...) = TEMPLATE.format(...).strip(): context sits between
    '<context>\\n' and '\\n</context>'."""
    PR, TPL, _, _ = _official()
    p = PR[task]
    full = TPL.format(instruction=p["instruction"], examples=p["examples"], post_prompt=p["post_prompt"],
                      context="\x00", question=question.strip()).strip()
    pre, post = full.split("\x00")
    return SYS + pre, post + END


def student_ids(tok, win, it):
    pre, post = prompt_parts(it["task"], it["question"])
    return tok.encode(pre, add_special_tokens=False) + list(win) + tok.encode(post, add_special_tokens=False)


def teacher_ids(tok, notes, win, it):
    """teacher: same prompt, the bAbI fact sentences (notes) put in the context before the window."""
    pre, post = prompt_parts(it["task"], it["question"])
    return tok.encode(pre, add_special_tokens=False) + list(notes) + list(win) + tok.encode(post, add_special_tokens=False)


def judge(it, output):
    _, _, compare_answers, TASK_LABELS = _official()
    return bool(compare_answers(it["answer_text"], output, it["question"], TASK_LABELS[it["task"]]))


# --------------------------------------------------------------------------------------------- stories
MOVE = re.compile(r"^(\w+) (?:went back to|went to|moved to|travelled to|journeyed to|went back|went) the (\w+)\.$")
EXTRA_Q = int(os.environ.get("BL_EXTRA_Q", "3"))   # extra questions per TRAIN story (simulated), 0 = off


def sim_locations(facts):
    """qa1/qa2 world: person -> (location, fact idx), object -> (location, support idx list)."""
    ploc, holder, oloc = {}, {}, {}
    for i, f in enumerate(facts):
        m = MOVE.match(f)
        if m:
            p_, l_ = m.groups()
            ploc[p_] = (l_, i)
            for o_, (h_, gi) in holder.items():
                if h_ == p_:
                    oloc[o_] = (l_, [gi, i])
            continue
        m = GET.match(f)
        if m:
            p_, _, o_, _ = m.groups()
            holder[o_] = (p_, i)
            if p_ in ploc:
                oloc[o_] = (ploc[p_][0], [ploc[p_][1], i])
            continue
        m = DROP.match(f)
        if m:
            p_, _, o_, _ = m.groups()
            holder.pop(o_, None)
            if p_ in ploc:
                oloc[o_] = (ploc[p_][0], [ploc[p_][1], i])
    return ploc, oloc


def extra_questions(task, facts4, main_q, rng):
    """simulated extra questions on the same facts (TRAIN only): (question, [answer per variant], [support per v])."""
    out = []
    if task in ("qa1", "qa2"):
        sims = [sim_locations(f) for f in facts4]
        people = sorted(set(sims[0][0]) & set(sims[1][0]) & set(sims[2][0]) & set(sims[3][0]))
        objs = sorted(set(sims[0][1]) & set(sims[1][1]) & set(sims[2][1]) & set(sims[3][1]))
        for p_ in people:
            q = f"Where is {p_}?"
            if q != main_q:
                out.append(("qa1", q, [s[0][p_][0] for s in sims], [[s[0][p_][1]] for s in sims]))
        if task == "qa2":
            for o_ in objs:
                q = f"Where is the {o_}?"
                if q != main_q:
                    out.append(("qa2", q, [s[1][o_][0] for s in sims], [s[1][o_][1] for s in sims]))
    elif task == "qa7":
        for p_ in PEOPLE:
            q = f"How many objects is {p_} carrying?"
            if q == main_q:
                continue
            hs = [simulate(f) for f in facts4]
            if any(h is None for h in hs) or not any(f.startswith(p_ + " ") for f in facts4[0]):
                continue
            cs = [len(h.get(p_, ())) for h in hs]
            if max(cs) <= 3:
                out.append(("qa7", q, [COUNT_W[c] for c in cs], [count_support(f, p_) for f in facts4]))
    rng.shuffle(out)
    return out[:EXTRA_Q]


def make_story(g: int, split: str = "eval"):
    rng = random.Random(f"{SALT}-{split}-{g}")
    task = TASKS[g % len(TASKS)]
    P_ = pool(task, split)
    for _ in range(1000):
        ex = P_[rng.randrange(len(P_))]
        if len(ex["facts"]) > MAX_FACTS:
            continue
        res = count_variants(rng, ex) if task == "qa7" else loc_variants(rng, ex)
        if res is not None:
            break
    facts, answers = res
    opts = list(answers)
    rng.shuffle(opts)
    items = {task: [dict(family=task, task=task, kind="gist", group=g, variant=v, split=split,
                         question=ex["question"], options=opts, answer=opts.index(answers[v]),
                         answer_text=answers[v], target_text=target_text(task, ex["question"], answers[v]),
                         option_texts=[target_text(task, ex["question"], o) for o in opts],
                         n_facts=len(ex["facts"]),
                         support=(count_support(facts[v], re.match(r"How many objects is (\w+)", ex["question"]).group(1))
                                  if task == "qa7" else ex["support"])) for v in range(4)]}
    if split == "train" and EXTRA_Q > 0:
        for j, (t2, q2, ans2, sup2) in enumerate(extra_questions(task, facts, ex["question"], rng)):
            items[f"{task}#{j}"] = [dict(family=f"{task}#{j}", task=t2, kind="extra", group=g, variant=v, split=split,
                                         question=q2, options=[], answer=-1, answer_text=ans2[v],
                                         target_text=target_text(t2, q2, ans2[v]), option_texts=[],
                                         n_facts=len(ex["facts"]), support=sup2[v],
                                         contrast=ans2[v] != ans2[(v + 1) % 4]) for v in range(4)]
    return dict(group=g, split=split, task=task, items=items, facts=facts, babi=ex,
                rng_seed=rng.randrange(1 << 30), bg_seed=rng.randrange(1 << 30))


@functools.lru_cache(maxsize=2)
def books(split):
    import json
    return [json.loads(x)["text"] for x in open(PG[split])]


_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z])|(?<=[.!?]["\')\]])\s+(?=["\'(\[]?[A-Z])')


def split_sents(text):
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def background(story, tok, need):
    """list of token lists (sentences of PG19, leading space as in BABILong), total >= need."""
    bk = books(story["split"])
    r = random.Random(story["bg_seed"])
    out, tot = [], 0
    while tot < need:
        text = bk[r.randrange(len(bk))]
        off = r.randrange(max(1, len(text) - 2000))
        sents = split_sents(text[off: off + max(4000, (need - tot) * 7)])[1:-1]
        for s in sents:
            ids = tok.encode(" " + s, add_special_tokens=False)
            for i in range(0, len(ids), MAX_SENT_TOK):
                out.append(ids[i:i + MAX_SENT_TOK])
                tot += len(out[-1])
            if tot >= need:
                break
    return out


def layout_official(story, tok, P=3400, W=256, margin=16):
    """28/10 (agent): OFFICIAL-like layout, as booydar/babilong: the facts go, in order, at sentence boundaries
    drawn uniformly (with replacement, sorted) over the WHOLE context (past + window), so a fact may fall inside
    the window and the last facts are often near the end. Total = P + W tokens; past = first P, window = last W.
    Differences with layout(): no distance D; the window may differ across variants (when it holds a fact);
    bounds keep only the PAST part of the facts (a fact cut by the past/window seam is kept clipped), in order,
    so facts_of()[i] is still fact i for i < n_fact_past. Extra keys: n_fact_past (facts fully in the past),
    n_fact_win (facts touching the window)."""
    task = story["task"]
    enc = lambda s: tok.encode(" " + s, add_special_tokens=False)  # noqa: E731
    fids = [[enc(f) for f in fv] for fv in story["facts"]]
    F = [sum(len(x) for x in fv) for fv in fids]
    if max(F) > MAX_FACT_FRAC * P:
        raise ValueError(f"facts {max(F)} tokens > {MAX_FACT_FRAC} x P {P}")
    T = P + W
    bg = background(story, tok, T - min(F) + 2 * margin)
    toks, starts = [], []
    for s in bg:
        starts.append(len(toks))
        toks += s
    hi = T - max(F)                                         # boundaries common to all variants
    elig = [b for b in starts if b <= hi] + ([hi] if hi not in starts else [])
    r = random.Random(story["rng_seed"] * 37 + P + 1)
    chosen = sorted(r.choice(elig) for _ in range(len(fids[0])))
    at = {}
    for i, b in enumerate(chosen):
        at.setdefault(b, []).append(i)
    out = []
    for v in range(4):
        nb = T - F[v]
        pts = sorted({0, nb} | {b for b in starts if 0 < b < nb} | {b for b in at if b <= nb})
        seq, bnd = [], []
        for a_, b_ in zip(pts, pts[1:] + [None]):
            for i in at.get(a_, []):
                q0 = len(seq)
                seq += fids[v][i]
                bnd.append((q0, len(seq), task))
            if b_ is not None and b_ > a_:
                q0 = len(seq)
                seq += toks[a_:b_]
                bnd.append((q0, len(seq), "scen"))
        assert len(seq) == T, (len(seq), T)
        past, window = seq[:P], seq[P:]
        bounds = [(a_, min(b_, P), k) for a_, b_, k in bnd if a_ < P]
        fall = [(a_, b_) for a_, b_, k in bnd if k == task]
        n_past = sum(1 for a_, b_ in fall if b_ <= P)
        fb = [(a_, b_) for a_, b_, k in bounds if k == task]
        sup = story["items"][task][v]["support"]
        se = [fb[i][1] - 1 for i in sup if i < len(fb)] or ([fb[-1][1] - 1] if fb else [P - 1])
        out.append(dict(past=past, window=window, bounds=bounds, D=P - fb[-1][1] if fb else P,
                        ep=(fb[0][0], fb[-1][1]) if fb else (0, 0), support_ends=se,
                        n_fact_past=n_past, n_fact_win=len(fall) - n_past))
    return out


def layout(story, tok, P=4096, W=256, D=0, margin=16, official=False):
    """4 variants: past (exactly P tokens: PG19 background with the facts at sentence boundaries, every fact at
    least ~D tokens before the end of the past), window (the W background tokens that follow, identical across
    variants), bounds [(a, b, kind)] with kind 'scen' (background piece) or the task (one fact sentence).
    official=True (28/10): layout_official() instead (facts anywhere, window included; D ignored)."""
    if official:
        return layout_official(story, tok, P=P, W=W, margin=margin)
    task = story["task"]
    enc = lambda s: tok.encode(" " + s, add_special_tokens=False)  # noqa: E731
    fids = [[enc(f) for f in fv] for fv in story["facts"]]
    F = [sum(len(x) for x in fv) for fv in fids]
    if max(F) > MAX_FACT_FRAC * P:
        raise ValueError(f"facts {max(F)} tokens > {MAX_FACT_FRAC} x P {P}")
    bg = background(story, tok, P - min(F) + W + 2 * margin)
    toks, starts = [], []
    for s in bg:
        starts.append(len(toks))
        toks += s
    end = len(toks) - W
    s_lo = end - (P - max(F)) + margin                     # region common to all variants
    elig = [b for b in starts if s_lo <= b <= end - D]
    if not elig:
        elig = [b for b in starts if s_lo <= b < end] or [end]
    r = random.Random(story["rng_seed"] * 31 + P)
    chosen = sorted(r.choice(elig) for _ in range(len(fids[0])))
    at = {}
    for i, b in enumerate(chosen):
        at.setdefault(b, []).append(i)
    out = []
    for v in range(4):
        s0 = end - (P - F[v])
        pts = [s0] + [b for b in starts if s0 < b < end] + [end]
        past, bounds = [], []
        for a_, b_ in zip(pts, pts[1:]):
            for i in at.get(a_, []):
                q0 = len(past)
                past += fids[v][i]
                bounds.append((q0, len(past), task))
            if b_ > a_:
                q0 = len(past)
                past += toks[a_:b_]
                bounds.append((q0, len(past), "scen"))
        for i in at.get(end, []):                          # fallback only (no boundary before the window)
            q0 = len(past)
            past += fids[v][i]
            bounds.append((q0, len(past), task))
        assert len(past) == P, (len(past), P)
        fb = [(a_, b_) for a_, b_, k in bounds if k == task]
        sup = story["items"][task][v]["support"]
        out.append(dict(past=past, window=toks[end:end + W], bounds=bounds, D=P - fb[-1][1] if fb else P,
                        ep=(fb[0][0], fb[-1][1]) if fb else (0, 0),
                        support_ends=[fb[i][1] - 1 for i in sup if i < len(fb)] or [fb[-1][1] - 1]))
    return out


def facts_of(lay, fam):
    fam = fam.split("#")[0]
    return [(a, b) for a, b, k in lay["bounds"] if k == fam]


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print("dedupe:", dedupe_report())
    for t in TASKS:
        print(t, "pool train", len(pool(t, "train")), "eval", len(pool(t, "eval")))
    import collections
    pf = collections.defaultdict(list)
    for g in range(20000, 20500):
        st = make_story(g, "eval")
        it = st["items"][st["task"]]
        for v in range(4):
            pres = [o for o in it[v]["options"] if any(re.search(rf"\b{o}\b", f) for f in st["facts"][v])]
            pf[st["task"]].append(1 / len(pres) if it[v]["answer_text"] in pres else 0.0)
    print("presence-only floor per question (answer among the options present in the facts, guessed):",
          {t: round(100 * sum(x) / len(x), 1) for t, x in pf.items()})
    for g in range(10000, 10000 + int(sys.argv[1] if len(sys.argv) > 1 else 5)):
        st = make_story(g, "eval")
        L = layout(st, tok, P=4096, W=256, D=300)
        it = st["items"][st["task"]]
        print(f"== {g} {st['task']} q={it[0]['question']!r} opts={it[0]['options']} ans={[x['answer'] for x in it]} "
              f"nfacts={it[0]['n_facts']} D={L[0]['D']} ep={L[0]['ep']} lens={[len(l['past']) for l in L]}")
        for v in range(4):
            diff = [i for i, (a, b) in enumerate(zip(st["facts"][0], st["facts"][v])) if a != b]
            print("   v", v, it[v]["answer_text"], "| changed facts", len(diff), "|",
                  " ".join(st["facts"][v][-4:])[:200])
        fb = facts_of(L[0], st["task"])
        print("   first fact in context:", repr(tok.decode(L[0]["past"][max(0, fb[0][0] - 30):fb[0][1] + 10])))
        print("   window:", repr(tok.decode(L[0]["window"][:60])))
