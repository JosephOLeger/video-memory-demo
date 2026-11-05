"""v3 stories: CUMULATIVE families, controlled spread (25/10).

The v2 gist families turned out to be local (count_start is one sentence; order's answer is marked
"At some point" in its own sentence). v3 queries facts that need several sentences spread over the
stream, one sentence per element, no ordinal marker, no recap:

    count_arrivals  in room R, n people arrive (one sentence each), n in a 4-subset of {2..6}; how many
    majority        in room R, 5 to 9 umbrellas hung (one sentence each), 4 colours, one strict majority
                    colour (never the first nor the last mention); which colour most often
    last_state      in room R, object X is put on a spot 3 to 5 times (one sentence each, no "then",
                    "finally"); where is X at the end (recency and overwriting, not aggregation)

Group structure as v2: 4 variants per story, each family has its own permutation variant -> answer,
everything else (distractors, scenery, window) is identical across variants.

Spread: per family and per group, the elements are placed between offsets o_f and o_f + S_f of the
fact episode, S_f ~ U[256, 1500] tokens (shared by the 4 variants), inner elements at sorted uniform
fractions. Placement snaps to sentence boundaries of a background stream (distractors + scenery)
that is identical across variants. Same-type distractors: other people arriving in OTHER rooms,
umbrellas hung in OTHER rooms, other objects moved in OTHER rooms, plus all v2 distractors.
"""
from __future__ import annotations

import random

import gen_story as G1
import gen_story2 as G2

SALT = "text-cumul-v3"
LET = "ABCD"
GIST = ["count_arrivals", "majority", "last_state"]
DETAIL = []
FAMILIES = GIST
# last_state is NOT certified (25/10: oracle 42 % per question, full context 15 to 25 % by group): the 3B
# host cannot pick the last of identical sentences. Kept in the eval, out of training.
TRAIN_FAMS = {"A": ["count_arrivals", "majority"], "B": ["count_arrivals", "majority"]}
HELD_OUT = None
SPREAD = (256, 1500)
EPISODE = 1700          # tokens of background in the fact episode (>= max spread + margin)
COLOURS = G2.COLOURS
SPOTS = G2.SPOTS
art = G2.art


def _perm(rng):
    p = list(range(4))
    rng.shuffle(p)
    return p


def _fracs(rng, k):
    """k positions in [0, 1]: first at 0, last at 1, inner sorted uniform."""
    if k == 1:
        return [0.0]
    inner = sorted(rng.random() for _ in range(k - 2))
    return [0.0] + inner + [1.0]


MAJ_COUNTS = {5: [2, 1, 1, 1], 6: [3, 1, 1, 1], 7: [3, 2, 1, 1], 8: [3, 2, 2, 1], 9: [4, 2, 2, 1]}


def fact_modules(rng, w):
    mods = []
    # count_arrivals
    # MASS BALANCE (25/10): every variant has the same N = max(ns) arrival sentences by the same N people;
    # the n first go to the queried room, the N - n others to a balance room. Only the room word differs,
    # so the number of fact sentences, the name bag and (almost) the token count do not depend on n.
    room = w.take("rooms", 1)[0]
    same = [r for r in w.rooms if len(r.split()) == len(room.split())]   # same word count -> same token count
    broom = same[0] if same else w.rooms[0]
    w.rooms.remove(broom)
    ns =[2, 3, 4, 5] if rng.random() < 0.5 else [3, 4, 5, 6]
    N = max(ns)
    ppl = w.take("names", N)
    pi = _perm(rng)
    sents = []
    for v in range(4):
        n = ns[pi[v]]
        sl = [f"{p} arrived at the {room}." for p in ppl[:n]] + [f"{p} arrived at the {broom}." for p in ppl[n:]]
        random.Random(f"{SALT}-arr-{v}-{room}").shuffle(sl)
        sents.append(sl)
    mods.append(dict(family="count_arrivals", question=f"How many people arrived at the {room} during the story?",
                     options=[str(x) for x in ns], answer=pi, sents=sents))
    # majority
    room = w.take("rooms", 1)[0]
    col = rng.sample(COLOURS, 4)
    m = rng.choice(sorted(MAJ_COUNTS))
    counts = MAJ_COUNTS[m]
    pi = _perm(rng)
    sents = []
    for v in range(4):
        maj = col[pi[v]]
        others = [c for c in col if c != maj]
        random.Random(f"{SALT}-maj-o-{v}-{room}").shuffle(others)
        bag = [maj] * counts[0] + [c for c, k in zip(others, counts[1:]) for _ in range(k)]
        r = random.Random(f"{SALT}-maj-{v}-{room}")
        for _ in range(1000):                       # majority colour never first nor last
            r.shuffle(bag)
            if bag[0] != maj and bag[-1] != maj:
                break
        sents.append([f"In the {room}, someone hung {art(c)} {c} umbrella by the door." for c in bag])
    mods.append(dict(family="majority", question=f"Which colour of umbrella was hung up most often in the {room}?",
                     options=col, answer=pi, sents=sents))
    # last_state
    room = w.take("rooms", 1)[0]
    obj = w.take("objs", 1)[0]
    sp = rng.sample(SPOTS, 4)
    k = rng.choice([3, 4, 5])
    pi = _perm(rng)
    sents = []
    for v in range(4):
        last = sp[pi[v]]
        rest = [s for s in sp if s != last]
        r = random.Random(f"{SALT}-last-{v}-{room}")
        r.shuffle(rest)
        if k == 3:
            seq = rest[:2]
        elif k == 4:
            seq = rest
        else:                                        # one repeat, never adjacent to itself
            seq = rest + [rest[0]] if rest[0] != rest[-1] else rest + [rest[1]]
        seq = seq + [last]
        sents.append([f"In the {room}, someone put the {obj} on the {s}." for s in seq])
    mods.append(dict(family="last_state", question=f"In the {room}, where was the {obj} at the end of the story?",
                     options=[f"the {s}" for s in sp], answer=pi, sents=sents))
    return mods


def distractor_events(rng, w, n):
    """v2 distractors plus the three v3 same-type distractors (other rooms, other objects)."""
    G2.distractor_events(rng, w, 0)                  # initialises the distractor pools
    out = []
    for _ in range(n):
        k = rng.randrange(12)
        if k < 9:
            out += G2.distractor_events(rng, w, 1)
            continue
        a = rng.choice(w.d_names)
        room = rng.choice(w.d_rooms)
        if k == 9:
            out.append(f"{a} arrived at the {room}.")
        elif k == 10:
            c = rng.choice(COLOURS)
            out.append(f"In the {room}, someone hung {art(c)} {c} umbrella by the door.")
        else:
            out.append(f"In the {room}, someone put the {rng.choice(w.d_objs)} on the {rng.choice(SPOTS)}.")
    return out


def make_story(g: int, split: str = "eval"):
    rng = random.Random(f"{SALT}-{split}-{g}")
    w = G2.World(rng, split)
    mods = fact_modules(rng, w)
    place = {}
    for m in mods:
        S = rng.randint(*SPREAD)
        o = rng.randint(0, EPISODE - S)
        kmax = max(len(m["sents"][v]) for v in range(4))
        place[m["family"]] = dict(spread=S, start=o,
                                  fracs=[_fracs(random.Random(f"{SALT}-fr-{g}-{m['family']}-{v}"), len(m["sents"][v]))
                                         for v in range(4)], kmax=kmax)
    items = {}
    for m in mods:
        f = m["family"]
        items[f] = [dict(family=f, kind="gist", group=g, variant=v, split=split, question=m["question"],
                         options=m["options"], answer=m["answer"][v], answer_text=m["options"][m["answer"][v]],
                         spread=place[f]["spread"]) for v in range(4)]
        assert len(set(m["answer"])) == 4
    return dict(group=g, items=items, mods=mods, place=place, rng_seed=rng.randrange(1 << 30), world=w)


def layout(story, tok, P=4096, W=256, D=0, ev_per_scenery=1.0):
    """Token streams for the 4 variants: [pre filler][fact episode: background with the fact sentences
    snapped to sentence boundaries at their target offsets][post ~D tokens][window W]."""
    rng = random.Random(story["rng_seed"])
    w = story["world"]
    enc = lambda s: tok.encode(" " + s, add_special_tokens=False)  # noqa: E731
    fill = G1.filler_sentences(rng, 6000)
    fi = [0]

    def mixed(ntok):
        out, tot = [], 0
        while tot < ntok:
            if rng.random() < ev_per_scenery / (1 + ev_per_scenery):
                s, kind = distractor_events(rng, w, 1)[0], "dist"
            else:
                s, kind = fill[fi[0] % len(fill)], "scen"; fi[0] += 1
            ids = enc(s)
            out.append((ids, kind)); tot += len(ids)
        return out, tot

    win_s, _ = mixed(W)
    post_s, Dx = mixed(D) if D > 0 else ([], 0)
    bg, _ = mixed(EPISODE)
    offs, t = [], 0
    for ids, _ in bg:
        offs.append(t); t += len(ids)
    offs.append(t)
    pre_pool, _ = mixed(P)
    out = []
    for v in range(4):
        ins = []                                       # (boundary index, target offset, ids, family)
        for m in story["mods"]:
            f = m["family"]
            pl = story["place"][f]
            for s, fr in zip(m["sents"][v], pl["fracs"][v]):
                tgt = pl["start"] + fr * pl["spread"]
                bi = min(range(len(offs)), key=lambda i: abs(offs[i] - tgt))
                ins.append((bi, tgt, enc(s), f))
        ins.sort(key=lambda x: (x[0], x[1]))
        ep, j = [], 0
        for bi in range(len(bg) + 1):
            while j < len(ins) and ins[j][0] == bi:
                ep.append((ins[j][2], ins[j][3])); j += 1
            if bi < len(bg):
                ep.append(bg[bi])
        sents = ep + post_s
        core = sum(len(x) for x, _ in sents)
        assert core < P, (core, P)
        pre, k = [], len(pre_pool) - 1
        while sum(len(x) for x, _ in pre) < P - core:
            pre.insert(0, pre_pool[k]); k -= 1
        excess = sum(len(x) for x, _ in pre) - (P - core)
        pre[0] = (pre[0][0][excess:], pre[0][1])
        past, bounds = [], []
        for ids, kind in pre + sents:
            a = len(past)
            past += ids
            bounds.append((a, len(past), kind))
        assert len(past) == P
        window = [x for ids, _ in win_s for x in ids]
        ep_a = bounds[len(pre)][0]
        ep_b = bounds[len(pre) + len(ep) - 1][1]
        out.append(dict(past=past, window=window, bounds=bounds, D=Dx, ep=(ep_a, ep_b)))
    return out


def facts_of(lay, fam):
    return [(a, b) for a, b, k in lay["bounds"] if k == fam]


if __name__ == "__main__":
    st = make_story(10000)
    for f, its in st["items"].items():
        print(f, "|", its[0]["question"], its[0]["options"], [it["answer"] for it in its], "spread", its[0]["spread"])
    for m in st["mods"]:
        for v in range(2):
            print(m["family"], v, m["sents"][v])
