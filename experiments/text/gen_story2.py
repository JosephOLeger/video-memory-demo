"""v2 stories: JOINT mode (gist and detail facts in the same story), same-type distractors, episodes.

One GROUP = 4 variants of one story. The story carries ALL 8 queried facts at once (5 gist, 3 detail).
Each family f has its own permutation pi_f: variant v gets answer option pi_f(v); the 4 answers are
distinct for every family, so every family is a counterfactual group on the same 4 stories. The
variants share every other sentence (distractors, scenery, window).

Structure of the past (tokens):
    [pre episodes: distractor events + scenery, trimmed at the front to make the past exactly P]
    [FACT episode: the 8 fact modules interleaved with same-type distractor events + scenery]
    [k post episodes of distractor events + scenery, about D tokens]
    then the window (scenery and distractor events, about W tokens, identical across variants).

Same-type distractors: other people chatting, entering and leaving OTHER rooms, wearing coloured
garments, owning lockers with numbers, objects on furniture in other rooms, travellers going to
other places, the same kinds of incidents in other rooms, other people losing other objects. They
are never queried and are identical across variants. Every question names an entity (person or
room) that only the fact module uses, so the distractors make memory selection hard but never make
a question ambiguous.

Queried families
  gist   count_start  in room R1, n in {3,4,5,6} named people chatting at the start; how many
         presence     in room R2, 4 people, 3 leave; who is still there
         place        person P visits 4 places; where is P at the end
         order        in room R3, 4 incidents; which happened first in R3
         main_object  person M cannot find object X, searches, finds it (X named 4 times)
  detail colour       4 people, one coloured garment each (permuted); colour of T's garment
         number       4 people's locker numbers (permuted); number of T's locker
         object_name  in room R4, 4 objects on 4 spots (permuted); what was on spot S in R4
"""
from __future__ import annotations

import hashlib
import random
import re

import gen_story as G1

SALT = "text-gist-v2"
LET = "ABCD"
GIST = ["count_start", "presence", "place", "order", "main_object"]
DETAIL = ["colour", "number", "object_name"]
FAMILIES = GIST + DETAIL

ROOMS = """kitchen living-room attic cellar workshop studio classroom office greenhouse waiting-room
laundry-room dining-room bedroom pantry study nursery music-room games-room boiler-room sewing-room
porch veranda hallway storeroom""".split()
NAMES = G1.NAMES + """Ada Boris Carla Dario Elsa Fabio Greta Hans Ines Jan Kaia Luca Maja Noah Orla Pavel
Runa Saul Tilda Uri Vanja Wim Yara Zeno Alba Bodo Cleo Dirk Edda Finn""".split()
OBJECTS = G1.OBJECTS + "kite globe lamp radio statue saucepan blanket book".split()
PLACES = G1.PLACES + "zoo school bank harbour-office town-hall swimming-pool".split()
COLOURS = G1.COLOURS
GARMENTS = G1.GARMENTS
SPOTS = G1.SPOTS
EVENTS = [e[0] for e in G1.EVENTS]


def _h(s):
    return int(hashlib.md5((SALT + s).encode()).hexdigest()[:8], 16)


def split_pool(pool, split):
    return [x for x in pool if (_h(x) % 2 == 0) == (split == "train")]


P_ = G1.pretty
art = G1.art


class World:
    """Pools for one story, drawn without replacement so that fact entities are unique."""

    def __init__(self, rng, split):
        self.rng = rng
        self.names = split_pool(NAMES, split)
        self.rooms = [P_(r) for r in split_pool(ROOMS, split)]
        self.places = [P_(p) for p in split_pool(PLACES, split)]
        self.objs = split_pool(OBJECTS, split)
        for x in (self.names, self.rooms, self.places, self.objs):
            rng.shuffle(x)

    def take(self, pool, n):
        out = getattr(self, pool)[:n]
        setattr(self, pool, getattr(self, pool)[n:])
        assert len(out) == n, (pool, n)
        return out


def _perm(rng):
    p = list(range(4))
    rng.shuffle(p)
    return p


def fact_modules(rng, w: World):
    """Each module: dict(family, question, options, sents[v] list of sentences, answer_idx[v])."""
    mods = []
    # count_start
    room = w.take("rooms", 1)[0]
    ppl = w.take("names", 6)
    pi = _perm(rng)
    ns = [3, 4, 5, 6]
    sents = []
    for v in range(4):
        n = ns[pi[v]]
        grp = ppl[:n]
        sents.append([f"At the beginning of the evening, {', '.join(grp[:-1])} and {grp[-1]} were chatting in the {room}."])
    mods.append(dict(family="count_start", question=f"How many people were chatting in the {room} at the beginning of the evening?",
                     options=[str(x) for x in ns], answer=pi, sents=sents))
    # presence
    room = w.take("rooms", 1)[0]
    four = w.take("names", 4)
    order = four[:]
    rng.shuffle(order)
    leave = rng.choice(["left the {r}", "walked out of the {r}", "stepped out of the {r}"]).format(r=room)
    neutral = rng.choice(["sat down by the window", "opened a book", "made a cup of tea", "started tidying the table"])
    pi = _perm(rng)
    opts = four[:]
    rng.shuffle(opts)
    sents = []
    for v in range(4):
        stay = opts[pi[v]]
        sl = [f"{four[0]}, {four[1]}, {four[2]} and {four[3]} were together in the {room}."]
        sl += [f"{p} {neutral}." if p == stay else f"{p} {leave}." for p in order]
        sents.append(sl)
    mods.append(dict(family="presence", question=f"Which of these people is still in the {room} at the end of the story?",
                     options=opts, answer=pi, sents=sents))
    # place
    who = w.take("names", 1)[0]
    pl = w.take("places", 4)
    pi = _perm(rng)
    opts = [f"the {p}" for p in pl]
    sents = []
    for v in range(4):
        last = pl[pi[v]]
        rest = [p for p in pl if p != last]
        random.Random(f"{SALT}-place-{v}-{who}").shuffle(rest)
        seq = rest + [last]
        sents.append([f"{who} started the day at the {seq[0]}.", f"Later, {who} walked over to the {seq[1]}.",
                      f"After a while, {who} went on to the {seq[2]}.", f"Then {who} walked on to the {seq[3]}."])
    mods.append(dict(family="place", question=f"Where is {who} at the end of the story?", options=opts, answer=pi,
                     sents=sents))
    # order
    room = w.take("rooms", 1)[0]
    ev = rng.sample(EVENTS, 4)
    pi = _perm(rng)
    opens = ["At some point", "Later", "After that", "Then"]
    sents = []
    for v in range(4):
        first = ev[pi[v]]
        rest = [e for e in ev if e != first]
        random.Random(f"{SALT}-order-{v}-{room}").shuffle(rest)
        seq = [first] + rest
        sents.append([f"{o}, in the {room}, {e}." if o != "Then" else f"Then, in the {room}, {e}." for o, e in zip(opens, seq)])
    mods.append(dict(family="order", question=f"Which of these happened first in the {room}?", options=ev, answer=pi,
                     sents=sents))
    # main_object
    who = w.take("names", 1)[0]
    ob = w.take("objs", 4)
    pi = _perm(rng)
    sents = []
    for v in range(4):
        x = ob[pi[v]]
        sents.append([f"{who} could not find the {x} anywhere.", f"{who} kept looking for the missing {x} under every cushion.",
                      f"{who} asked everyone about the missing {x}.", f"At last {who} found the {x} in the pocket of an old coat."])
    # the 3 other options appear as incidental objects (identical bag across variants)
    inc = []
    for v in range(4):
        x = ob[pi[v]]
        rest = [o for o in ob if o != x]
        inc.append(rest)
    # incidental sentences must be identical across variants -> they mention all 4 objects except x,
    # so we use a fixed sentence per object and drop the one equal to x (keeps the bag identical)
    holders = w.take("names", 3)
    for v in range(4):
        x = ob[pi[v]]
        rest = [o for o in ob if o != x]
        sents[v] += [f"{holders[i]} was carrying {art(o)} {o}." for i, o in enumerate(rest)]
    mods.append(dict(family="main_object", question=f"Which object had {who} lost and searched for?",
                     options=[f"the {o}" for o in ob], answer=pi, sents=sents, n_core=4))
    # colour
    four = w.take("names", 4)
    gar = [rng.choice(GARMENTS) for _ in four]
    col = rng.sample(COLOURS, 4)
    tgt = rng.randrange(4)
    pi = _perm(rng)
    sents = []
    for v in range(4):
        c = col[pi[v]]
        rest = [x for x in col if x != c]
        assign, k = [], 0
        for i in range(4):
            if i == tgt:
                assign.append(c)
            else:
                assign.append(rest[k]); k += 1
        sents.append([f"{four[i]} was wearing {art(assign[i])} {assign[i]} {gar[i]}." for i in range(4)])
    mods.append(dict(family="colour", question=f"What colour was {four[tgt]}'s {gar[tgt]}?", options=col, answer=pi,
                     sents=sents))
    # number
    four = w.take("names", 4)
    nums = sorted(rng.sample(range(12, 99), 4))
    tgt = rng.randrange(4)
    pi = _perm(rng)
    sents = []
    for v in range(4):
        c = nums[pi[v]]
        rest = [x for x in nums if x != c]
        assign, k = [], 0
        for i in range(4):
            if i == tgt:
                assign.append(c)
            else:
                assign.append(rest[k]); k += 1
        sents.append([f"{four[i]}'s locker was number {assign[i]}." for i in range(4)])
    mods.append(dict(family="number", question=f"What was the number of {four[tgt]}'s locker?",
                     options=[str(x) for x in nums], answer=pi, sents=sents))
    # object_name
    room = w.take("rooms", 1)[0]
    sp = rng.sample(SPOTS, 4)
    ob = w.take("objs", 4)
    tgt = rng.randrange(4)
    pi = _perm(rng)
    sents = []
    for v in range(4):
        c = ob[pi[v]]
        rest = [x for x in ob if x != c]
        assign, k = [], 0
        for i in range(4):
            if i == tgt:
                assign.append(c)
            else:
                assign.append(rest[k]); k += 1
        sents.append([f"In the {room}, there was {art(assign[i])} {assign[i]} on the {sp[i]}." for i in range(4)])
    mods.append(dict(family="object_name", question=f"In the {room}, what was on the {sp[tgt]}?",
                     options=[f"{art(o)} {o}" for o in ob], answer=pi, sents=sents))
    return mods


def distractor_events(rng, w: World, n):
    """n same-type distractor sentences using the world's remaining (never queried) entities.
    Draws with replacement from the distractor pools (fixed per story)."""
    if not hasattr(w, "d_names"):
        w.d_names, w.d_rooms, w.d_places, w.d_objs = w.names[:], w.rooms[:], w.places[:], w.objs[:]
        assert len(w.d_names) >= 6 and len(w.d_rooms) >= 3 and len(w.d_places) >= 3 and len(w.d_objs) >= 3
    out = []
    for _ in range(n):
        k = rng.randrange(9)
        a, b, c = rng.sample(w.d_names, 3)
        room = rng.choice(w.d_rooms)
        if k == 0:
            out.append(f"{a}, {b} and {c} were chatting in the {room}.")
        elif k == 1:
            out.append(f"{a} {rng.choice(['left the', 'walked out of the', 'came into the', 'stepped into the'])} {room}.")
        elif k == 2:
            cc = rng.choice(COLOURS)
            out.append(f"{a} was wearing {art(cc)} {cc} {rng.choice(GARMENTS)}.")
        elif k == 3:
            out.append(f"{a}'s locker was number {rng.randrange(12, 99)}.")
        elif k == 4:
            o = rng.choice(w.d_objs)
            out.append(f"In the {room}, there was {art(o)} {o} on the {rng.choice(SPOTS)}.")
        elif k == 5:
            out.append(f"{a} walked over to the {rng.choice(w.d_places)}.")
        elif k == 6:
            out.append(f"In the {room}, {rng.choice(EVENTS)}.")
        elif k == 7:
            out.append(f"{a} could not find the {rng.choice(w.d_objs)} anywhere.")
        else:
            o = rng.choice(w.d_objs)
            out.append(f"{a} was carrying {art(o)} {o}.")
    return out


def make_story(g: int, split: str = "eval", dist_rate=0.5):
    """Group g: dict(items[family] -> 4 item dicts, fact_sents[v] (ordered, tagged by family),
    world rng state). Layout is done by `layout`."""
    rng = random.Random(f"{SALT}-{split}-{g}")
    w = World(rng, split)
    mods = fact_modules(rng, w)
    # interleave modules (keep each module's internal order), constant across variants
    seq = []
    for mi, m in enumerate(mods):
        seq += [mi] * len(m["sents"][0])
    rng.shuffle(seq)
    # count_start sentence says "at the beginning of the evening": it must come first among facts
    cs = [i for i, m in enumerate(mods) if m["family"] == "count_start"][0]
    seq.remove(cs)
    seq.insert(0, cs)
    # in-episode distractors: dist_rate per fact sentence, positions fixed
    nd = int(round(dist_rate * len(seq)))
    dpos = sorted(rng.sample(range(1, len(seq) + nd), nd))
    var_sents = []   # per variant: list of (sentence, family or None)
    dist_ep = distractor_events(rng, w, nd)
    for v in range(4):
        cur = [0] * len(mods)
        out, di = [], 0
        fi = 0
        for pos in range(len(seq) + nd):
            if di < nd and dpos[di] == pos:
                out.append((dist_ep[di], None)); di += 1
            else:
                mi = seq[fi]; fi += 1
                out.append((mods[mi]["sents"][v][cur[mi]], mods[mi]["family"])); cur[mi] += 1
        var_sents.append(out)
    items = {}
    for m in mods:
        f = m["family"]
        items[f] = [dict(family=f, kind="gist" if f in GIST else "detail", group=g, variant=v, split=split,
                         question=m["question"], options=m["options"], answer=m["answer"][v],
                         answer_text=m["options"][m["answer"][v]]) for v in range(4)]
        assert len(set(m["answer"])) == 4
    return dict(group=g, items=items, var_sents=var_sents, rng_seed=rng.randrange(1 << 30), world=w)


def layout(story, tok, P=8192, W=256, D=0, ev_per_scenery=1.0):
    """Token streams for the 4 variants. Post-fact episodes (~D tokens) and window mix distractor
    events and scenery (ev_per_scenery events per scenery sentence), identical across variants."""
    rng = random.Random(story["rng_seed"])
    w = story["world"]
    enc = lambda s: tok.encode(" " + s, add_special_tokens=False)  # noqa: E731
    fill = G1.filler_sentences(rng, 4000)
    fi = [0]

    def scen():
        s = fill[fi[0] % len(fill)]; fi[0] += 1
        return s

    def mixed(ntok):
        """whole sentences, events and scenery mixed, >= ntok tokens -> [(ids, kind)]"""
        out, tot = [], 0
        while tot < ntok:
            if rng.random() < ev_per_scenery / (1 + ev_per_scenery):
                s, kind = distractor_events(rng, w, 1)[0], "dist"
            else:
                s, kind = scen(), "scen"
            ids = enc(s)
            out.append((ids, kind)); tot += len(ids)
        return out, tot

    win_s, _ = mixed(W)
    post_s, Dx = mixed(D) if D > 0 else ([], 0)
    nfs = len(story["var_sents"][0])
    gaps = [(enc(scen()), "scen") for _ in range(nfs - 1)]
    pre_pool, _ = mixed(P)        # enough for any trimming
    out = []
    for v in range(4):
        sents = []
        for k, (s, fam) in enumerate(story["var_sents"][v]):
            sents.append((enc(s), fam if fam else "dist"))
            if k < nfs - 1:
                sents.append(gaps[k])
        sents += post_s
        core = sum(len(x) for x, _ in sents)
        assert core < P, (core, P)
        pre, j = [], len(pre_pool) - 1
        while sum(len(x) for x, _ in pre) < P - core:
            pre.insert(0, pre_pool[j]); j -= 1
        excess = sum(len(x) for x, _ in pre) - (P - core)
        pre[0] = (pre[0][0][excess:], pre[0][1])
        allp = pre + sents
        past, bounds = [], []
        for ids, kind in allp:
            a = len(past)
            past += ids
            bounds.append((a, len(past), kind))
        assert len(past) == P
        window = [x for ids, _ in win_s for x in ids]
        ep_a = bounds[len(pre)][0]
        ep_b = bounds[len(pre) + len(sents) - len(post_s) - 1][1]
        out.append(dict(past=past, window=window, bounds=bounds, D=Dx, ep=(ep_a, ep_b)))
    return out


def facts_of(lay, fam):
    return [(a, b) for a, b, k in lay["bounds"] if k == fam]


if __name__ == "__main__":
    st = make_story(10000)
    for f, its in st["items"].items():
        print(f, "|", its[0]["question"], its[0]["options"], [it["answer"] for it in its])
    for v in range(2):
        print("---- variant", v)
        for s, f in st["var_sents"][v]:
            print(f"   [{f or '.'}] {s}")
    bag = lambda v: sorted(re.findall(r"[a-z0-9'\-]+", " ".join(s for s, _ in st["var_sents"][v]).lower()))  # noqa
    print("bags identical:", [bag(v) == bag(0) for v in range(4)])
