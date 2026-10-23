"""Synthetic long stories for a text memory benchmark (gist vs detail), counterfactual groups.

One GROUP = 4 variants of the same story. The variants share every filler sentence, every
slot position and the window; they differ only in the content of the fact slots, so that
the asked fact changes (4 distinct answers, one per variant, letter = variant index after
a per-group option shuffle that keeps each letter used exactly once).

Anti copy shortcut: in every family except count_start and count_end, the BAG OF WORDS of
the 4 variants is identical. The variants permute the same words over the same slots; only
the binding (who wore which colour, which place came last, ...) changes. A memory that keeps
surface keywords without order or binding cannot separate the variants.

Families
  gist   count_end   people in a room, 3 arrivals or departures; how many at the end
         count_start named people at the start (3 to 6); how many at the start
         presence    4 people, 3 leave, one stays; who is still there
         place       a character visits 4 places; where is he at the end
         order       4 events in some order; which happened first
         main_object a character lost an object and searches for it ("it"); 3 incidental
                     objects mentioned once; what had she lost
  detail colour      4 people, one garment each, colours permuted; colour of X's garment
         number      4 people's locker numbers permuted; number of X's locker
         object_name 4 spots in a room with one object each, permuted; what was on spot S

Layout (tokens, built by `layout`): [prefix filler, trimmed at the front to make the past
exactly P tokens][fact span: slot, filler, slot, filler, ..., slot][post filler, about D
tokens][window filler, about W tokens]. Post and window are whole filler sentences shared
by the 4 variants, so the window and the exact distance are identical across variants.

Split: names, objects, places and rooms are partitioned between train and eval by a
salted hash; the group RNG is salted by split and family.
"""
from __future__ import annotations

import hashlib
import random
import re

SALT = "text-gist-v1"
LET = "ABCD"

NAMES = """Anna Ben Carl Dora Eli Fay Gus Hana Ivo Jade Kurt Lena Milo Nora Omar Pia Quinn Rosa
Sami Tess Ugo Vera Walt Xena Yuri Zoe Abel Bea Cyril Dina Emil Flora Gil Hilde Igor Joan Karl
Lise Marc Nina Otto Paula Rene Sofia Tomas Una Vik Wanda Yann Zita Arno Bruna Colin Della Enzo
Frida Hugo Iris Jonas Klara Leo Mona Nils Olga Piet Rita Sven Thea Udo Viola""".split()
OBJECTS = """compass umbrella notebook lantern kettle wallet scarf flute camera hammer teapot
bracelet telescope mirror candle basket violin pencil ladder bucket clock ribbon trumpet
envelope backpack helmet thermos whistle shovel puzzle chessboard harmonica stapler magnet
""".split()
PLACES = """harbour market library station bakery museum church cinema pharmacy stadium bridge
garage pier hospital theatre factory cemetery lighthouse post-office barber-shop""".split()
ROOMS = """kitchen living-room attic cellar workshop studio classroom office greenhouse
waiting-room""".split()
COLOURS = "red green blue yellow purple orange pink brown".split()
GARMENTS = "scarf hat coat jumper jacket cap".split()
SPOTS = ["windowsill", "shelf", "desk", "chair", "piano", "sideboard", "mantelpiece", "stool"]
EVENTS = [("the power went out", "the power went out"),
          ("a dog started barking", "a dog started barking"),
          ("the phone rang", "the phone rang"),
          ("it began to rain", "it began to rain"),
          ("the doorbell rang", "the doorbell rang"),
          ("a glass fell and broke", "a glass fell and broke"),
          ("the fire alarm went off", "the fire alarm went off"),
          ("the neighbours started to sing", "the neighbours started to sing")]
NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}

GIST = ["count_end", "count_start", "presence", "place", "order", "main_object", "count_end2", "count_left", "main_object2"]
DETAIL = ["colour", "number", "object_name"]
FAMILIES = GIST + DETAIL


def _h(s: str) -> int:
    return int(hashlib.md5((SALT + s).encode()).hexdigest()[:8], 16)


def split_pool(pool, split):
    """Disjoint halves of a pool for train and eval (salted hash)."""
    keep = [x for x in pool if (_h(x) % 2 == 0) == (split == "train")]
    return keep


def pretty(w):
    return w.replace("-", " ")


# ------------------------------------------------------------------ filler (scenery, no people)
F_OPEN = ["", "", "", "Outside, ", "For a while, ", "Somewhere nearby, ", "Meanwhile, ", "Now and then, ",
          "In the distance, ", "Slowly, ", "Once again, ", "At that moment, ", "Quietly, ", "Far away, "]
F_CORE = [
    "the wind moved through the tall grass", "a clock ticked on the wall", "the light shifted across the floorboards",
    "clouds drifted over the hills", "a kettle hummed on a stove in another house", "leaves gathered against the fence",
    "the old house creaked as it settled", "a cart rolled along the cobbled lane", "birds circled above the rooftops",
    "the smell of bread came from a neighbouring street", "a shutter knocked softly against its frame",
    "the river kept its slow pace under the trees", "a church bell rang somewhere across the valley",
    "the shadows grew longer on the wall", "dust floated in a beam of sunlight", "the curtains stirred in the draught",
    "a train whistled far beyond the fields", "the radiator ticked as it warmed up", "frost melted on the window panes",
    "the trees bent a little in the breeze", "a moth bumped against the lamp", "the hills turned soft in the haze",
    "a gate swung and settled again", "the floor was warm where the sun had been", "the sky slowly changed its mood",
    "a pair of swallows chased each other over the yard", "the scent of cut grass drifted in",
    "somebody far away was mowing a lawn", "the roof gutters dripped steadily", "the air smelled of wood smoke",
    "the evening crept over the gardens", "a breeze lifted the corner of a newspaper on a bench",
    "the town hall clock struck the quarter hour", "a tractor worked its way across a distant field",
    "the hedges rustled along the road", "a bee wandered from flower to flower", "the stones of the wall held the warmth",
    "a cat stretched on a sunny step and went back to sleep", "the pond reflected the passing clouds",
    "the weather vane turned on the barn roof", "the fields shimmered under the afternoon sun",
    "a low hum came from the power lines", "the pages of an open book lifted in the wind",
    "the street lamps flickered and steadied", "a seagull cried over the rooftops", "the hours seemed to stretch out",
    "the floorboards cooled as the day went on", "the scent of rosemary came from the garden",
    "a distant engine faded into silence", "the sun slipped behind a line of poplars",
]
F_TAIL = ["", "", "", " without hurry", " as it always did", " for a long time", " and then it was quiet again",
          " in the usual way", ", as on most days"]


def _filler_bank():
    bad = set(w.lower() for w in NAMES + OBJECTS + COLOURS + GARMENTS + SPOTS) | set(
        pretty(w) for w in PLACES + ROOMS) | {"dog", "phone", "power", "rain", "doorbell", "glass", "alarm", "sing"}
    out = []
    for o in F_OPEN:
        for c in F_CORE:
            for t in F_TAIL:
                s = (o + c + t + ".")
                s = s[0].upper() + s[1:]
                words = set(re.findall(r"[a-z\-]+", s.lower()))
                if words & bad:
                    continue
                out.append(s)
    return sorted(set(out))


def art(w):
    return "an" if w[0] in "aeiou" else "a"


FILLER = _filler_bank()
FILLER_VOCAB = set(w for s in FILLER for w in re.findall(r"[a-z\-]+", s.lower()))


def filler_sentences(rng: random.Random, n: int):
    out, last = [], set()
    while len(out) < n:
        s = rng.choice(FILLER)
        core = s.split(",")[-1][:25]
        if core in last:
            continue
        out.append(s)
        last = set(list(last)[-6:]) | {core}
    return out


# ------------------------------------------------------------------ families
def _verbs_leave(rng):
    return rng.choice(["left the {r}", "went out", "said goodbye and left", "headed home", "stepped out of the {r}",
                       "walked out"])


def _verbs_join(rng):
    return rng.choice(["came into the {r}", "joined the others", "walked into the {r}", "came in and sat down",
                       "stepped into the {r}"])


def make_group(family: str, g: int, split: str = "eval"):
    """4 variants. Each: dict(family, kind, group, variant, slots[list of str], question, options,
    answer (index), answer_text, filler_seed)."""
    rng = random.Random(f"{SALT}-{split}-{family}-{g}")
    names = split_pool(NAMES, split)
    objs = split_pool(OBJECTS, split)
    places = split_pool(PLACES, split)
    rooms = split_pool(ROOMS, split)
    rng.shuffle(names)
    room = pretty(rng.choice(rooms))
    var_slots, answers_text, q = [], [], None
    opt_order = None

    if family == "count_end":
        n0 = rng.randint(4, 6)
        init, outside = names[:n0], names[n0:n0 + 3]
        leavers = rng.sample(init, 3)
        # finals n0-3, n0-1, n0+1, n0+3 for 0..3 arrivals
        jv = [_verbs_join(rng) for _ in range(3)]
        lv = [_verbs_leave(rng) for _ in range(3)]
        for v in range(4):
            vr = random.Random(f"{SALT}-{split}-{family}-{g}-{v}")
            kinds = ["join"] * v + ["leave"] * (3 - v)
            vr.shuffle(kinds)
            ji = li = 0
            sl = [f"{', '.join(init[:-1])} and {init[-1]} were chatting in the {room}.",
                  f"{', '.join(outside[:-1])} and {outside[-1]} were waiting in the hall outside."]
            for k in kinds:
                if k == "join":
                    sl.append(f"{outside[ji]} {jv[len(sl) - 2].format(r=room)}.")
                    ji += 1
                else:
                    sl.append(f"{leavers[li]} {lv[len(sl) - 2].format(r=room)}.")
                    li += 1
            var_slots.append(sl)
            answers_text.append(str(n0 - 3 + 2 * v))
        q = f"How many people are in the {room} at the end of the story?"
    elif family == "count_start":
        ns = [3, 4, 5, 6]
        others = names[6:9]
        for v in range(4):
            grp = names[:ns[v]]
            sl = [f"{', '.join(grp[:-1])} and {grp[-1]} were chatting in the {room} at the beginning of the evening.",
                  f"Later, {others[0]} and {others[1]} passed by the window without coming in.",
                  f"Much later, {others[2]} phoned to ask about the weekend."]
            var_slots.append(sl)
            answers_text.append(str(ns[v]))
        q = f"How many people were chatting in the {room} at the beginning of the evening?"
    elif family == "presence":
        four = names[:4]
        order = four[:]
        rng.shuffle(order)
        neutral = rng.choice(["sat down by the window", "opened a book", "started tidying the table",
                              "stretched and yawned", "made a cup of tea"])
        leave = _verbs_leave(rng)
        for v in range(4):
            sl = [f"{four[0]}, {four[1]}, {four[2]} and {four[3]} were together in the {room}."]
            for p in order:
                if p == four[v]:
                    sl.append(f"{p} {neutral}.")
                else:
                    sl.append(f"{p} {leave.format(r=room)}.")
            var_slots.append(sl)
            answers_text.append(four[v])
        q = f"Which of these people is still in the {room} at the end of the story?"
        opt_order = four
    elif family == "place":
        who = names[0]
        pl = [pretty(p) for p in rng.sample(places, 4)]
        for v in range(4):
            vr = random.Random(f"{SALT}-{split}-{family}-{g}-{v}")
            rest = [p for i, p in enumerate(pl) if i != v]
            vr.shuffle(rest)
            seq = rest + [pl[v]]
            sl = [f"{who} started the day at the {seq[0]}.",
                  f"Later, {who} walked over to the {seq[1]}.",
                  f"After a while, {who} went on to the {seq[2]}.",
                  f"Then {who} walked on to the {seq[3]}."]
            var_slots.append(sl)
            answers_text.append(f"the {pl[v]}")
        q = f"Where is {who} at the end of the story?"
    elif family == "order":
        ev = [e[0] for e in rng.sample(EVENTS, 4)]
        opens = ["At some point,", "Later,", "After that,", "Then"]
        for v in range(4):
            vr = random.Random(f"{SALT}-{split}-{family}-{g}-{v}")
            rest = [e for i, e in enumerate(ev) if i != v]
            vr.shuffle(rest)
            seq = [ev[v]] + rest
            sl = [f"{o} {e}." for o, e in zip(opens, seq)]
            sl = [s[0].upper() + s[1:] for s in sl]
            var_slots.append(sl)
            answers_text.append(ev[v])
        q = "Which of these happened first in the story?"
    elif family == "main_object":
        who = names[0]
        ob = rng.sample(objs, 4)
        inc_t = ["{A} {o} lay forgotten on the table.", "Someone had left {a} {o} by the door.",
                 "There was {a} {o} on top of the cupboard."]
        # slot plan: lost, inc, search, inc, ask, inc, found (fixed per group)
        for v in range(4):
            rest = [o for i, o in enumerate(ob) if i != v]
            sl = [f"{who} realised that the {ob[v]} was missing.",
                  inc_t[0].format(o=rest[0], A=art(rest[0]).capitalize()),
                  f"{who} searched for it under every cushion.",
                  inc_t[1].format(o=rest[1], a=art(rest[1])),
                  f"{who} asked around whether anyone had seen it.",
                  inc_t[2].format(o=rest[2], a=art(rest[2])),
                  f"At last {who} found it in the pocket of an old coat."]
            var_slots.append(sl)
            answers_text.append(f"the {ob[v]}")
        q = f"What was {who} looking for?"
    elif family in ("count_end2", "count_left"):
        # departures only (Josep's example): 5 named people, 4 event slots, k = 1..4 of them leave,
        # the other slots are neutral actions by people who stay
        init = names[:5]
        order = init[:4]
        rng.shuffle(order)
        leave = _verbs_leave(rng)
        neutral = rng.choice(["opened a window", "poured some water", "looked at the clock", "turned on a lamp"])
        for v in range(4):
            k = v + 1
            vr = random.Random(f"{SALT}-{split}-{family}-{g}-{v}")
            who_leaves = set(vr.sample(range(4), k))
            sl = [f"{', '.join(init[:-1])} and {init[-1]} were chatting in the {room}."]
            for i, p in enumerate(order):
                sl.append(f"{p} {leave.format(r=room)}." if i in who_leaves else f"{p} {neutral}.")
            var_slots.append(sl)
            answers_text.append(str(5 - k) if family == "count_end2" else str(k))
        q = (f"How many people are still in the {room} at the end of the story?" if family == "count_end2"
             else f"How many people left the {room} during the story?")
    elif family == "main_object2":
        who = names[0]
        ob = rng.sample(objs, 4)
        inc_t = ["{A} {o} lay on the table.", "{A} {o} hung by the door.", "{A} {o} stood on the cupboard."]
        for v in range(4):
            rest = [o for i, o in enumerate(ob) if i != v]
            sl = [f"{who} could not find the {ob[v]} anywhere.",
                  inc_t[0].format(o=rest[0], A=art(rest[0]).capitalize()),
                  f"{who} kept looking for the missing {ob[v]} under every cushion.",
                  inc_t[1].format(o=rest[1], A=art(rest[1]).capitalize()),
                  f"{who} asked everyone about the missing {ob[v]}.",
                  inc_t[2].format(o=rest[2], A=art(rest[2]).capitalize()),
                  f"At last {who} found the {ob[v]} in the pocket of an old coat."]
            var_slots.append(sl)
            answers_text.append(f"the {ob[v]}")
        q = f"Which object had {who} lost and searched for?"
    elif family == "colour":
        four = names[:4]
        gar = [rng.choice(GARMENTS) for _ in four]
        col = rng.sample(COLOURS, 4)
        tgt = rng.randrange(4)
        for v in range(4):
            # colour index for person i in variant v: target gets col[v]
            perm = [c for i, c in enumerate(col) if i != v]
            assign, k = [], 0
            for i in range(4):
                if i == tgt:
                    assign.append(col[v])
                else:
                    assign.append(perm[k])
                    k += 1
            sl = [f"{four[i]} was wearing a {assign[i]} {gar[i]}." for i in range(4)]
            var_slots.append(sl)
            answers_text.append(col[v])
        q = f"What colour was {four[tgt]}'s {gar[tgt]}?"
    elif family == "number":
        four = names[:4]
        nums = rng.sample(range(12, 99), 4)
        tgt = rng.randrange(4)
        for v in range(4):
            perm = [c for i, c in enumerate(nums) if i != v]
            assign, k = [], 0
            for i in range(4):
                if i == tgt:
                    assign.append(nums[v])
                else:
                    assign.append(perm[k])
                    k += 1
            sl = [f"{four[i]}'s locker was number {assign[i]}." for i in range(4)]
            var_slots.append(sl)
            answers_text.append(str(nums[v]))
        q = f"What was the number of {four[tgt]}'s locker?"
    elif family == "object_name":
        sp = rng.sample(SPOTS, 4)
        ob = rng.sample(objs, 4)
        tgt = rng.randrange(4)
        for v in range(4):
            perm = [c for i, c in enumerate(ob) if i != v]
            assign, k = [], 0
            for i in range(4):
                if i == tgt:
                    assign.append(ob[v])
                else:
                    assign.append(perm[k])
                    k += 1
            sl = [f"In the {room}, there was {art(assign[i])} {assign[i]} on the {sp[i]}." for i in range(4)]
            var_slots.append(sl)
            answers_text.append(f"{art(ob[v])} {ob[v]}")
        q = f"In the {room}, what was on the {sp[tgt]}?"
    else:
        raise ValueError(family)

    # options: the 4 answers, shown in a per-group shuffled order; letter of variant v = position
    opts = answers_text[:]
    if family in ("count_end", "count_start", "number", "count_end2", "count_left"):
        opts = sorted(opts, key=int)
    else:
        rng.shuffle(opts)
    if opt_order is not None:
        opts = [answers_text[i] for i in range(4)]
        rng.shuffle(opts)
    fseed = rng.randrange(1 << 30)
    items = []
    for v in range(4):
        assert len(var_slots[v]) == len(var_slots[0])
        items.append(dict(family=family, kind="gist" if family in GIST else "detail", group=g, variant=v,
                          split=split, slots=var_slots[v], question=q, options=opts,
                          answer=opts.index(answers_text[v]), answer_text=answers_text[v], filler_seed=fseed))
    assert len(set(it["answer"] for it in items)) == 4
    return items


def bag(text):
    return sorted(re.findall(r"[a-z0-9\-']+", text.lower()))


# ------------------------------------------------------------------ token layout
def layout(items, tok, P=3072, W=256, D=0, gap=1):
    """Token streams for the 4 variants of a group.

    Returns per variant dict(past [P ids], window [W' ids], fact_spans [(a,b)] in past coords,
    slot_ids [list of id lists], sent_bounds [(a,b,is_fact)] for the past).
    The window and the post-fact filler are identical across variants (asserted)."""
    rng = random.Random(items[0]["filler_seed"])
    fill = filler_sentences(rng, 600)
    enc = lambda s: tok.encode(" " + s, add_special_tokens=False)  # noqa: E731
    fi = 0
    # window (whole sentences, >= W tokens)
    win, wsent = [], []
    while len(win) < W:
        ids = enc(fill[fi]); fi += 1
        win += ids
    # post-fact filler, ~D tokens (whole sentences); built backwards from the window
    post_s = []
    tot = 0
    while tot < D:
        ids = enc(fill[fi]); fi += 1
        post_s.insert(0, ids)
        tot += len(ids)
    nslot = len(items[0]["slots"])
    gaps = []
    for _ in range(nslot - 1):
        gg = []
        for _ in range(gap):
            gg.append(enc(fill[fi])); fi += 1
        gaps.append(gg)
    pre_pool = fill[fi:]
    out = []
    for it in items:
        sents = []  # (ids, is_fact)
        for k, s in enumerate(it["slots"]):
            sents.append((enc(s), True))
            if k < nslot - 1:
                for gg in gaps[k]:
                    sents.append((gg, False))
        for ids in post_s:
            sents.append((ids, False))
        core = sum(len(x) for x, _ in sents)
        assert core < P, (core, P)
        pre, j = [], 0
        need = P - core
        while sum(len(x) for x, _ in pre) < need:
            pre.insert(0, (enc(pre_pool[j % len(pre_pool)]), False)); j += 1
        excess = sum(len(x) for x, _ in pre) - need
        pre[0] = (pre[0][0][excess:], False)
        allp = pre + sents
        past, bounds, spans = [], [], []
        for ids, isf in allp:
            a = len(past)
            past += ids
            bounds.append((a, len(past), isf))
            if isf:
                spans.append((a, len(past)))
        assert len(past) == P
        out.append(dict(past=past, window=list(win), fact_spans=spans, sent_bounds=bounds,
                        slot_ids=[enc(s) for s in it["slots"]], D=tot, span_len=spans[-1][1] - spans[0][0]))
    return out


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "eval"
    for fam in FAMILIES:
        its = make_group(fam, 10000, split)
        print("=" * 20, fam, its[0]["question"], its[0]["options"])
        for it in its:
            print(f"  v{it['variant']} ans={LET[it['answer']]} ({it['answer_text']}):", " | ".join(it["slots"]))
        bags = [bag(" ".join(it["slots"])) for it in its]
        print("  identical bag of words across variants:", all(b == bags[0] for b in bags))
    print(len(FILLER), "filler sentences; sample:", FILLER[:3])
