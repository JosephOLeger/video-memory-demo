"""bench30 (29/10, agent bench30): TRAINING / dev stories for 30 classic memory question types, in varied
surface forms (narration, diary, chat log, email, dialogue, notes, care log), hidden in PG19 as BABILong does.

Same API as gen_babilong (make_story, layout, facts_of, student_ids, teacher_ids, judge, TRAIN_FAMS ...), so
train_bl.py / train_b30.py use it through TEXT_GEN=gen_bench30 (PYTHONPATH must hold this directory).

A story = 1 to 4 independent EPISODES (one question type each, disjoint people / objects), their fact segments
interleaved in time, then spread at PG19 sentence boundaries over the past. 4 counterfactual VARIANTS per story:
same background, same boundaries, same number of segments, only the values differ, and every question has a
different answer in (almost) every variant. Placeholder episodes are PRESENCE-BALANCED: the 4 options all occur in
the facts of every variant (variant v maps value slot i to option (i + v) % 4), so "which values were mentioned"
does not give the answer.

Questions are question-blind at write time (the trainer writes the past without the question).
BABILong stories (gen_babilong, bAbI TRAIN + PG19 train) are mixed in with probability B30_BL_FRAC (default 0.25),
so the official answer formats of qa1-qa4, qa7 stay trained. The official BABILong test is never touched here.
Held-out types (HELD_TYPES) are never drawn for split "train".
"""
from __future__ import annotations

import functools
import os
import random
import re

import gen_babilong as GB

HERE = os.path.dirname(os.path.abspath(__file__))
# re-exports used by the trainer (official eval, preservation pool)
BABI, FILES, parse_babi, BL = GB.BABI, GB.FILES, GB.parse_babi, GB.BL
LOCS, PEOPLE = GB.LOCS, GB.PEOPLE
SALT = "bench30-v1"
BL_FRAC = float(os.environ.get("B30_BL_FRAC", "0.25"))
MAX_FACT_FRAC = 0.4
MAX_SENT_TOK = 160

# ------------------------------------------------------------------------------------------------ pools
NAMES_F = ["Anna", "Clara", "Emma", "Grace", "Iris", "Kate", "Maria", "Nina", "Rosa", "Tina", "Wendy", "Zoe", "Lucy",
           "Sofia", "Ruth", "Mei", "Elena", "Julia", "Nora", "Sara", "Helen", "Alice", "Laura", "Olga"]
NAMES_M = ["Ben", "David", "Frank", "Henry", "Jack", "Leo", "Oscar", "Paul", "Sam", "Victor", "Yusuf", "Hugo", "Omar",
           "Pedro", "Tom", "Ken", "Ivan", "Max", "Raj", "Theo", "George", "Peter", "Carlos", "Simon"]
GENDER = {**{n: "f" for n in NAMES_F}, **{n: "m" for n in NAMES_M}}
NAMES = NAMES_F + NAMES_M
ROLES = ["Dr. Lee", "the nurse", "the pharmacist", "my daughter", "my son", "the neighbour", "the postman",
         "the physiotherapist", "my brother", "the carer", "the optician", "my granddaughter"]
ROOMS = ["kitchen", "bathroom", "bedroom", "hallway", "garden", "garage", "living room", "cellar", "attic", "balcony",
         "study", "porch"]
PLACES = ["pharmacy", "bakery", "park", "library", "post office", "market", "station", "church", "clinic", "bank",
          "cafe", "supermarket", "museum", "hairdresser", "town hall", "swimming pool"]
SPOTS = ["drawer", "top shelf", "kitchen table", "coat pocket", "bedside table", "fridge", "windowsill", "blue bag",
         "sofa", "bread bin", "glovebox", "sewing box", "fruit bowl", "shoe rack", "tool box", "biscuit tin"]
SPOT_PREP = {"drawer": "in the", "top shelf": "on the", "kitchen table": "on the", "coat pocket": "in the",
             "bedside table": "on the", "fridge": "in the", "windowsill": "on the", "blue bag": "in the",
             "sofa": "under the", "bread bin": "in the", "glovebox": "in the", "sewing box": "in the",
             "fruit bowl": "in the", "shoe rack": "on the", "tool box": "in the", "biscuit tin": "in the"}
OBJECTS = ["keys", "glasses", "wallet", "phone", "umbrella", "scarf", "remote control", "hearing aid", "pill box",
           "watch", "purse", "gloves", "notebook", "charger", "passport", "letter", "book", "hat", "ring", "torch",
           "walking stick", "mug", "photo album", "radio", "bus pass", "camera", "headphones", "thermos"]
THINGS_COL = ["car", "umbrella", "coat", "bicycle", "front door", "teapot", "armchair", "scarf", "suitcase", "kettle",
              "jumper", "handbag", "curtains", "mailbox"]
COLOURS = ["red", "blue", "green", "yellow", "black", "white", "brown", "purple", "orange", "grey", "pink"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["8 am", "9 am", "10 am", "11 am", "1 pm", "2 pm", "3 pm", "4 pm", "5 pm", "6 pm", "7 pm"]
PETS = ["Whiskers", "Biscuit", "Luna", "Pepper", "Ginger", "Shadow", "Coco", "Rocky", "Bella", "Milo", "Pickles",
        "Daisy"]
PET_KINDS = ["cat", "dog", "parrot", "rabbit", "goldfish", "tortoise", "hamster"]
FOODS = ["coffee", "salt", "sugar", "alcohol", "cheese", "chocolate", "fried food", "grapefruit", "red meat",
         "spicy food"]
DRINKS = ["tea", "coffee", "orange juice", "hot chocolate", "lemonade", "milk", "sparkling water", "apple juice",
          "cocoa"]
TOWNS = ["Leeds", "Bristol", "Dover", "York", "Bath", "Oxford", "Exeter", "Chester", "Norwich", "Durham", "Brighton",
         "Cardiff"]
LENT = ["ladder", "drill", "lawnmower", "tent", "suitcase", "camera", "bicycle pump", "sewing machine", "wheelbarrow",
        "cool box"]
CODE_KEYS = ["front door", "garage", "safe", "bike lock", "back gate", "locker", "alarm", "shed", "laptop", "storage unit"]
EVENTS_WHEN = ["Maria's visit", "the plumber's visit", "the hairdresser appointment", "the bin collection",
               "the book club", "the eye test", "the choir practice", "the grocery delivery", "the bank appointment",
               "the dentist appointment", "the family lunch", "the yoga class"]
ACTS = [("did the laundry", "laundry"), ("watered the plants", "plants"), ("called the bank", "bank"),
        ("went for a walk", "walk"), ("baked bread", "bread"), ("read the newspaper", "newspaper"),
        ("fed the cat", "cat"), ("paid the bills", "bills"), ("took a nap", "nap"), ("washed the dishes", "dishes"),
        ("wrote a letter", "letter"), ("ironed the shirts", "shirts"), ("cleaned the windows", "windows"),
        ("made soup", "soup"), ("did the crossword", "crossword"), ("polished the shoes", "shoes")]
ACT_KEY = dict(ACTS)
PROMISES = [("I'll bring the cake", "bring the cake", "cake"), ("I'll pick up the medicine", "pick up the medicine", "medicine"),
            ("I'll fix the gate", "fix the gate", "gate"), ("I'll book the taxi", "book the taxi", "taxi"),
            ("I'll water the roses", "water the roses", "roses"), ("I'll call the plumber", "call the plumber", "plumber"),
            ("I'll walk the dog", "walk the dog", "dog"), ("I'll buy the stamps", "buy the stamps", "stamps")]
TOPICS = {
    "a birthday party": (["birthday", "party"], ["We need balloons and a big cake.", "Everyone should arrive by six so we can surprise him.",
                                                 "Please do not tell him anything, it is a surprise.", "I have already ordered the candles."]),
    "a broken heater": (["heater", "heating", "radiator"], ["The radiator in the bedroom is cold again.",
                                                             "The repair man can only come next week.",
                                                             "It makes a loud banging noise at night.", "Keep a blanket close until the heating works."]),
    "a doctor's appointment": (["doctor", "appointment", "check"], ["Bring the list of your tablets.",
                                                                    "The surgery is on the second floor.",
                                                                    "Do not eat anything after midnight before the blood test.",
                                                                    "The doctor wants to check your blood pressure."]),
    "a holiday trip": (["holiday", "trip", "travel", "vacation"], ["The hotel is right by the sea.", "Pack the sun cream and the swimsuits.",
                                                                   "The train leaves very early on Saturday.", "I booked two rooms for the whole week."]),
    "a leaking roof": (["roof", "leak"], ["Water came through the ceiling during the storm.", "We put a bucket under the drip in the attic.",
                                          "Some tiles are missing near the chimney.", "The roofer will look at it on Friday."]),
    "a new puppy": (["puppy", "dog"], ["He chewed one of my slippers already.", "We bought a small basket and a lead.",
                                       "He needs his vaccinations next month.", "He sleeps in the kitchen at night."]),
    "a wedding": (["wedding", "marriage", "married"], ["The ceremony starts at two in the old church.", "I still need a hat for the day.",
                                                       "The bride chose white roses for the tables.", "There will be dancing after dinner."]),
    "an electricity bill": (["electricity", "bill", "energy"], ["The amount is much higher than last month.", "The meter reading looks wrong.",
                                                                "It has to be paid before the end of the month.", "I will call the supplier to complain."]),
}
TOPIC_LIST = list(TOPICS)
FORMS = [f for f in os.environ.get("B30_FORMS", "narration,diary,chat,email,dialogue,note,carelog").split(",")]  # opt-in subset
TIME_SLOTS = [("yesterday", 0), ("this morning", 1), ("at noon", 2), ("this afternoon", 3), ("this evening", 4)]
NUMW = ["none", "one", "two", "three", "four", "five"]


def cap(s):
    return s[0].upper() + s[1:] if s else s


def poss(n):
    return "my" if n == "I" else n + "'s"


def pron(n):
    return {"f": "she", "m": "he"}.get(GENDER.get(n), "they")


def spot(s):
    return f"{SPOT_PREP[s]} {s}"


# ------------------------------------------------------------------------------------------------ events
def ev(subj, t3, t1=None, speaker=None):
    """one fact segment: subj (a name, 'I' or None), third-person text, first-person text (said by subj)."""
    return {"subj": subj, "t3": t3, "t1": t1 or t3, "speaker": speaker}


def render(e, form, r, ctx):
    """ev -> one segment string in the episode's surface form. r: render rng (same draws in every variant)."""
    subj, t3, t1 = e["subj"], e["t3"], e["t1"]
    other = ctx["other"]
    hh = r.randrange(7, 22)
    mm = r.choice(["00", "05", "12", "20", "30", "41", "45", "52"])
    pick = r.random()
    pre = r.choice(["", "", "Later, ", "Then ", "After a while, ", "Some time later, ", "That day, "])
    if subj == "I":
        txt = t1
    else:
        txt = t3
    if form == "narration":
        return cap(pre + txt if pre and not txt.startswith(("Dr.",)) else txt) + "."
    if form == "diary":
        d = r.choice(["Dear diary, ", "Note for today: ", "", "", "Today ", "In the morning, ", "In the evening, "])
        return cap(d + txt) + "."
    if form == "chat":
        if e.get("speaker"):
            sp = e["speaker"]
            return f"[{hh:02d}:{mm}] {sp}: {cap(t1)}."
        if subj and subj != "I" and pick < 0.6:
            return f"[{hh:02d}:{mm}] {subj}: {cap(t1)}."
        sp = "Me" if subj == "I" else other
        return f"[{hh:02d}:{mm}] {sp}: {cap(txt)}."
    if form == "email":
        sender = e.get("speaker") or (subj if subj and subj != "I" else other)
        to = ctx["other2"] if sender != ctx["other2"] else other
        body = t1 if (e.get("speaker") or (subj and subj != "I")) else txt
        if pick < 0.5:
            return f"Email from {sender} to {to}: \"{cap(body)}.\""
        return f"{sender} wrote in an email: \"{cap(body)}.\""
    if form == "dialogue":
        sp = e.get("speaker") or (subj if subj and subj != "I" else None)
        if sp:
            verb = r.choice(["said", "told " + (other if other != sp else ctx["other2"]), "explained", "mentioned"])
            return f"\"{cap(t1)},\" {sp} {verb}."
        return f"\"{cap(txt)},\" I said to {other}."
    if form == "note":
        h = r.choice(["Note: ", "Reminder: ", "Note to self: ", "Remember: ", "To remember: "])
        if e.get("speaker"):
            return f"{h}{e['speaker']} said \"{t1}\"."
        return f"{h}{txt}."
    if form == "carelog":
        h = r.choice([f"Care log, {DAYS[r.randrange(7)]}: ", f"Care log {hh:02d}:{mm}: ", "Carer's note: "])
        if e.get("speaker"):
            return f"{h}{e['speaker']} said \"{t1}\"."
        return f"{h}{txt}."
    raise ValueError(form)


# --------------------------------------------------------------------------------------- episode generators
# each returns dict(events=[ev with '{V0}'..'{V3}' placeholders] (placeholder types) OR events4=[4 lists],
# question, answers=[4], support=[list of event idx per variant] (or one list for all), accept=[[...] per v],
# options (4 option texts), kind ('detail' / 'gist'), yn=True for yes/no style answers)

def _who(r, pool_names, k, allow_i=False):
    out = [pool_names.pop() for _ in range(k)]
    if allow_i and r.random() < 0.5:
        out[0] = "I"
    return out


def g_where_obj(r, E):
    objs = [E["obj"].pop() for _ in range(4)]
    sp = r.sample(SPOTS, 4)
    who = "I" if r.random() < 0.6 else E["name"].pop()
    others = [E["name"].pop() for _ in range(2)]
    put = r.choice([("put", "put"), ("left", "left"), ("placed", "placed"), ("stored", "stored")])[0]
    evs = []
    order = r.sample(range(4), 4)
    for i in order:
        s = who if (i == 0 or r.random() < 0.5) else r.choice(others)
        evs.append(ev(s, f"{s} {put} the {objs[i]} {{P{i}}}", f"I {put} the {objs[i]} {{P{i}}}"))
    vals = sp
    q = r.choice([f"Where did {who} put the {objs[0]}?", f"Where is the {objs[0]}?", f"Where can I find the {objs[0]}?",
                  f"Where was the {objs[0]} left?"])
    return dict(events=evs, slot_vals=[spot(x) for x in vals], answers_raw=vals, question=q,
                support=[order.index(0)], kind="detail")


def g_where_person(r, E):
    p = E["name"].pop()
    o = [E["name"].pop() for _ in range(2)]
    pl = r.sample(ROOMS if r.random() < 0.5 else PLACES, 4)
    mv = lambda s, i: ev(s, f"{s} {r.choice(['went to', 'moved to', 'walked to', 'headed to', 'went back to'])} the {{P{i}}}",  # noqa: E731
                         f"I {r.choice(['went to', 'moved to', 'walked to'])} the {{P{i}}}")
    evs = [mv(p, 2), mv(o[0], 1), mv(p, 3), mv(o[1], 2), mv(p, 0), mv(o[0], 3)]
    if r.random() < 0.5:
        evs = evs[:5] + [mv(o[1], 1)]
    q = r.choice([f"Where is {p}?", f"Where is {p} now?", f"Where did {p} go last?"])
    return dict(events=evs, slot_vals=pl, answers_raw=pl, question=q, support=[4], kind="detail")


def g_where_before(r, E):
    p, obj = E["name"].pop(), E["obj"].pop()
    pl = r.sample(ROOMS, 5)              # pl[4] = X (fixed, in the question); slots 0..3 vary
    # path: slot2 -> slot1? we need answer (slot 0) right before X. path = [V2, V1, V0, X, V3]
    evs = [ev(p, f"{p} went to the {{P2}} and picked up the {obj} there", f"I went to the {{P2}} and picked up the {obj} there"),
           ev(p, f"{p} carried the {obj} to the {{P1}}", f"I carried the {obj} to the {{P1}}"),
           ev(p, f"{p} took the {obj} into the {{P0}}", f"I took the {obj} into the {{P0}}"),
           ev(p, f"{p} went to the {pl[4]} with the {obj}", f"I went to the {pl[4]} with the {obj}"),
           ev(p, f"{p} brought the {obj} to the {{P3}}", f"I brought the {obj} to the {{P3}}")]
    q = f"Where was the {obj} before the {pl[4]}?"
    return dict(events=evs, slot_vals=pl[:4], answers_raw=pl[:4], question=q, support=[0, 2, 3], kind="detail")


def g_count_carried(r, E, list_mode=False):
    p = E["name"].pop()
    o = E["name"].pop()
    objs = [E["obj"].pop() for _ in range(5)]
    c0 = r.randrange(4)
    ev4, ans, acc, sup4 = [], [], [], []
    base_order = r.sample(range(4), 4)
    for v in range(4):
        c = (c0 + v) % 4
        mine = set(base_order[:c])
        evs = []
        for i in range(4):
            s = p if i in mine else o
            evs.append(ev(s, f"{s} {['picked up', 'took', 'grabbed', 'got'][i % 4]} the {objs[i]}",
                          f"I {['picked up', 'took', 'grabbed', 'got'][i % 4]} the {objs[i]}"))
        evs.insert(1, ev(p, f"{p} took the {objs[4]}", f"I took the {objs[4]}"))
        evs.insert(4, ev(p, f"{p} put down the {objs[4]}", f"I put down the {objs[4]}"))
        ev4.append(evs)
        held = [objs[i] for i in range(4) if i in mine]
        sup4.append([j for j, e in enumerate(evs) if e["subj"] == p])
        if list_mode:
            ans.append(" and ".join(held) if held else "nothing")
            acc.append(held if held else ["nothing"])
            univ = objs
        else:
            ans.append(NUMW[c])
            acc.append([str(c)])
    q = f"What is {p} carrying?" if list_mode else r.choice([f"How many objects is {p} carrying?",
                                                            f"How many things is {p} holding now?"])
    return dict(events4=ev4, answers=ans, accept=acc, question=q, support4=sup4, kind="detail", allkeys=list_mode,
                universe=objs if list_mode else None)


def g_coref_where(r, E):
    p = E["name"].pop()
    o = next(n for n in E["name"][::-1] if GENDER[n] != GENDER[p])
    E["name"].remove(o)
    pl = r.sample(ROOMS, 4)
    th = lambda: r.choice(["then", "after that", "afterwards", "later"])  # noqa: E731
    # one segment per person (the pronoun and its antecedent stay together whatever the interleaving)
    evs = [ev(p, f"{p} went to the {{P1}}. {cap(th())} {pron(p)} went to the {{P0}}",
              f"I went to the {{P1}}. {cap(th())} I went to the {{P0}}"),
           ev(o, f"{o} went to the {{P2}}. {cap(th())} {pron(o)} moved to the {{P3}}",
              f"I went to the {{P2}}. {cap(th())} I moved to the {{P3}}")]
    if r.random() < 0.5:
        evs = evs[::-1]
    q = f"Where is {p}?"
    return dict(events=evs, slot_vals=pl, answers_raw=pl, question=q, support=[evs.index(next(e for e in evs if e["subj"] == p))],
                kind="detail")


def g_time_where(r, E):
    p = E["name"].pop()
    pl = r.sample(PLACES, 5)
    xslot = r.randrange(1, 5)            # X at time slot xslot, answer at xslot - 1
    slots = list(range(5))
    rest = [s for s in slots if s not in (xslot, xslot - 1)]
    # values: slot xslot -> X (fixed), slot xslot-1 -> V0, others -> V1..V3
    assign = {xslot: "X", xslot - 1: "{P0}"}
    for k, s in enumerate(rest):
        assign[s] = "{P%d}" % (k + 1)
    order = r.sample(slots, 5)
    evs = []
    for s in order:
        tw = TIME_SLOTS[s][0]
        val = pl[4] if assign[s] == "X" else assign[s]
        evs.append(ev(p, f"{tw} {p} went to the {val}", f"{tw} I went to the {val}"))
    q = f"Where was {p} before the {pl[4]}?"
    return dict(events=evs, slot_vals=pl[:4], answers_raw=pl[:4], question=q,
                support=[order.index(xslot), order.index(xslot - 1)], kind="detail")


def _slot_attr(r, E, owners, fmt3, fmt1, vals, q, kind="detail"):
    order = r.sample(range(4), 4)
    evs = [ev(owners[i], fmt3(owners[i], "{P%d}" % i), fmt1(owners[i], "{P%d}" % i)) for i in order]
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[order.index(0)], kind=kind)


def g_colour(r, E):
    things = r.sample(THINGS_COL, 4)
    own = [E["name"].pop() for _ in range(4)]
    if r.random() < 0.4:
        own[0] = "I"
    vals = r.sample(COLOURS, 4)
    vb = r.choice(["is", "is painted", "is a shade of"])
    f3 = lambda n, v, t=None: f"{poss(n).capitalize() if n != 'I' else 'My'} {things[own.index(n)]} {vb} {v}"  # noqa: E731
    f1 = lambda n, v: f"my {things[own.index(n)]} {vb} {v}"  # noqa: E731
    who = own[0]
    q = f"What colour is {poss(who)} {things[0]}?"
    return _slot_attr(r, E, own, f3, f1, vals, q)


def g_number_code(r, E):
    keys = r.sample(CODE_KEYS, 4)
    vals = [str(x) for x in r.sample(range(1000, 9999), 4)]
    style = r.random()
    if style < 0.4:      # RULER-like needle
        f3 = lambda n, v: f"the special magic number for the {keys[own.index(n)]} is {v}"  # noqa: E731
        q = f"What is the special magic number for the {keys[0]}?"
    else:
        f3 = lambda n, v: r.choice([f"the code for the {keys[own.index(n)]} is {v}",  # noqa: E731
                                    f"the {keys[own.index(n)]} code is {v}"])
        q = r.choice([f"What is the code for the {keys[0]}?", f"What is the {keys[0]} code?"])
    own = ["k0", "k1", "k2", "k3"]
    out = _slot_attr(r, E, own, f3, f3, vals, q)
    for e in out["events"]:
        e["subj"] = None
    return out


def g_name_of(r, E):
    own = [E["name"].pop() for _ in range(4)]
    kinds = r.sample(PET_KINDS, 4)
    vals = r.sample(PETS, 4)
    f3 = lambda n, v: r.choice([f"{n}'s {kinds[own.index(n)]} is called {v}", f"{n}'s {kinds[own.index(n)]} is named {v}"])  # noqa: E731
    f1 = lambda n, v: f"my {kinds[own.index(n)]} is called {v}"  # noqa: E731
    q = r.choice([f"What is the name of {own[0]}'s {kinds[0]}?", f"What is {own[0]}'s {kinds[0]} called?"])
    return _slot_attr(r, E, own, f3, f1, vals, q)


def g_who_met(r, E):
    pl = r.sample(PLACES, 4)
    people = [E["name"].pop() for _ in range(2)] + r.sample(ROLES, 2)
    r.shuffle(people)
    order = r.sample(range(4), 4)
    vb = r.choice(["talked to", "met", "chatted with", "ran into", "spoke with"])
    evs = [ev("I", f"at the {pl[i]} I {vb} {{P{i}}}", f"at the {pl[i]} I {vb} {{P{i}}}") for i in order]
    q = r.choice([f"Who did I talk to at the {pl[0]}?", f"Who did I meet at the {pl[0]}?"])
    return dict(events=evs, slot_vals=people, answers_raw=people, question=q, support=[order.index(0)], kind="detail")


def g_what_said(r, E):
    sp = r.sample(ROLES, 4)
    vals = r.sample(FOODS, 4)
    order = r.sample(range(4), 4)
    evs = []
    for i in order:
        t = r.choice([f"{sp[i]} told me to avoid {{P{i}}}", f"{sp[i]} said I should avoid {{P{i}}}",
                      f"{sp[i]} advised me to avoid {{P{i}}}"])
        evs.append(ev(None, t, None))
    q = r.choice([f"What did {sp[0]} tell me to avoid?", f"According to {sp[0]}, what should I avoid?"])
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[order.index(0)], kind="detail")


def g_when_day(r, E):
    evn = r.sample(EVENTS_WHEN, 4)
    vals = r.sample(DAYS, 4)
    order = r.sample(range(4), 4)
    evs = [ev(None, r.choice([f"{evn[i]} is on {{P{i}}}", f"{evn[i]} will be on {{P{i}}}",
                               f"{evn[i]} was moved to {{P{i}}}"]), None) for i in order]
    q = f"On which day is {evn[0]}?"
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[order.index(0)], kind="detail")


def g_when_time(r, E):
    evn = r.sample(EVENTS_WHEN, 4)
    vals = r.sample(TIMES, 4)
    order = r.sample(range(4), 4)
    evs = [ev(None, r.choice([f"{evn[i]} is at {{P{i}}}", f"{evn[i]} starts at {{P{i}}}"]), None) for i in order]
    q = f"At what time is {evn[0]}?"
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[order.index(0)], kind="detail")


def g_how_many_times(r, E):
    p = E["name"].pop()
    oth = [E["name"].pop() for _ in range(2)]
    act = r.choice([("called", "call"), ("knocked on the door", "knock on the door"), ("sent a message", "send a message"),
                    ("rang the bell", "ring the bell")])
    c0 = r.randrange(4)
    slots = r.sample(range(6), 6)
    ev4, ans, acc, sup4 = [], [], [], []
    for v in range(4):
        c = (c0 + v) % 4
        mine = set(slots[:c])
        evs = []
        for i in range(6):
            s = p if i in mine else oth[i % 2]
            evs.append(ev(s, f"{s} {act[0]}", f"I {act[0]}"))
        ev4.append(evs)
        ans.append(NUMW[c]); acc.append([str(c)])
        sup4.append(sorted(mine) or [0])
    q = f"How many times did {p} {act[1]}?"
    return dict(events4=ev4, answers=ans, accept=acc, question=q, support4=sup4, kind="gist")


def g_order_first(r, E):
    p = "I" if r.random() < 0.5 else E["name"].pop()
    acts = r.sample(ACTS, 4)
    vals = [a[0] for a in acts]
    # sequence: V0 first, then V1, V2, V3 (order of slots fixed: the first event is always slot 0)
    seqw = ["First", "Next", "After that", "Finally"]
    evs = [ev(p, f"{seqw[i]} {p} {{P{i}}}" if p != "I" else f"{seqw[i]} I {{P{i}}}", f"{seqw[i]} I {{P{i}}}")
           for i in range(4)]
    if r.random() < 0.5:   # no ordinal words: order only by position
        evs = [ev(p, f"{p} {{P{i}}}", f"I {{P{i}}}") for i in range(4)]
    q = f"What did {p} do first?"
    out = dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[0, 1, 2, 3], kind="gist")
    out["keys"] = [ACT_KEY[x] for x in vals]
    return out


def g_what_after(r, E):
    p = "I" if r.random() < 0.5 else E["name"].pop()
    acts = r.sample(ACTS, 5)
    X = acts[4][0]
    vals = [a[0] for a in acts[:4]]
    xpos = r.randrange(0, 4)
    seq = []
    rest = ["{P1}", "{P2}", "{P3}"]
    for k in range(5):
        if k == xpos:
            seq.append(X)
        elif k == xpos + 1:
            seq.append("{P0}")
        else:
            seq.append(rest.pop(0))
    evs = [ev(p, f"{p} {a}", f"I {a}") for a in seq]
    q = f"What did {p} do just after {('I' if p == 'I' else pron(p))} {X}?"
    out = dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[xpos, xpos + 1], kind="detail")
    out["keys"] = [ACT_KEY[x] for x in vals]
    return out


def g_knowledge_update(r, E, prev=False):
    own = [E["name"].pop() for _ in range(3)]
    kind = r.choice(["phone extension", "room number", "locker number", "PIN"])
    vals = [str(x) for x in r.sample(range(100, 999), 4)]
    # own[0]: old value V1 then new V0 (prev=False) / old V0 then new V1 (prev=True); own[1] V2, own[2] V3
    a, b = ("{P1}", "{P0}") if not prev else ("{P0}", "{P1}")
    evs = [ev(own[0], f"{own[0]}'s {kind} is {a}", f"my {kind} is {a}"),
           ev(own[1], f"{own[1]}'s {kind} is {{P2}}", f"my {kind} is {{P2}}"),
           ev(own[0], r.choice([f"{own[0]}'s {kind} changed to {b}", f"{own[0]} has a new {kind}: {b}"]),
              r.choice([f"my {kind} changed to {b}", f"I have a new {kind}: {b}"])),
           ev(own[2], f"{own[2]}'s {kind} is {{P3}}", f"my {kind} is {{P3}}")]
    if r.random() < 0.5:
        evs[1], evs[3] = evs[3], evs[1]
    q = (f"What was {own[0]}'s {kind} before it changed?" if prev else
         r.choice([f"What is {own[0]}'s {kind} now?", f"What is {own[0]}'s current {kind}?"]))
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[0, 2], kind="detail")


def g_multi_hop(r, E):
    own = [E["name"].pop() for _ in range(4)]
    things = r.sample(LENT, 4)
    vals = r.sample(TOWNS, 4)
    lend = [ev(own[i], f"{own[i]} lent me the {things[i]}", f"I lent you the {things[i]}") for i in range(4)]
    live = [ev(own[i], f"{own[i]} lives in {{P{i}}}", f"I live in {{P{i}}}") for i in range(4)]
    ol, ov = r.sample(range(4), 4), r.sample(range(4), 4)
    evs = [lend[i] for i in ol[:2]] + [live[i] for i in ov[:2]] + [lend[i] for i in ol[2:]] + [live[i] for i in ov[2:]]
    sup = [evs.index(lend[0]), evs.index(live[0])]
    q = f"Where does the person who lent me the {things[0]} live?"
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=sup, kind="detail")


def g_most_frequent(r, E):
    p = "I" if r.random() < 0.6 else E["name"].pop()
    fr = [E["name"].pop() for _ in range(4)]
    vb = r.choice([("visited", "visit"), ("phoned", "phone"), ("had lunch with", "have lunch with")])
    seq = ["{P0}"] * 3 + ["{P1}"] * 2 + ["{P2}", "{P3}"]
    r.shuffle(seq)
    evs = [ev(p, f"{p} {vb[0]} {x}", f"I {vb[0]} {x}") for x in seq]
    q = f"Who did {p} {vb[1]} most often?"
    return dict(events=evs, slot_vals=fr, answers_raw=fr, question=q, support=[i for i, x in enumerate(seq) if x == "{P0}"],
                kind="gist")


def g_variable_track(r, E):
    own = [E["name"].pop() for _ in range(5)]
    kind = r.choice(["locker code", "gate code", "wifi code"])
    vals = [str(x) for x in r.sample(range(1000, 9999), 4)]
    evs = [ev(own[0], f"{own[0]}'s {kind} is {{P0}}", f"my {kind} is {{P0}}"),
           ev(own[2], f"{own[2]}'s {kind} is {{P1}}", f"my {kind} is {{P1}}"),
           ev(own[1], f"{own[1]} uses the same {kind} as {own[0]}", f"I use the same {kind} as {own[0]}"),
           ev(own[3], f"{own[3]}'s {kind} is {{P2}}", f"my {kind} is {{P2}}"),
           ev(own[4], f"{own[4]}'s {kind} is {{P3}}", f"my {kind} is {{P3}}")]
    q = f"What is {own[1]}'s {kind}?"
    return dict(events=evs, slot_vals=vals, answers_raw=vals, question=q, support=[0, 2], kind="detail")


NOTM = ["not mentioned", "not say", "not said", "no information", "unknown", "not stated", "not mention", "no mention",
        "not specified", "not known", "cannot be determined", "not given", "does not say", "doesn't say", "not provided",
        "no record", "not clear", "isn't mentioned", "is not mentioned", "never mentioned", "don't know", "do not know"]


def g_abstention(r, E):
    own = [E["name"].pop() for _ in range(4)]
    thing = r.choice(THINGS_COL)
    col = r.sample(COLOURS, 4)
    ev4, ans, acc = [], [], []
    for v in range(4):
        cs = col[v:] + col[:v]
        names = list(own)
        if v == 2:                       # the asked person's item is NOT mentioned (another person's instead)
            names[0] = E["spare"]
        evs = [ev(names[i], f"{names[i]}'s {thing} is {cs[i]}", f"my {thing} is {cs[i]}") for i in range(4)]
        ev4.append(evs)
        if v == 2:
            ans.append("not mentioned"); acc.append(NOTM)
        else:
            ans.append(cs[0]); acc.append([cs[0]])
    q = f"What colour is {own[0]}'s {thing}?"
    return dict(events4=ev4, answers=ans, accept=acc, question=q, support4=[[0]] * 4, kind="detail", notm=True)


def g_gist_topic(r, E):
    snd, o1 = E["name"].pop(), E["name"].pop()
    tops = r.sample(TOPIC_LIST, 2 + 2)
    ev4, ans, acc = [], [], []
    for v in range(4):
        t_main, t_oth = tops[v], tops[(v + 1) % 4]
        cue_m = TOPICS[t_main][1][:3]
        cue_o = TOPICS[t_oth][1][:2]
        evs = [ev(None, f"{c[:-1]}", None, speaker=snd) for c in cue_m] + \
              [ev(None, f"{c[:-1]}", None, speaker=o1) for c in cue_o]
        evs = [evs[0], evs[3], evs[1], evs[2], evs[4]]
        ev4.append(evs)
        ans.append(t_main); acc.append(TOPICS[t_main][0])
    q = r.choice([f"What was {snd}'s message about?", f"What was {snd} talking about?"])
    return dict(events4=ev4, answers=ans, accept=acc, question=q, support4=[[0, 2, 3]] * 4, kind="gist",
                force_speaker=True)


def g_who_gave(r, E):                    # held out (bAbI qa5 analogue)
    p = E["name"].pop()
    rec = [E["name"].pop() for _ in range(3)] + [r.choice(ROLES)]
    objs = [E["obj"].pop() for _ in range(4)]
    g2 = E["name"].pop()
    evs = [ev(p, f"{p} gave the {objs[0]} to {{P0}}", f"I gave the {objs[0]} to {{P0}}"),
           ev(p, f"{p} handed the {objs[1]} to {{P1}}", f"I handed the {objs[1]} to {{P1}}"),
           ev(g2, f"{g2} gave the {objs[2]} to {{P2}}", f"I gave the {objs[2]} to {{P2}}"),
           ev(p, f"{p} passed the {objs[3]} to {{P3}}", f"I passed the {objs[3]} to {{P3}}")]
    order = r.sample(range(4), 4)
    evs = [evs[i] for i in order]
    q = f"Who did {p} give the {objs[0]} to?"
    return dict(events=evs, slot_vals=rec, answers_raw=rec, question=q, support=[order.index(0)], kind="detail")


def g_yes_no(r, E):                      # held out (qa6)
    p, o = E["name"].pop(), E["name"].pop()
    X, Y, Z = r.sample(ROOMS, 3)
    mv = lambda s, l: ev(s, f"{s} went to the {l}", f"I went to the {l}")  # noqa: E731
    seqs = [[mv(p, Y), mv(o, Z), mv(p, X)], [mv(p, X), mv(o, Z), mv(p, Y)],
            [mv(p, Z), mv(o, Y), mv(p, X)], [mv(p, X), mv(o, X), mv(p, Z)]]
    return dict(events4=seqs, answers=["yes", "no", "yes", "no"], accept=[["yes"], ["no"], ["yes"], ["no"]],
                question=f"Is {p} in the {X}?", support4=[[2]] * 4, kind="detail", yn=True)


def g_negation(r, E):                    # held out (qa9)
    p, o = E["name"].pop(), E["name"].pop()
    X, Y = r.sample(ROOMS, 2)
    seqs = [[ev(p, f"{p} went to the {X}", f"I went to the {X}"), ev(o, f"{o} is not in the {Y}", f"I am not in the {Y}")],
            [ev(p, f"{p} went to the {X}", f"I went to the {X}"), ev(p, f"{p} is no longer in the {X}", f"I am no longer in the {X}")],
            [ev(p, f"{p} is not in the {Y}", f"I am not in the {Y}"), ev(p, f"{p} went to the {X}", f"I went to the {X}")],
            [ev(o, f"{o} went to the {X}", f"I went to the {X}"), ev(p, f"{p} is not in the {X}", f"I am not in the {X}")]]
    return dict(events4=seqs, answers=["yes", "no", "yes", "no"], accept=[["yes"], ["no"], ["yes"], ["no"]],
                question=f"Is {p} in the {X}?", support4=[[0], [0, 1], [1], [1]], kind="detail", yn=True)


def g_indefinite(r, E):                  # held out (qa10)
    p, o = E["name"].pop(), E["name"].pop()
    X, Y, Z = r.sample(PLACES, 3)
    seqs = [[ev(p, f"{p} is either in the {X} or the {Y}", f"I am either in the {X} or the {Y}"), ev(o, f"{o} went to the {Z}", f"I went to the {Z}")],
            [ev(p, f"{p} went to the {X}", f"I went to the {X}"), ev(o, f"{o} is either in the {Y} or the {Z}", f"I am either in the {Y} or the {Z}")],
            [ev(p, f"{p} is either in the {Y} or the {Z}", f"I am either in the {Y} or the {Z}"), ev(o, f"{o} went to the {X}", f"I went to the {X}")],
            [ev(o, f"{o} is either in the {X} or the {Y}", f"I am either in the {X} or the {Y}"), ev(p, f"{p} went to the {Z}", f"I went to the {Z}")]]
    return dict(events4=seqs, answers=["maybe", "yes", "no", "no"], accept=[["maybe"], ["yes"], ["no"], ["no"]],
                question=f"Is {p} in the {X}?", support4=[[0], [0], [0], [1]], kind="detail", yn=True)


def g_preference(r, E):                  # held out
    own = [E["name"].pop() for _ in range(4)]
    if r.random() < 0.3:
        own[0] = r.choice(["Grandpa", "Grandma"])
    vals = r.sample(DRINKS, 4)
    f3 = lambda n, v: r.choice([f"{n} prefers {v}", f"{n} always asks for {v}", f"{n}'s favourite drink is {v}"])  # noqa: E731
    f1 = lambda n, v: r.choice([f"I prefer {v}", f"my favourite drink is {v}"])  # noqa: E731
    q = r.choice([f"What drink does {own[0]} prefer?", f"Which drink does {own[0]} like best?"])
    return _slot_attr(r, E, own, f3, f1, vals, q)


def g_speaker(r, E):                     # held out (InfiniteBench En.Dia / LoCoMo)
    sp = [E["name"].pop() for _ in range(4)]
    pr = r.sample(PROMISES, 4)
    order = r.sample(range(4), 4)
    evs = [ev(None, f"{{P{i}}} said they would {pr[i][1]}", pr[i][0], speaker="{P%d}" % i) for i in order]
    q = f"Who said they would {pr[0][1]}?"
    return dict(events=evs, slot_vals=sp, answers_raw=sp, question=q, support=[order.index(0)], kind="detail",
                force_speaker=True)


TYPES = {   # name: (generator, source benchmarks, held_out)
    "where_obj": (g_where_obj, "Ego4D NLQ 'where did I put X' / BABILong qa2", False),
    "where_person": (g_where_person, "BABILong qa1 (single supporting fact)", False),
    "where_before": (g_where_before, "BABILong qa3 (three supporting facts)", False),
    "count_carried": (g_count_carried, "BABILong qa7 (counting)", False),
    "coref_where": (g_coref_where, "BABILong qa11 / qa13 (coreference)", False),
    "time_where": (g_time_where, "BABILong qa14 (time reasoning)", False),
    "colour": (g_colour, "LongMemEval single-session-user / Ego4D 'what colour was X'", False),
    "number_code": (g_number_code, "RULER NIAH single / multikey, LongBench passkey / KV retrieval", False),
    "name_of": (g_name_of, "LoCoMo single-hop", False),
    "who_met": (g_who_met, "Ego4D NLQ 'who did I talk to' / LoCoMo single-hop", False),
    "what_said": (g_what_said, "LoCoMo single-hop / LongMemEval single-session-assistant", False),
    "when_day": (g_when_day, "LoCoMo temporal / LongMemEval temporal-reasoning", False),
    "when_time": (g_when_time, "LongMemEval single-session-user (appointments)", False),
    "how_many_times": (g_how_many_times, "LongMemEval multi-session (counting) / Ego4D 'how many times'", False),
    "order_first": (g_order_first, "LoCoMo temporal / EgoSchema order of events", False),
    "knowledge_update": (g_knowledge_update, "LongMemEval knowledge-update (latest value)", False),
    "previous_value": (lambda r, E: g_knowledge_update(r, E, prev=True), "LongMemEval knowledge-update ('what changed')", False),
    "multi_hop": (g_multi_hop, "LoCoMo multi-hop / HotpotQA-style two-hop", False),
    "most_frequent": (g_most_frequent, "RULER frequent-words extraction (aggregation), transposed", False),
    "abstention": (g_abstention, "LongMemEval abstention / LoCoMo adversarial", False),
    "gist_topic": (g_gist_topic, "LongBench / QMSum gist, LoCoMo event summary (topic)", False),
    "variable_track": (g_variable_track, "RULER variable tracking", False),
    "who_gave": (g_who_gave, "BABILong qa5 (three-argument relations) / Ego4D 'who did I give X to'", True),
    "yes_no": (g_yes_no, "BABILong qa6 (yes/no)", True),
    "list_carried": (lambda r, E: g_count_carried(r, E, list_mode=True), "BABILong qa8 (lists / sets)", True),
    "negation": (g_negation, "BABILong qa9 (simple negation)", True),
    "indefinite": (g_indefinite, "BABILong qa10 (indefinite knowledge)", True),
    "what_after": (g_what_after, "EgoSchema / TVQA 'what happened right after X'", True),
    "preference": (g_preference, "LongMemEval single-session-preference", True),
    "speaker": (g_speaker, "InfiniteBench En.Dia / LoCoMo 'who said X'", True),
}
TRAIN_TYPES = [k for k, v in TYPES.items() if not v[2]]
HELD_TYPES = [k for k, v in TYPES.items() if v[2]]
FAMILIES = list(TYPES)
GIST = ["ep"]
DETAIL = []
TRAIN_FAMS = {"A": TRAIN_TYPES, "B": TRAIN_TYPES}
HELD_OUT = None


# ----------------------------------------------------------------------------------------------- stories
def _sub(s, vals):
    return re.sub(r"\{P(\d)\}", lambda m: vals[int(m.group(1))], s)


def make_episode(r, typ, E, split):
    gen = TYPES[typ][0]
    d = gen(r, E)
    form = r.choice(FORMS)
    if d.get("force_speaker") and form in ("narration", "diary"):
        form = r.choice([f for f in ("chat", "dialogue", "email") if f in FORMS] or ["dialogue"])
    ctx = {"other": E["name"].pop(), "other2": E["name"].pop()}
    rseed = r.randrange(1 << 30)
    facts4, answers, accept, sup4 = [], [], [], []
    if "events" in d:
        sv = d["slot_vals"]
        for v in range(4):
            vals = [sv[(i + v) % 4] for i in range(4)]
            rr = random.Random(rseed)
            evs = []
            for e in d["events"]:
                e2 = {k: (_sub(x, vals) if isinstance(x, str) else x) for k, x in e.items()}
                evs.append(e2)
            facts4.append([render(e, form, rr, ctx) for e in evs])
            ai = v % 4                      # slot 0 -> option v
            ans = d["answers_raw"][ai]
            answers.append(ans)
            keys = d.get("keys")
            accept.append([ans] + ([keys[ai]] if keys else []))
            sup4.append(d["support"])
        opts = list(d["answers_raw"])
    else:
        for v in range(4):
            rr = random.Random(rseed)
            facts4.append([render(e, form, rr, ctx) for e in d["events4"][v]])
        answers, accept, sup4 = d["answers"], d["accept"], d["support4"]
        opts = list(dict.fromkeys(answers))
        pad = {"yn": ["yes", "no", "maybe", "not mentioned"]}.get("yn" if d.get("yn") else "", [])
        for x in pad + ["not mentioned", "nothing", "none", "unknown"]:
            if len(opts) >= 4:
                break
            if x not in opts:
                opts.append(x)
    return dict(type=typ, form=form, facts4=facts4, answers=answers, accept=accept, support4=sup4,
                question=d["question"], options=opts[:4], kind=d["kind"], yn=bool(d.get("yn")),
                allkeys=bool(d.get("allkeys")), universe=d.get("universe"))


N_EP_W = [(int(k), float(w)) for k, w in (x.split(":") for x in
          os.environ.get("B30_NEP", "1:0.25,2:0.35,3:0.25,4:0.15").split(","))]   # episodes per story (load)


def make_story(g: int, split: str = "eval", types=None, n_ep=None):
    r = random.Random(f"{SALT}-{split}-{g}")
    if split == "train" and types is None and r.random() < BL_FRAC:
        st = GB.make_story(g, split)
        st["src"] = "bl"
        return st
    pool = types or (TRAIN_TYPES if split == "train" else FAMILIES)
    if n_ep is None:
        x, n_ep = r.random(), 1
        for k, w in N_EP_W:
            x -= w
            if x <= 0:
                n_ep = k
                break
    if types is not None:                 # explicit episode types, in this order (eval files)
        typs = list(types)
    elif split == "train":
        typs = [r.choice(pool) for _ in range(n_ep)]
    else:
        typs = [pool[g % len(pool)]] + [r.choice(pool) for _ in range(n_ep - 1)]
    E = {"name": r.sample(NAMES, len(NAMES)), "obj": r.sample(OBJECTS, len(OBJECTS))}
    E["spare"] = E["name"].pop()
    eps = []
    for t in typs:
        if len(E["name"]) < 10 or len(E["obj"]) < 6:
            break
        eps.append(make_episode(r, t, E, split))
    # interleave the episodes' segments in time (same merge for every variant)
    seq = [j for j, e in enumerate(eps) for _ in e["facts4"][0]]
    r.shuffle(seq)
    # keep within-episode order: the k-th occurrence of j is segment k of episode j
    pos_of = {}
    cnt = [0] * len(eps)
    merged_idx = []
    for j in seq:
        merged_idx.append((j, cnt[j]))
        pos_of[(j, cnt[j])] = len(merged_idx) - 1
        cnt[j] += 1
    facts = [[eps[j]["facts4"][v][k] for j, k in merged_idx] for v in range(4)]
    items = {}
    for j, e in enumerate(eps):
        fam = f"ep#{e['type']}#{j}"
        its = []
        for v in range(4):
            sup = sorted(pos_of[(j, k)] for k in e["support4"][v])
            its.append(dict(family=fam, task=e["type"], kind=e["kind"], group=g, variant=v, split=split,
                            question=e["question"], options=e["options"], answer=e["options"].index(e["answers"][v])
                            if e["answers"][v] in e["options"] else -1,
                            answer_text=e["answers"][v], target_text=e["answers"][v], accept=e["accept"][v],
                            option_texts=e["options"], n_facts=len(facts[0]), support=sup, form=e["form"],
                            yn=e["yn"], allkeys=e["allkeys"], universe=e["universe"], held=TYPES[e["type"]][2],
                            contrast=e["answers"][v] != e["answers"][(v + 1) % 4]))
        items[fam] = its
    return dict(group=g, split=split, task="ep", src="b30", items=items, facts=facts, episodes=[
        {"type": e["type"], "form": e["form"]} for e in eps],
        rng_seed=r.randrange(1 << 30), bg_seed=r.randrange(1 << 30))


# ----------------------------------------------------------------------------------------------- layout
def _place(story, tok, P, W, D, margin, official):
    enc = lambda s: tok.encode(" " + s, add_special_tokens=False)  # noqa: E731
    fids = [[enc(f) for f in fv] for fv in story["facts"]]
    F = [sum(len(x) for x in fv) for fv in fids]
    if max(F) > MAX_FACT_FRAC * P:
        raise ValueError(f"facts {max(F)} tokens > {MAX_FACT_FRAC} x P {P}")
    sup_all = sorted({i for its in story["items"].values() for i in its[0]["support"]})
    if official:            # facts uniformly over past + window (as the official BABILong), window per variant
        st = dict(story, task="ep")
        st["items"] = {"ep": [dict(support=sup_all)] * 4}
        return _official(st, tok, fids, F, P, W, margin)
    bg = GB.background(story, tok, P - min(F) + W + 2 * margin)
    toks, starts = [], []
    for s in bg:
        starts.append(len(toks))
        toks += s
    end = len(toks) - W
    s_lo = end - (P - max(F)) + margin
    elig = [b for b in starts if s_lo <= b <= end - D]
    if not elig:
        elig = [b for b in starts if s_lo <= b < end - 32] or [b for b in starts if s_lo <= b < end] or [end]
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
                bounds.append((q0, len(past), "ep"))
            if b_ > a_:
                q0 = len(past)
                past += toks[a_:b_]
                bounds.append((q0, len(past), "scen"))
        for i in at.get(end, []):
            q0 = len(past)
            past += fids[v][i]
            bounds.append((q0, len(past), "ep"))
        assert len(past) == P, (len(past), P)
        fb = [(a_, b_) for a_, b_, k in bounds if k == "ep"]
        out.append(dict(past=past, window=toks[end:end + W], bounds=bounds, D=P - fb[-1][1] if fb else P,
                        ep=(fb[0][0], fb[-1][1]) if fb else (0, 0),
                        support_ends=[fb[i][1] - 1 for i in sup_all if i < len(fb)] or [fb[-1][1] - 1],
                        n_fact_past=len(fb), n_fact_win=0))
    return out


def _official(st, tok, fids, F, P, W, margin):
    T = P + W
    bg = GB.background(st, tok, T - min(F) + 2 * margin)
    toks, starts = [], []
    for s in bg:
        starts.append(len(toks))
        toks += s
    hi = T - max(F)
    elig = [b for b in starts if b <= hi] + ([hi] if hi not in starts else [])
    r = random.Random(st["rng_seed"] * 37 + P + 1)
    chosen = sorted(r.choice(elig) for _ in range(len(fids[0])))
    at = {}
    for i, b in enumerate(chosen):
        at.setdefault(b, []).append(i)
    sup_all = st["items"]["ep"][0]["support"]
    out = []
    for v in range(4):
        nb = T - F[v]
        pts = sorted({0, nb} | {b for b in starts if 0 < b < nb} | {b for b in at if b <= nb})
        seq, bnd = [], []
        for a_, b_ in zip(pts, pts[1:] + [None]):
            for i in at.get(a_, []):
                q0 = len(seq)
                seq += fids[v][i]
                bnd.append((q0, len(seq), "ep"))
            if b_ is not None and b_ > a_:
                q0 = len(seq)
                seq += toks[a_:b_]
                bnd.append((q0, len(seq), "scen"))
        assert len(seq) == T, (len(seq), T)
        past, window = seq[:P], seq[P:]
        bounds = [(a_, min(b_, P), k) for a_, b_, k in bnd if a_ < P]
        fall = [(a_, b_) for a_, b_, k in bnd if k == "ep"]
        n_past = sum(1 for a_, b_ in fall if b_ <= P)
        fb = [(a_, b_) for a_, b_, k in bounds if k == "ep"]
        se = [fb[i][1] - 1 for i in sup_all if i < len(fb)] or ([fb[-1][1] - 1] if fb else [P - 1])
        out.append(dict(past=past, window=window, bounds=bounds, D=P - fb[-1][1] if fb else P,
                        ep=(fb[0][0], fb[-1][1]) if fb else (0, 0), support_ends=se,
                        n_fact_past=n_past, n_fact_win=len(fall) - n_past))
    return out


OFFICIAL_FRAC = float(os.environ.get("B30_OFFICIAL_FRAC", "0.3"))


def layout(story, tok, P=4096, W=256, D=0, margin=16, official=False):
    """bench30 stories: by default every fact segment lies in the past, >= ~D tokens before the window (memory
    questions, nothing readable from the window). official=True: with probability B30_OFFICIAL_FRAC the
    BABILong-official placement (facts anywhere in past + window; the trainer masks the margin then), else the
    past-only placement with D. BABILong stories are delegated to gen_babilong."""
    if story.get("src") == "bl":
        return GB.layout(story, tok, P=P, W=W, D=D, margin=margin, official=official)
    if official:
        off = random.Random(story["rng_seed"] * 7 + P).random() < OFFICIAL_FRAC
        return _place(story, tok, P, W, D, margin, off)
    return _place(story, tok, P, W, D, margin, False)


def facts_of(lay, fam):
    fam = fam.split("#")[0]
    return [(a, b) for a, b, k in lay["bounds"] if k == fam]


STOP = set("""a an the and or to of in on at by with for from into there then that this it its is are was were be been i me my you
your he she they him her them his their said told explained mentioned wrote email note reminder remember self dear diary today
log care carer's later after while some time day morning evening afternoon noon yesterday""".split())


def content_mask(tok, ids):
    """opt-in (train_b30 --b30-content): True for content tokens of a fact span (names, objects, places, values,
    digits), False for function words, punctuation, timestamps scaffolding words."""
    out = []
    for t in ids:
        w = tok.decode([t]).strip().lower()
        out.append(bool(w) and any(ch.isalnum() for ch in w) and w not in STOP)
    return out


# ----------------------------------------------------------------------------------------------- prompts
SYS = GB.SYS
END = GB.END
PRE = ("You read a very long text in pieces. Only its most recent part is shown below; earlier parts are not shown.\n"
       "Answer the question about the whole text with a few words only.\n\n<context>\n")


def _post(q):
    return f"\n</context>\n\nQuestion: {q}"


def student_ids(tok, win, it):
    if str(it.get("task", "")).startswith("qa"):
        return GB.student_ids(tok, win, it)
    return tok.encode(SYS + PRE, add_special_tokens=False) + list(win) + \
        tok.encode(_post(it["question"]) + END, add_special_tokens=False)


def teacher_ids(tok, notes, win, it):
    if str(it.get("task", "")).startswith("qa"):
        return GB.teacher_ids(tok, notes, win, it)
    return tok.encode(SYS + PRE, add_special_tokens=False) + list(notes) + tok.encode("\n...", add_special_tokens=False) + \
        list(win) + tok.encode(_post(it["question"]) + END, add_special_tokens=False)


NUMWORDS = {"zero": "0", "none": "0", "no": "0", "one": "1", "once": "1", "two": "2", "twice": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "thrice": "3", "nothing": "0"}


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(\d{1,2})(?::00)?\s*([ap])\.?\s*m\.?\b", lambda m: f"{int(m.group(1))} {m.group(2)}m", s)
    s = re.sub(r"\b(\d{1,2}):00\b", lambda m: (f"{int(m.group(1)) - 12} pm" if int(m.group(1)) > 12 else
                                               f"{int(m.group(1))} am"), s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in ("the", "a", "an", "at", "in", "on", "her", "his", "my", "of"))


def judge(it, output):
    if str(it.get("task", "")).startswith("qa"):
        return GB.judge(it, output)
    o = " " + norm(output) + " "
    acc = it.get("accept") or [it["answer_text"]]
    if it.get("yn"):
        w = norm(output).split()
        return bool(w) and w[0] == norm(acc[0])
    if it.get("answer_text", "").isalpha() and it.get("answer_text") in NUMW and acc and acc[0].isdigit():
        ws = [NUMWORDS.get(x, x) for x in norm(output).split()]
        return acc[0] in ws
    if acc and all(x.isdigit() for x in acc) and len(acc) == 1 and len(acc[0]) == 1:
        ws = [NUMWORDS.get(x, x) for x in norm(output).split()]
        return acc[0] in ws
    if it.get("allkeys"):          # list answer: the set of named objects must be exactly the gold set
        named = {x for x in (it.get("universe") or acc) if f" {norm(x)} " in o}
        if acc == ["nothing"]:
            return not named and any(f" {w} " in o for w in ("nothing", "none", "no", "0", "empty"))
        return named == set(acc)
    return any(norm(x) and f" {norm(x)} " in o for x in acc)


# ----------------------------------------------------------------------------------------------- checks
def _checks(n=60, tok=None):
    import collections
    bad = collections.Counter()
    stats = collections.defaultdict(collections.Counter)
    for typ in FAMILIES:
        for g in range(n):
            st = make_story(900_000 + g, "eval", types=[typ], n_ep=1)
            (fam, its), = st["items"].items()
            ans = [x["answer_text"] for x in its]
            stats[typ]["distinct_" + str(len(set(ans)))] += 1
            for v in range(4):
                fv = " ".join(st["facts"][v]).lower()
                it = its[v]
                if not it["yn"] and it["answer_text"] not in ("not mentioned", "nothing") and \
                        it["answer_text"] not in NUMW and it["answer_text"].lower() not in fv:
                    bad[(typ, "answer_absent")] += 1
                # presence balance: all 4 options in the facts of every variant (placeholder types)
                if typ not in ("count_carried", "list_carried", "how_many_times", "yes_no", "negation", "indefinite",
                               "abstention", "gist_topic"):
                    if not all(o.lower() in fv for o in it["options"]):
                        bad[(typ, "presence_unbalanced")] += 1
                if not judge(it, it["answer_text"]):
                    bad[(typ, "judge_rejects_gold")] += 1
                for o in it["options"]:
                    if o != it["answer_text"] and judge(it, o) and not it["yn"]:
                        bad[(typ, "judge_accepts_wrong_option")] += 1
            if len(set(len(f) for f in st["facts"])) != 1:
                bad[(typ, "unequal_n_facts")] += 1
            stats[typ]["form_" + its[0]["form"]] += 1
    return bad, stats


if __name__ == "__main__":
    import sys
    bad, stats = _checks(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    print("TYPES", len(FAMILIES), "train", len(TRAIN_TYPES), "held", len(HELD_TYPES))
    print("CHECK FAILURES:", dict(bad) or "none")
    for t, c in stats.items():
        print(f"  {t:18s}", dict(sorted(c.items())))
    for typ in FAMILIES:
        st = make_story(123, "eval", types=[typ], n_ep=1)
        (fam, its), = st["items"].items()
        print(f"== {typ} [{its[0]['form']}] Q: {its[0]['question']} | answers {[x['answer_text'] for x in its]} | sup {its[0]['support']}")
        for s in st["facts"][0]:
            print("     ", s)
