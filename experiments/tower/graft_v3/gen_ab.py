"""23/10 : A/B "gist only" vs "everything" -- rich training tableaux + dense free-form questions.

Training scenes (split TRAIN only, seed salt "ab-train"): ONE rich tableau per variant, built
in the v3c "unit" style (persons + decoy stacks whose colour histogram equals a person's, so
colour mass does not give the count), plus every v3 element:
  persons n (1..8, distinct shirt colours), decoys (0..units-n), door open/closed, light on/off,
  big object (ball/box/cone/star, 60 %), small number sign (60 %), one hat (60 %, train hat
  colours only), one pair of coloured shoes (60 %), a small poster with a shape (60 %).
A group = 4 variants sharing decor (room), filler, window and D; the contents are independent.

Questions are generated from the generator's facts (ground truth known), short FREE answers
("3", "red", "ball", "open"), several paraphrases per family. Partition (lead, 23/10):
  gist   (arms A and B): count, colour, big_object, room_state (door, light, both)
  detail (arm B only)  : sign_number, hat_color, shoe_color, poster_shape
  side   : never trained (held-out gist family, eval only)
Eval paraphrases are held out from training (TRAIN_Q vs EVAL_Q).

    python gen_ab.py --contact contact_ab.png      # 2 groups x 4 variants, facts + questions
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
V3C_DIR = os.environ.get("AB_V3C_DIR", os.path.join(HERE, "ab_gen"))   # frozen copy of scenes/
sys.path.insert(0, V3C_DIR)
import gen_tableau as T            # noqa: E402  (v3, md5 ddf4e1d1)
import gen_tableau_v3c as C3       # noqa: E402

G = T.G
SIZE, FPS, E = T.SIZE, T.FPS, T.E
CNAMES = T.CNAMES
LET = "ABCD"
OBJECTS = T.OBJECTS
SHAPES = ["circle", "triangle", "square", "cross"]
HAT_TRAIN = [c for c in CNAMES if c not in G.HAT_HELDOUT]
GIST_FAMS = ["count", "colour", "big_object", "room_state"]
DETAIL_FAMS = ["sign_number", "hat_color", "shoe_color", "poster_shape"]
GEN_MD5 = hashlib.md5(open(T.__file__, "rb").read() + open(G.__file__, "rb").read()
                      + open(C3.__file__, "rb").read()).hexdigest()[:8]


@dataclass
class TableauAB:
    scene: G.Scene
    units: list            # C3.Unit (x, k, colour, person)
    feet: int
    door: bool
    light: bool
    obj: str | None
    obj_x: int
    sign: int | None
    hat: tuple | None      # (unit index, colour)
    shoes: tuple | None    # (unit index, colour)
    poster: tuple | None   # (x, y, shape)
    style: T.TStyle = field(default_factory=lambda: T.TStyle(count_gap=True, door_bright=True))


def _draw_shape(dr, shape, cx, cy, r, col=(20, 20, 20)):
    if shape == "circle":
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "square":
        dr.rectangle([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=col)
    elif shape == "triangle":
        dr.polygon([(cx, cy - r), (cx + r, cy + r * 0.8), (cx - r, cy + r * 0.8)], fill=col)
    elif shape == "cross":
        w = r * 0.38
        dr.rectangle([cx - w, cy - r, cx + w, cy + r], fill=col)
        dr.rectangle([cx - r, cy - w, cx + r, cy + w], fill=col)


def render_tableau(tb, n_frames: int) -> np.ndarray:
    """Rich tableau (TableauAB) or any v3 / v3c tableau (delegated)."""
    if not isinstance(tb, TableauAB):
        return C3.render_tableau(tb, n_frames)
    sc, st = tb.scene, tb.style
    font = ImageFont.truetype(G.FONT if sc.split == "train" else G.FONT_EVAL, st.sign_font)
    lamp_x = {40: 330, 184: 380, 320: 160}[sc.door_x]
    out = []
    for f in range(n_frames):
        img = Image.new("RGB", (SIZE, SIZE))
        dr = G.draw_room(img, sc, tb.door)
        if not tb.light:
            img = Image.fromarray((np.asarray(img).astype(np.float32) * 0.62).astype(np.uint8))
            dr = ImageDraw.Draw(img)
        if tb.door:
            x0, y0, x1, y1 = sc.door_x + 22, 70, sc.door_x + 88, G.WALL_Y
            dr.rectangle([x0, y0, x1, y1 - 50], fill=(150, 210, 255))
            dr.rectangle([x0, y1 - 50, x1, y1], fill=(90, 190, 80))
        T.draw_lamp(dr, tb.light, lamp_x)
        if tb.sign is not None:
            x, y = G.sign_box(sc)
            dr.rectangle([x, y, x + 3.0 * st.sign_font, y + 1.5 * st.sign_font],
                         fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            dr.text((x + 1.5 * st.sign_font, y + 0.75 * st.sign_font), str(tb.sign),
                    fill=(0, 0, 0), font=font, anchor="mm")
        if tb.poster is not None:
            px, py, shape = tb.poster
            dr.rectangle([px, py, px + 40, py + 40], fill=(250, 250, 240), outline=(90, 60, 30), width=3)
            _draw_shape(dr, shape, px + 20, py + 20, 12)
        if tb.obj:
            T.draw_object(dr, tb.obj, tb.obj_x, st.obj_size)
        for i, u in enumerate(tb.units):
            if not u.person:
                continue
            h, w = C3.dims(u.k)
            bob = 2 * ((f + i) % 2)
            feet = tb.feet - bob
            hat = tb.hat[1] if tb.hat and tb.hat[0] == i else None
            G.draw_person(dr, u.x, feet, u.colour, hat, G.Style(fig_h=h, fig_w=w))
            if tb.shoes and tb.shoes[0] == i:
                lw = w * 0.32
                col = G.COLORS[tb.shoes[1]]
                for xl in (u.x - w * 0.38, u.x + w * 0.38 - lw):
                    dr.rectangle([xl - 3, feet - 7, xl + lw + 3, feet], fill=col, outline=(20, 20, 20))
        arr = np.array(img)
        for u in tb.units:
            if not u.person:
                paint_any_decoy(arr, u, tb.feet)
        out.append(arr)
    return np.stack(out)


def paint_any_decoy(arr, u, feet):
    """v3c decoys: 'stack' (in a person's slot), 'flat' (box lying in the front row) or 'wall'
    (coloured picture on the wall); the design is carried by the unit (u.design)."""
    d = getattr(u, "design", "stack")
    if d == "stack":
        C3.paint_decoy(arr, u, feet=feet)
    else:
        C3.paint_decoy(arr, u, design=d)


V3C_DESIGN = hasattr(C3, "FEET_FLAT")      # certified v3c = flat floor boxes + wall pictures


def sample_rich(rng: random.Random, scene) -> TableauAB:
    if V3C_DESIGN:
        return sample_rich_v3c(rng, scene)
    feet = rng.choice([372, 396])
    obj = rng.choice(OBJECTS) if rng.random() < 0.6 else None
    obj_slot = rng.randrange(7) if obj else None
    free = [s for s in range(8) if not obj or s not in (obj_slot, obj_slot + 1)]
    n_units = len(free)
    n = rng.randint(1, n_units)
    n_dec = rng.randint(0, n_units - n)
    slots = sorted(rng.sample(free, n + n_dec))
    roles = [True] * n + [False] * n_dec
    rng.shuffle(roles)
    k = rng.uniform(0.85, 1.1)
    pcols = rng.sample(CNAMES, n)
    units, pi = [], 0
    for s, isp in zip(slots, roles):
        if isp:
            units.append(C3.Unit(44 + 51 * s, k, pcols[pi], True))
            pi += 1
        else:
            units.append(C3.Unit(44 + 51 * s, k, rng.choice(CNAMES), False))
    pidx = [i for i, u in enumerate(units) if u.person]
    hat = (rng.choice(pidx), rng.choice(HAT_TRAIN)) if rng.random() < 0.6 else None
    shoes = (rng.choice(pidx), rng.choice(CNAMES)) if rng.random() < 0.6 else None
    sign = rng.randrange(10, 100) if rng.random() < 0.6 else None
    poster = None
    if rng.random() < 0.6:
        sx = G.sign_box(scene)[0]
        bad = [(scene.door_x - 50, scene.door_x + 100)]
        if sign is not None:
            bad.append((sx - 45, sx + 65))
        xs = [x for x in range(8, SIZE - 48, 4) if all(not (lo <= x <= hi) for lo, hi in bad)]
        poster = (rng.choice(xs), rng.randrange(120, 190), rng.choice(SHAPES))
    return TableauAB(scene, units, feet, rng.random() < 0.5, rng.random() < 0.5, obj,
                     44 + 51 * obj_slot + 25 if obj else 0, sign, hat, shoes, poster)


def sample_rich_v3c(rng: random.Random, scene) -> TableauAB:
    """Rich tableau with the CERTIFIED v3c decoys: persons stand on the back row (feet
    C3.FEET_FLAT, scale 0.75-0.92 as v3c), decoys are boxes lying in the front row ('flat',
    pixel counts of a person) and coloured pictures on the wall ('wall'); both carry a shirt
    colour, so neither colour mass nor 'coloured blobs' give count or colour."""
    k = rng.uniform(0.75, 0.92)
    obj = rng.choice(OBJECTS) if rng.random() < 0.6 else None
    obj_slot = rng.randrange(7) if obj else None
    free = [s for s in range(8) if not obj or s not in (obj_slot, obj_slot + 1)]
    n = rng.randint(1, len(free))
    slots = sorted(rng.sample(free, n))
    pcols = rng.sample(CNAMES, n)
    units = [C3.Unit(44 + 51 * s, k, c, True) for s, c in zip(slots, pcols)]
    ref = units[0].x
    n_flat = rng.randint(0, 4)
    for fx in sorted(rng.sample(C3.FRONT_X, n_flat)):
        u = C3.Unit(ref, k, rng.choice(CNAMES), False, ref, fx)
        u.design = "flat"
        units.append(u)
    n_wall = rng.randint(0, 3)
    wxs = C3.wall_xs(rng, scene, n_wall) if n_wall else []
    for wx in wxs:
        u = C3.Unit(ref, k, rng.choice(CNAMES), False, ref, wx)
        u.design = "wall"
        units.append(u)
    pidx = [i for i, u in enumerate(units) if u.person]
    hat = (rng.choice(pidx), rng.choice(HAT_TRAIN)) if rng.random() < 0.6 else None
    shoes = (rng.choice(pidx), rng.choice(CNAMES)) if rng.random() < 0.6 else None
    sign = rng.randrange(10, 100) if rng.random() < 0.6 else None
    poster = None
    if rng.random() < 0.6:
        sx = G.sign_box(scene)[0]
        bad = [(scene.door_x - 50, scene.door_x + 100)] + [(wx - C3.PW - 50, wx + C3.PW) for wx in wxs]
        if sign is not None:
            bad.append((sx - 45, sx + 65))
        xs = [x for x in range(8, SIZE - 48, 4) if all(not (lo <= x <= hi) for lo, hi in bad)]
        if xs:
            poster = (rng.choice(xs), rng.randrange(120, 170), rng.choice(SHAPES))
    return TableauAB(scene, units, C3.FEET_FLAT, rng.random() < 0.5, rng.random() < 0.5, obj,
                     44 + 51 * obj_slot + 25 if obj else 0, sign, hat, shoes, poster)


def sample_calque(rng: random.Random) -> TableauAB:
    """Distractor episode of the SAME nature (cartoon people, decoy stacks, object, hat, shoes,
    number board) but OUTDOORS: no room, wall, door or lamp. Never questioned. Shared by the 4
    variants of a group (only the target room differs), placed anywhere in the past."""
    sc = G.make_scene(rng, "train")
    tb = sample_rich(rng, sc)
    tb.door = tb.light = None
    tb.poster = None
    return tb


def render_calque(tb: TableauAB, filler: np.ndarray) -> np.ndarray:
    """Draw the calque's elements onto filler frames [F, 448, 448, 3] (same breathing as rooms)."""
    st = tb.style
    font = ImageFont.truetype(G.FONT, st.sign_font)
    out = []
    for f in range(filler.shape[0]):
        img = Image.fromarray(filler[f].copy())
        dr = ImageDraw.Draw(img)
        if tb.sign is not None:
            x, y = 300, 200
            dr.rectangle([x + 28, y + 30, x + 34, 300], fill=(110, 80, 50))
            dr.rectangle([x, y, x + 3.0 * st.sign_font, y + 1.5 * st.sign_font],
                         fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            dr.text((x + 1.5 * st.sign_font, y + 0.75 * st.sign_font), str(tb.sign),
                    fill=(0, 0, 0), font=font, anchor="mm")
        if tb.obj:
            T.draw_object(dr, tb.obj, tb.obj_x, st.obj_size)
        for i, u in enumerate(tb.units):
            if not u.person:
                continue
            h, w = C3.dims(u.k)
            feet = tb.feet - 2 * ((f + i) % 2)
            hat = tb.hat[1] if tb.hat and tb.hat[0] == i else None
            G.draw_person(dr, u.x, feet, u.colour, hat, G.Style(fig_h=h, fig_w=w))
            if tb.shoes and tb.shoes[0] == i:
                lw = w * 0.32
                col = G.COLORS[tb.shoes[1]]
                for xl in (u.x - w * 0.38, u.x + w * 0.38 - lw):
                    dr.rectangle([xl - 3, feet - 7, xl + lw + 3, feet], fill=col, outline=(20, 20, 20))
        arr = np.array(img)
        for u in tb.units:
            if not u.person:
                C3.paint_decoy(arr, u, feet=tb.feet)
        out.append(arr)
    return np.stack(out)


def facts(tb: TableauAB) -> dict:
    ps = [u for u in tb.units if u.person]
    xs = [u.x for u in ps]
    return {"n": len(ps), "colours": sorted(u.colour for u in ps),
            "decoy_colours": sorted(u.colour for u in tb.units if not u.person),
            "n_decoys": sum(not u.person for u in tb.units),
            "door": tb.door, "light": tb.light, "obj": tb.obj, "sign": tb.sign,
            "hat": tb.hat[1] if tb.hat else None, "shoes": tb.shoes[1] if tb.shoes else None,
            "poster": tb.poster[2] if tb.poster else None,
            "side": "left" if max(xs) < SIZE / 2 else ("right" if min(xs) > SIZE / 2 else "mixed")}


def make_group_ab(g: int, split: str = "train"):
    """4 variants, same room, same filler seed; contents independent."""
    assert split == "train", "rich scenes are a training distribution"
    key = f"ab-{split}-{g}"
    rng = random.Random(key)
    scene = G.make_scene(rng, split)
    tbs = [sample_rich(random.Random(f"{key}-{v}"), scene) for v in range(4)]
    return [{"group": g, "variant": v, "split": split, "tableau": tb, "facts": facts(tb),
             "filler_seed": key} for v, tb in enumerate(tbs)]


# ------------------------------------------------------------------ questions
INTROS = {
    "train": ["The video stream shows outdoor scenery and, at one point, a room with cartoon people. ",
              "Earlier in this stream there was a short scene inside a room with cartoon figures. ",
              "At some point the video showed a room with people drawn as cartoons. "],
    "eval": [T.INTRO],
}
BOXES = " Boxes and pictures are not people."
TRAIN_Q = {
    "count": ["How many people were in the room?{b} Answer with a number.",
              "Count the people standing in the room.{b} Reply with just the number.",
              "What was the number of people in the room?{b} Give only a number."],
    "colour": ["One of these colours, {opts}, was the shirt colour of a person in the room; the others "
               "were not worn by anyone (they may appear on boxes or pictures). Which colour was it? "
               "Answer with the colour.",
               "Which of {opts} was a person's shirt colour in the room? Reply with one colour word."],
    "big_object": ["What large object stood on the floor of the room? Answer with one word.",
                   "Name the big object on the floor of the room, in one word."],
    "door": ["Was the door of the room open or closed? Answer with one word.",
             "In the room, was the door open or closed? Reply open or closed."],
    "light": ["Was the ceiling light in the room on or off? Answer with one word.",
              "Was the lamp of the room switched on or off? Reply on or off."],
    "room_state": ["State the door and the light of the room. Answer exactly in the form 'door open, light off' "
                   "(door open or closed, light on or off).",
                   "Answer in the form 'door closed, light on': was the door of the room open or closed, and "
                   "was its light on or off?"],
    "sign_number": ["What number was written on the small sign on the wall? Answer with the number.",
                    "A small sign hung on the wall. Which number did it show? Reply with the number only."],
    "hat_color": ["One person wore a hat. What colour was the hat? Answer with the colour.",
                  "What was the colour of the hat that one of the people wore? One word."],
    "shoe_color": ["One person wore coloured shoes. What colour were the shoes? Answer with the colour.",
                   "Which colour were the shoes of the person with coloured shoes? One word."],
    "poster_shape": ["A small poster hung on the wall. What shape was drawn on it? Answer with one word.",
                     "Which shape was drawn on the small poster on the wall? Reply with the shape's name only."],
}
EVAL_Q = {   # held-out paraphrases (free-form eval of the bench items)
    "count": "In the room scene, how many people were standing there?{b} Reply with a number only.",
    "colour": "Some of the colours {opts} were only on boxes. Which one did a person wear as a shirt? "
              "Reply with the colour only.",
    "big_object": "Which big object could be seen on the floor in the room? Reply with one word.",
    "room_state": "Describe the room in the form 'door closed, light off': was the door open or closed, and "
                  "was the ceiling lamp on or off?",
    "side": "Were the people in the room grouped on the left or on the right? Reply left or right.",
    "sign_number": "Which number was written on the small sign on the room's wall? Reply with the number only.",
    "hat_color": "Someone in the room had a hat on. Which colour was that hat? Reply with the colour only.",
}
FAM_OF = {"door": "room_state", "light": "room_state"}


def kind_of(fam):
    fam = FAM_OF.get(fam, fam)
    return "gist" if fam in GIST_FAMS or fam == "side" else "detail"


def room_answer(f):
    return f"door {'open' if f['door'] else 'closed'}, light {'on' if f['light'] else 'off'}"


def answer(q: dict, f: dict):
    """The true free answer of question ``q`` for a scene with facts ``f`` (None: not applicable)."""
    fam = q["qfam"]
    if fam == "count":
        return str(f["n"])
    if fam == "colour":
        hits = [c for c in q["opts"] if c in f["colours"]]
        return hits[0] if len(hits) == 1 else None
    if fam == "big_object":
        return f["obj"]
    if fam == "door":
        return "open" if f["door"] else "closed"
    if fam == "light":
        return "on" if f["light"] else "off"
    if fam == "room_state":
        return room_answer(f)
    if fam == "sign_number":
        return None if f["sign"] is None else str(f["sign"])
    if fam == "hat_color":
        return f["hat"]
    if fam == "shoe_color":
        return f["shoes"]
    if fam == "poster_shape":
        return f["poster"]
    if fam == "side":
        return None if f.get("side") not in ("left", "right") else f["side"]
    raise ValueError(fam)


def questions_for(it, rng: random.Random):
    """All applicable training questions of one variant (one paraphrase drawn per family)."""
    f = it["facts"]
    intro = rng.choice(INTROS["train"])
    out = []
    for qf, temps in TRAIN_Q.items():
        q = {"qfam": qf, "family": FAM_OF.get(qf, qf), "kind": kind_of(qf)}
        if qf == "colour":
            non = [c for c in CNAMES if c not in f["colours"]]
            if len(non) < 3:
                continue
            gold = rng.choice(f["colours"])
            dec = [c for c in non if c in f["decoy_colours"]]
            rest = [c for c in non if c not in dec]
            rng.shuffle(dec)
            rng.shuffle(rest)
            opts = [gold] + (dec + rest)[:3]
            rng.shuffle(opts)
            q["opts"] = opts
        a = answer(q, f)
        if a is None:
            continue
        b = BOXES if f["n_decoys"] else ""
        text = rng.choice(temps).format(b=b, opts=", ".join(q.get("opts", [])[:-1]) + " or " + q["opts"][-1]
                                        if "opts" in q else "")
        q.update(prompt=intro + text, answer=a, intro=intro)
        out.append(q)
    return out


def eval_free_question(item: dict):
    """Free-form version of a bench item (v3 / v3c generator): prompt, gold string, option strings."""
    fam, f, opts = item["family"], item["facts"], item["options"]
    if fam == "count":
        b = BOXES if f.get("decoy_colours") else ""
        golds = [str(x) for x in opts]
        text = EVAL_Q["count"].format(b=b)
    elif fam == "colour":
        golds = list(opts)
        text = EVAL_Q["colour"].format(opts=", ".join(opts[:-1]) + " or " + opts[-1])
    elif fam == "big_object":
        golds = [o.replace("a ", "") for o in opts]
        text = EVAL_Q["big_object"]
    elif fam == "room_state":
        golds = list(opts)
        text = EVAL_Q["room_state"]
    elif fam == "side":
        golds = [o.replace("on the ", "") for o in opts]
        text = EVAL_Q["side"]
    else:
        golds = list(opts)
        text = EVAL_Q[fam]
    return {"prompt": T.INTRO + text, "golds": golds, "answer": golds[item["answer"]]}


NUMW = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
SYN = {"grey": "gray", "violet": "purple", "magenta": "pink", "cube": "box", "square box": "box",
       "pyramid": "cone", "ellipse": "circle", "plus": "cross", "x": "cross"}


def norm_answer(s: str) -> str:
    """Lenient normal form: case, punctuation, articles, number words, a few synonyms,
    'the door was open' -> 'open' style trailing words for one-word answers."""
    s = s.strip().lower().replace("\n", " ")
    s = "".join(ch for ch in s if ch.isalnum() or ch in " ,").strip()
    for p in ("a ", "an ", "the "):
        if s.startswith(p):
            s = s[len(p):]
    s = " ".join(NUMW.get(w, w) for w in s.split())
    return SYN.get(s, s)


def free_match(pred: str, gold: str) -> bool:
    p, g = norm_answer(pred), norm_answer(gold)
    if p == g:
        return True
    if " " not in g and "," not in g:            # one-word gold: accept it as the only content word
        toks = [t for t in p.replace(",", " ").split() if t not in ("is", "was", "were", "it", "door", "light",
                                                                     "lamp", "the", "shirt", "hat", "shoes", "color", "colour")]
        return toks == [g]
    return p.replace(" ", "") == g.replace(" ", "")


MCQ_Q = {
    "count": "How many people were in the room?{b}",
    "colour": "Which of these colours was the shirt colour of a person in the room?",
    "big_object": "What large object stood on the floor of the room?",
    "door": "Was the door of the room open or closed?",
    "light": "Was the ceiling light of the room on or off?",
    "room_state": "What was the state of the door and of the light in the room?",
    "sign_number": "What number was written on the small sign on the wall?",
    "hat_color": "One person wore a hat. What colour was the hat?",
    "shoe_color": "One person wore coloured shoes. What colour were the shoes?",
    "poster_shape": "What shape was drawn on the small poster on the wall?",
}


def mcq_for(q: dict, f: dict, rng: random.Random, intro: str):
    """Lettered version of a free question (bench style): (prompt, options, gold index) or None."""
    fam, a = q["qfam"], q["answer"]
    if fam == "count":
        opts = [str(x) for x in ([1, 3, 5, 7] if int(a) % 2 else [2, 4, 6, 8])]
    elif fam == "colour":
        opts = list(q["opts"])
    elif fam == "big_object":
        opts = list(OBJECTS)
    elif fam == "door":
        opts = ["open", "closed"]
    elif fam == "light":
        opts = ["on", "off"]
    elif fam == "room_state":
        opts = list(T.ROOM_OPTS)
    elif fam == "sign_number":
        opts = sorted([int(a)] + rng.sample([x for x in range(10, 100) if x != int(a)], 3))
        opts = [str(x) for x in opts]
    elif fam in ("hat_color", "shoe_color"):
        opts = [a] + rng.sample([c for c in CNAMES if c != a], 3)
        rng.shuffle(opts)
    elif fam == "poster_shape":
        opts = list(SHAPES)
    else:
        return None
    if a not in opts:
        return None
    b = BOXES if f.get("n_decoys") else ""
    show = [f"a {o}" if fam == "big_object" else o for o in opts]
    prompt = "\n".join([intro + MCQ_Q[fam].format(b=b)] + [f"{LET[i]}. {o}" for i, o in enumerate(show)] + [T.TAIL])
    return prompt, opts, opts.index(a)


# ------------------------------------------------------------------ contact sheet
def contact(groups, path, thumb=224):
    font = ImageFont.truetype(G.FONT.replace("-Bold", ""), 11)
    items = [it for grp in groups for it in grp]
    rows = (len(items) + 3) // 4
    pad = 70
    sheet = Image.new("RGB", (4 * thumb, rows * (thumb + pad)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        fr = render_tableau(it["tableau"], 1)[0]
        x0, y0 = (i % 4) * thumb, (i // 4) * (thumb + pad)
        f = it["facts"]
        qs = questions_for(it, random.Random(f"q-{it['group']}-{it['variant']}"))
        lines = [f"g{it['group']} v{it['variant']} n={f['n']} dec={f['n_decoys']} obj={f['obj']}",
                 f"door={f['door']} light={f['light']} sign={f['sign']} hat={f['hat']}",
                 f"shoes={f['shoes']} poster={f['poster']} side={f['side']}",
                 " ".join(f"{q['qfam'][:5]}={q['answer']}" for q in qs)[:60]]
        for j, l in enumerate(lines):
            d.text((x0 + 2, y0 + 2 + 13 * j), l, fill=(0, 0, 0), font=font)
        sheet.paste(Image.fromarray(fr).resize((thumb, thumb), Image.BILINEAR), (x0, y0 + pad))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", default="contact_ab.png")
    ap.add_argument("--groups", default="0,1,2")
    ap.add_argument("--stats", type=int, default=0, help="question stats over N groups")
    a = ap.parse_args()
    groups = [make_group_ab(int(g)) for g in a.groups.split(",")]
    contact(groups, a.contact)
    print("saved", a.contact, "gen", GEN_MD5)
    for it in groups[0][:1]:
        for q in questions_for(it, random.Random(0)):
            print(f"  [{q['kind']}/{q['qfam']}] {q['prompt']}  ->  {q['answer']}")
    if a.stats:
        from collections import Counter
        import time
        c, t0 = Counter(), time.time()
        for g in range(a.stats):
            for it in make_group_ab(g):
                for q in questions_for(it, random.Random(f"{g}-{it['variant']}")):
                    c[q["qfam"]] += 1
        print({k: round(v / (4 * a.stats), 2) for k, v in c.items()}, f"{time.time() - t0:.1f}s")
        t0 = time.time()
        render_tableau(groups[0][0]["tableau"], 12)
        print(f"render 12 frames {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
