"""v3 "snapshot gist" generator: a held TABLEAU embedded in a procedural filler stream.

Why v3 (lead + audit, 22/10): Qwen2.5-VL-3B reads single-frame facts well (init count
88 % from the first 4 s alone, sign 100 % with shuffled frames) but does not integrate
events across a clip (v2 with leaks removed: first_left 27, order = blind, n_left 40).
Questions that need the HOST to reason over time test the host, not the memory. So
every v3 gist fact is readable from ANY single frame of the tableau.

Stream (constant length, every item):  [PRE filler][TABLEAU E s][POST filler][WINDOW filler]
  * filler = outdoor scenery (sky, hills, trees, drifting clouds; no people, no door,
    no sign, no digits), seeded per GROUP: identical across the 4 variants.
  * the tableau is a room held still (agents breathe +-2 px) for E seconds.
  * the window (last WIN s) is filler shared by the group, so window-only and blind are
    chance per item and 0 per group BY CONSTRUCTION; no end-state leak exists because
    the episode has no "end state" in the window at all.

Families (counterfactual groups of 4; same text, options, decor, filler; each option
correct once, binary families twice each):
  gist   count        how many people (4 values from 1..6)
         colour       which shirt colour was present (exactly one option colour present)
         room_state   door open/closed x light on/off (4-way)
         big_object   the one large object on the floor: ball / box / cone / star
         side         group stands on the left or right (binary; EVAL-ONLY family)
  detail hat_color    one small hat on one person
         sign_number  a small number sign on the wall
Splits: seed salt, held-out decor, held-out hat colours, small-sign font/position, and
paraphrased templates in eval (as gen_scenes v2).
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gen_scenes as G

FPS = 2
SIZE = 448
PRE, E, POST, WIN = 4, 6, 8, 8          # seconds
CNAMES = G.CNAMES
COLORS = G.COLORS
LET = "ABCD"
OBJECTS = ["ball", "box", "cone", "star"]
OBJ_COL = (90, 140, 200)                # one neutral colour for every object


@dataclass
class TStyle:
    fig_h: int = 120
    fig_w: int = 38
    hat_scale: float = 1.0
    sign_font: int = 20
    obj_size: int = 80
    count_gap: bool = False   # count options {1,3,5,7} or {2,4,6,8} (no adjacent values)
    count_obj: bool = True    # big object present in count tableaux
    door_bright: bool = False # room v2: open door shows daylight (not dimmed by the lamp),
                              # and the lamp hangs away from the door (no overlap)


@dataclass
class Tableau:
    scene: G.Scene
    agents: list            # [(colour, x, hat)]
    door: bool
    light: bool
    obj: str | None
    obj_x: int
    sign: int | None
    style: TStyle = field(default_factory=TStyle)


# ------------------------------------------------------------------ drawing
def draw_lamp(dr, light: bool, cx: int = SIZE // 2 + 10):
    dr.line([cx, 0, cx, 40], fill=(40, 40, 40), width=3)
    dr.polygon([(cx - 30, 70), (cx + 30, 70), (cx + 16, 40), (cx - 16, 40)], fill=(60, 60, 70))
    if light:
        dr.ellipse([cx - 14, 62, cx + 14, 90], fill=(255, 240, 120), outline=(200, 170, 0), width=2)
        for k in range(8):
            a = math.pi * (0.1 + 0.8 * k / 7)
            dr.line([cx + 22 * math.cos(a), 78 + 22 * math.sin(a),
                     cx + 40 * math.cos(a), 78 + 40 * math.sin(a)], fill=(255, 230, 80), width=3)
    else:
        dr.ellipse([cx - 14, 62, cx + 14, 90], fill=(110, 110, 110), outline=(60, 60, 60), width=2)


def draw_object(dr, kind, x, s):
    y1 = G.BACK_FEET + 30
    y0 = y1 - s
    c, o = OBJ_COL, (20, 30, 60)
    if kind == "ball":
        dr.ellipse([x - s / 2, y0, x + s / 2, y1], fill=c, outline=o, width=3)
    elif kind == "box":
        dr.rectangle([x - s / 2, y0, x + s / 2, y1], fill=c, outline=o, width=3)
    elif kind == "cone":
        dr.polygon([(x, y0), (x + s / 2, y1), (x - s / 2, y1)], fill=c, outline=o)
    elif kind == "star":
        pts = []
        for k in range(10):
            r = s / 2 if k % 2 == 0 else s / 5
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((x + r * math.cos(a), (y0 + y1) / 2 + r * math.sin(a)))
        dr.polygon(pts, fill=c, outline=o)


def render_tableau(tb: Tableau, n_frames: int) -> np.ndarray:
    st, sc = tb.style, tb.scene
    gst = G.Style(fig_h=st.fig_h, fig_w=st.fig_w, hat_scale=st.hat_scale)
    font = ImageFont.truetype(G.FONT if sc.split == "train" else G.FONT_EVAL, st.sign_font)
    out = []
    for f in range(n_frames):
        img = Image.new("RGB", (SIZE, SIZE))
        dr = G.draw_room(img, sc, tb.door)
        if not tb.light:   # darker room when the light is off
            arr = (np.asarray(img).astype(np.float32) * 0.62).astype(np.uint8)
            img = Image.fromarray(arr)
            dr = ImageDraw.Draw(img)
        if tb.door and st.door_bright:   # daylight through the open door
            x0, y0, x1, y1 = sc.door_x + 22, 70, sc.door_x + 88, G.WALL_Y
            dr.rectangle([x0, y0, x1, y1 - 50], fill=(150, 210, 255))
            dr.rectangle([x0, y1 - 50, x1, y1], fill=(90, 190, 80))
        if st.door_bright:
            draw_lamp(dr, tb.light, {40: 330, 184: 380, 320: 160}[sc.door_x])
        else:
            draw_lamp(dr, tb.light)
        if tb.sign is not None:
            x, y = G.sign_box(sc)
            dr.rectangle([x, y, x + 3.0 * st.sign_font, y + 1.5 * st.sign_font],
                         fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            dr.text((x + 1.5 * st.sign_font, y + 0.75 * st.sign_font), str(tb.sign),
                    fill=(0, 0, 0), font=font, anchor="mm")
        if tb.obj:
            draw_object(dr, tb.obj, tb.obj_x, st.obj_size)
        for i, (c, x, hat) in enumerate(tb.agents):
            bob = 2 * ((f + i) % 2)            # breathing, so it is a video
            G.draw_person(dr, x, G.BACK_FEET - bob, c, hat, gst)
        out.append(np.asarray(img))
    return np.stack(out)


def render_filler(seed: str, n_frames: int, t0: int = 0) -> np.ndarray:
    """Outdoor scenery, slowly panning; no people, door, sign or digits."""
    r = random.Random("filler-" + seed)
    sky = (r.randint(90, 150), r.randint(150, 200), r.randint(210, 250))
    grass = (r.randint(40, 90), r.randint(120, 170), r.randint(40, 80))
    hills = [(r.uniform(0, 900), r.uniform(60, 130), r.uniform(150, 260)) for _ in range(4)]
    trees = [(r.uniform(0, 900), r.uniform(300, 380), r.uniform(18, 34)) for _ in range(7)]
    clouds = [(r.uniform(0, 900), r.uniform(30, 120), r.uniform(25, 45)) for _ in range(4)]
    out = []
    for f in range(t0, t0 + n_frames):
        img = Image.new("RGB", (SIZE, SIZE), sky)
        dr = ImageDraw.Draw(img)
        pan = 4 * f
        for (hx, hh, hw) in hills:
            x = (hx - pan * 0.5) % 900 - 200
            dr.ellipse([x - hw, 280 - hh, x + hw, 280 + hh], fill=(70, 120, 80))
        dr.rectangle([0, 280, SIZE, SIZE], fill=grass)
        for (cx, cy, cr) in clouds:
            x = (cx - pan * 0.3) % 900 - 200
            for dx in (-cr, 0, cr):
                dr.ellipse([x + dx - cr, cy - cr * 0.6, x + dx + cr, cy + cr * 0.6], fill=(245, 245, 250))
        for (tx, ty, tr) in trees:
            x = (tx - pan) % 900 - 200
            dr.rectangle([x - 5, ty - 10, x + 5, ty + 30], fill=(100, 70, 40))
            dr.ellipse([x - tr, ty - 2 * tr, x + tr, ty], fill=(30, 110, 40))
        out.append(np.asarray(img))
    return np.stack(out)


# ------------------------------------------------------------------ sampling
def place(rng, n, side=None, avoid=()):
    """n agent x positions on the slot grid (side: 'left'/'right' -> that half only)."""
    slots = [s for s in range(8) if s not in avoid]
    if side == "left":
        slots = [s for s in slots if s < 4]
    elif side == "right":
        slots = [s for s in slots if s >= 4]
    return [44 + 51 * s for s in sorted(rng.sample(slots, n))]


def sample_tableau(rng, scene, style, *, n=None, colours=None, door=None, light=None,
                   obj="rand", sign=None, hat=None, side=None, pool=None):
    if n is None:
        n = rng.randint(1, 4)
    pool = list(pool or CNAMES)
    cols = list(colours or [])
    rest = [c for c in pool if c not in cols]
    rng.shuffle(rest)
    cols = (cols + rest)[:n]
    rng.shuffle(cols)
    if obj == "rand":
        obj = rng.choice(OBJECTS + [None, None])
    obj_slot = rng.randrange(7) if obj else None      # 7: the object never clips the edge
    avoid = {obj_slot, (obj_slot or 0) + 1} if obj else set()
    if side:
        avoid = set()
        if obj:   # object on the other half
            obj_slot = rng.randrange(4, 7) if side == "left" else rng.randrange(0, 3)
    xs = place(rng, n, side, avoid if not side else ())
    agents = [(c, x, None) for c, x in zip(cols, xs)]
    if hat is not None:
        i = rng.randrange(n)
        agents[i] = (agents[i][0], agents[i][1], hat)
    return Tableau(scene, agents,
                   rng.random() < 0.5 if door is None else door,
                   rng.random() < 0.5 if light is None else light,
                   obj, 44 + 51 * obj_slot + 25 if obj else 0, sign, style)


def tfacts(tb: Tableau) -> dict:
    xs = [x for _, x, _ in tb.agents]
    return {"n": len(tb.agents), "colours": sorted(c for c, _, _ in tb.agents),
            "door": tb.door, "light": tb.light, "obj": tb.obj, "sign": tb.sign,
            "hat": next((h for _, _, h in tb.agents if h), None),
            "side": "left" if max(xs) < SIZE / 2 else ("right" if min(xs) > SIZE / 2 else "mixed")}


# ------------------------------------------------------------------ questions
FAMILIES = ["count", "colour", "room_state", "big_object", "side", "hat_color", "sign_number"]
GIST = ["count", "colour", "room_state", "big_object", "side"]
DETAIL = ["hat_color", "sign_number"]
EVAL_ONLY = ["side"]
ROOM_OPTS = ["door open, light on", "door open, light off",
             "door closed, light on", "door closed, light off"]
Q = {
    "count": ("How many people were in the room?",
              "In the room scene, how many people were standing there?"),
    "colour": ("Which of these shirt colours did one of the people in the room wear?",
               "One person in the room wore a shirt of one of these colours. Which one?"),
    "room_state": ("In the room, was the door open or closed, and was the ceiling light on or off?",
                   "Describe the room: the state of the door and of the ceiling lamp."),
    "big_object": ("What large object was standing on the floor of the room?",
                   "Which big object could be seen on the floor in the room?"),
    "side": ("On which side of the room were the people standing?",
             "Were the people in the room grouped on the left or on the right?"),
    "hat_color": ("One person in the room wore a hat. What colour was the hat?",
                  "Someone in the room had a hat on. Which colour was that hat?"),
    "sign_number": ("A small sign with a number hung on the wall of the room. What number was on it?",
                    "Which number was written on the small sign on the room's wall?"),
}
INTRO = ("The video stream shows outdoor scenery and, at one point, a room with cartoon "
         "people, each wearing a coloured shirt. ")
TAIL = "Answer with the option's letter from the given choices directly."


def answer_from_facts(fam, f, opts):
    if fam == "count":
        return opts.index(str(f["n"]))
    if fam == "colour":
        hits = [i for i, o in enumerate(opts) if o in f["colours"]]
        assert len(hits) == 1, (opts, f)
        return hits[0]
    if fam == "room_state":
        return opts.index(f"door {'open' if f['door'] else 'closed'}, light {'on' if f['light'] else 'off'}")
    if fam == "big_object":
        return opts.index(f"a {f['obj']}")
    if fam == "side":
        return opts.index(f"on the {f['side']}")
    if fam == "hat_color":
        return opts.index(f["hat"])
    if fam == "sign_number":
        return opts.index(str(f["sign"]))
    raise ValueError(fam)


def make_group(fam, gseed, style=None, split="eval", qset=None):
    assert split in G.SPLITS
    if split == "train" and fam in EVAL_ONLY:
        raise ValueError(f"{fam} is eval-only")
    style = style or TStyle()
    qset = qset or (1 if split == "train" else 2)
    key = f"v3-{split}-{fam}-{gseed}"
    rng = random.Random(key)
    scene = G.make_scene(rng, split)
    vr = lambda v: random.Random(f"{key}-{v}")      # noqa: E731
    if fam == "count":
        if style.count_gap:
            opts = [str(x) for x in rng.choice([[1, 3, 5, 7], [2, 4, 6, 8]])]
        else:
            opts = [str(x) for x in sorted(rng.sample(range(1, 7), 4))]
        kw = {} if style.count_obj and not style.count_gap else {"obj": None}
        tbs = [sample_tableau(vr(v), scene, style, n=int(o), **kw) for v, o in enumerate(opts)]
    elif fam == "colour":
        cs = rng.sample(CNAMES, 4)
        opts = list(cs)
        pool = [c for c in CNAMES if c not in cs]
        tbs = []
        for v, c in enumerate(cs):
            r = vr(v)
            n = r.randint(1, 4)
            tbs.append(sample_tableau(r, scene, style, n=n, colours=[c], pool=[c] + pool))
    elif fam == "room_state":
        opts = list(ROOM_OPTS)
        tbs = [sample_tableau(vr(v), scene, style, door=v < 2, light=v % 2 == 0) for v in range(4)]
    elif fam == "big_object":
        opts = [f"a {o}" for o in OBJECTS]
        tbs = [sample_tableau(vr(v), scene, style, obj=o) for v, o in enumerate(OBJECTS)]
    elif fam == "side":
        opts = ["on the left", "on the right"]
        tbs = []
        for v in range(4):
            r = vr(v)
            tbs.append(sample_tableau(r, scene, style, n=r.randint(2, 4), side=opts[v % 2][7:]))
    elif fam == "hat_color":
        hp = CNAMES if split == "eval" else [c for c in CNAMES if c not in G.HAT_HELDOUT]
        hats = rng.sample(hp, 4)
        opts = list(hats)
        pool = [c for c in CNAMES if c not in hats]
        tbs = [sample_tableau(vr(v), scene, style, hat=h, pool=pool) for v, h in enumerate(hats)]
    elif fam == "sign_number":
        nums = sorted(rng.sample(range(10, 100), 4))
        opts = [str(x) for x in nums]
        tbs = [sample_tableau(vr(v), scene, style, sign=n) for v, n in enumerate(nums)]
    else:
        raise ValueError(fam)
    prompt = "\n".join([INTRO + Q[fam][qset - 1]] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL])
    out = []
    for v, tb in enumerate(tbs):
        f = tfacts(tb)
        ans = answer_from_facts(fam, f, opts)
        gold = v % 2 if len(opts) == 2 else v
        assert ans == gold, (fam, gseed, v, ans, gold, f)
        out.append({"family": fam, "group": gseed, "variant": v, "split": split, "tableau": tb,
                    "filler_seed": key, "prompt": prompt, "options": opts, "answer": ans, "facts": f})
    return out


def stream(it) -> dict:
    """Frames of the stream pieces for one item (filler shared by the group)."""
    fs = it["filler_seed"]
    pre = render_filler(fs, PRE * FPS, 0)
    tab = render_tableau(it["tableau"], E * FPS)
    post = render_filler(fs, POST * FPS, (PRE + E) * FPS)
    win = render_filler(fs, WIN * FPS, (PRE + E + POST) * FPS)
    return {"pre": pre, "tab": tab, "post": post, "win": win,
            "full": np.concatenate([pre, tab, post, win])}


def contact(items, path, every=2, thumb=200, ncol=13):
    rows = []
    labels = []
    for it in items:
        s = stream(it)["full"][::every]
        rows.append(s)
        f = it["facts"]
        labels.append(f"{it['family']} v{it['variant']}: ans={it['options'][it['answer']]} | n={f['n']} "
                      f"cols={f['colours']} door={f['door']} light={f['light']} obj={f['obj']} "
                      f"hat={f['hat']} sign={f['sign']} side={f['side']}")
    pad = 24
    nl = [(len(r) + ncol - 1) // ncol for r in rows]
    sheet = Image.new("RGB", (ncol * thumb, sum(pad + n * thumb for n in nl)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(G.FONT.replace("-Bold", ""), 14)
    y = 0
    for r, lab, n in zip(rows, labels, nl):
        d.text((4, y + 4), lab, fill=(0, 0, 0), font=font)
        y += pad
        for i, f in enumerate(r):
            x0, y0 = (i % ncol) * thumb, y + (i // ncol) * thumb
            sheet.paste(Image.fromarray(f).resize((thumb, thumb), Image.BILINEAR), (x0, y0))
            d.text((x0 + 3, y0 + 2), f"{i * every / FPS:.0f}s", fill=(255, 255, 0), font=font)
        y += n * thumb
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", default="contact_v3.png")
    ap.add_argument("--families", default="count,room_state,sign_number")
    ap.add_argument("--variants", default="0,1")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    items = []
    for fam in a.families.split(","):
        g = make_group(fam, a.group)
        items += [g[int(v)] for v in a.variants.split(",")]
    contact(items, a.contact, a.every)
    print("saved", a.contact)


if __name__ == "__main__":
    main()
