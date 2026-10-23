"""v3c: `count` and `colour` rebuilt so that COLOUR MASS carries no answer (23/10).

Problem (ladder/bayes_mass.py on v3b): a logistic regression on the per-colour pixel mass of
the tableau segment decodes count at 99.9 % per question (99.5 % per group) and colour at
95.5 / 84.5. Cause: in v3b every person has the same size and parts, so the foreground mass
is proportional to n; in colour the asked colour is the only option colour present, so its
mass is > 0 in exactly one variant.

v3c fix: "units". Every unit sits on a back-row slot and is EITHER a person OR a decoy
(a stack: dark plank, trouser-coloured crate, shirt-coloured box, skin-coloured lid). A decoy
is built row by row so that each of its colours has EXACTLY the pixel count of the
corresponding part of the person it replaces (shirt fill, skin, legs, black outline). So
swapping a person for a decoy leaves the colour histogram of the frame unchanged, to the
pixel.
  count  : 8 units per tableau (sizes and shirt colours per slot shared by the group);
           variant v makes n_v of them persons, the rest decoys. Options {1,3,5,7} or
           {2,4,6,8} as v3b. Door, light shared by the group, no big object.
  colour : n persons (1..4) + 3 decoys, layout and sizes shared by the group. The 4 option
           colours are ALL present in every variant: one on a person's shirt (the answer),
           three on decoy boxes. Other persons wear non-option colours (shared). No object.
=> the 4 variants of a group have pixel-identical colour histograms (persons bob +-2 px,
   which moves pixels but keeps counts). Any decoder of colour mass scores 0 per group.

Other families are delegated untouched to the frozen v3 generator (gen_tableau).
    python gen_tableau_v3c.py --contact contact_v3c.png
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gen_tableau as T
from gen_tableau import (FPS, SIZE, PRE, E, POST, WIN, CNAMES, COLORS, LET, TStyle, INTRO, TAIL,  # noqa: F401
                         render_filler, FAMILIES, GIST, DETAIL, EVAL_ONLY)

G = T.G
OUTLINE = (20, 20, 20)
FEET = 396        # whole unit (person or decoy) on the uniform floor, below the wall line
V3C = ("count", "colour")
Q3C = {
    "count": ("How many people were in the room? (Stacks of boxes are not people.)",
              "In the room scene, how many people were standing there? Do not count the stacks of boxes."),
    "colour": ("One person in the room wore a shirt of one of these colours; the other colours were "
               "only on stacks of boxes. Which colour was a person's shirt?",
               "Some colours below were only on boxes. Which of these colours did one of the people in "
               "the room wear as a shirt?"),
}


@dataclass
class Unit:
    x: int
    k: float              # size scale
    colour: str
    person: bool
    ref_x: int | None = None   # decoy box takes the shirt pixel count of a person at ref_x


@dataclass
class TableauC:
    scene: G.Scene
    units: list
    door: bool
    light: bool
    style: TStyle = field(default_factory=TStyle)


def dims(k):
    return int(round(120 * k)), int(round(38 * k))


@lru_cache(maxsize=None)
def part_counts(h, w, x, feet):
    """Exact pixel counts of a person's parts (shirt fill, skin, legs, outline), drawn at the
    SAME (x, feet) as in the tableau: PIL rounding of float coordinates is not invariant to
    integer shifts, so the counts are taken in place. Also checked at feet-2 (breathing)."""
    res = []
    for f in (feet, feet - 2):
        img = Image.new("RGB", (SIZE, SIZE), (1, 2, 3))
        G.draw_person(ImageDraw.Draw(img), x, f, "red", None, G.Style(fig_h=h, fig_w=w))
        a = np.asarray(img).reshape(-1, 3)
        cnt = lambda c: int((a == np.array(c)).all(1).sum())  # noqa: E731
        res.append({"legs": cnt(G.LEGS), "shirt": cnt(COLORS["red"]), "skin": cnt(G.SKIN),
                    "out": cnt(OUTLINE)})
    assert res[0] == res[1], (h, w, x, res)
    return res[0]


def paint_block(arr, cx, bottom, width, count, col):
    """Paint exactly `count` pixels: full rows of `width` from `bottom` upwards, plus one
    centred partial row. Returns the new top (first free row above)."""
    full, rem = divmod(count, width)
    x0 = cx - width // 2
    y = bottom
    for _ in range(full):
        arr[y, x0:x0 + width] = col
        y -= 1
    if rem:
        r0 = cx - rem // 2
        arr[y, r0:r0 + rem] = col
        y -= 1
    return y


def paint_decoy(arr, u: Unit, feet=FEET):
    h, w = dims(u.k)
    pc = part_counts(h, w, u.x, feet)
    wide = w + 6
    y = paint_block(arr, u.x, feet, wide, pc["out"], OUTLINE)          # dark plank
    y = paint_block(arr, u.x, y, w, pc["legs"], G.LEGS)                 # crate
    shirt = pc["shirt"] if u.ref_x is None else part_counts(h, w, u.ref_x, feet)["shirt"]
    y = paint_block(arr, u.x, y, w, shirt, COLORS[u.colour])            # coloured box
    paint_block(arr, u.x, y, wide, pc["skin"], G.SKIN)                  # lid


def render_tableau(tb, n_frames: int) -> np.ndarray:
    if not isinstance(tb, TableauC):
        return T.render_tableau(tb, n_frames)
    sc = tb.scene
    out = []
    for f in range(n_frames):
        img = Image.new("RGB", (SIZE, SIZE))
        dr = G.draw_room(img, sc, tb.door)
        if not tb.light:
            img = Image.fromarray((np.asarray(img).astype(np.float32) * 0.62).astype(np.uint8))
            dr = ImageDraw.Draw(img)
        if tb.door:   # v3b door_bright
            x0, y0, x1, y1 = sc.door_x + 22, 70, sc.door_x + 88, G.WALL_Y
            dr.rectangle([x0, y0, x1, y1 - 50], fill=(150, 210, 255))
            dr.rectangle([x0, y1 - 50, x1, y1], fill=(90, 190, 80))
        T.draw_lamp(dr, tb.light, {40: 330, 184: 380, 320: 160}[sc.door_x])
        for i, u in enumerate(tb.units):
            if u.person:
                h, w = dims(u.k)
                bob = 2 * ((f + i) % 2)
                G.draw_person(dr, u.x, FEET - bob, u.colour, None, G.Style(fig_h=h, fig_w=w))
        arr = np.array(img)
        for u in tb.units:
            if not u.person:
                paint_decoy(arr, u)
        out.append(arr)
    return np.stack(out)


def tfacts_c(tb: TableauC) -> dict:
    ps = [u for u in tb.units if u.person]
    return {"n": len(ps), "colours": sorted(u.colour for u in ps),
            "decoy_colours": sorted(u.colour for u in tb.units if not u.person),
            "door": tb.door, "light": tb.light, "obj": None, "sign": None, "hat": None,
            "side": None}


def make_group(fam, gseed, style=None, split="eval", qset=None):
    if fam not in V3C:
        return T.make_group(fam, gseed, style, split, qset)
    style = style or TStyle(count_gap=True, door_bright=True)
    qset = qset or (1 if split == "train" else 2)
    key = f"v3c-{split}-{fam}-{gseed}"
    rng = random.Random(key)
    scene = G.make_scene(rng, split)
    door, light = rng.random() < 0.5, rng.random() < 0.5
    xs = [44 + 51 * s for s in range(8)]
    ks = [rng.uniform(0.85, 1.1) for _ in range(8)]
    if fam == "count":
        opts = [str(x) for x in rng.choice([[1, 3, 5, 7], [2, 4, 6, 8]])]
        cols = rng.sample(CNAMES, 8)
        tbs = []
        for v, o in enumerate(opts):
            pers = set(random.Random(f"{key}-{v}").sample(range(8), int(o)))
            tbs.append(TableauC(scene, [Unit(xs[s], ks[s], cols[s], s in pers) for s in range(8)],
                                door, light, style))
    else:
        cs = rng.sample(CNAMES, 4)
        opts = list(cs)
        pool = [c for c in CNAMES if c not in cs]
        n = rng.randint(1, 4)
        slots = sorted(rng.sample(range(8), n + 3))
        rng.shuffle(slots)
        pslots, dslots = slots[:n], slots[n:]
        k = rng.uniform(0.85, 1.1)                    # one size: shirt <-> box swap is exact
        others = rng.sample(pool, n - 1)
        tbs = []
        for v, c in enumerate(cs):
            r = random.Random(f"{key}-{v}")
            rest = [x for x in cs if x != c]
            r.shuffle(rest)
            units = [Unit(xs[pslots[0]], k, c, True)]
            units += [Unit(xs[s], k, oc, True) for s, oc in zip(pslots[1:], others)]
            units += [Unit(xs[s], k, dc, False, xs[pslots[0]]) for s, dc in zip(dslots, rest)]
            units.sort(key=lambda u: u.x)
            tbs.append(TableauC(scene, units, door, light, style))
    prompt = "\n".join([INTRO + Q3C[fam][qset - 1]] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL])
    out = []
    for v, tb in enumerate(tbs):
        f = tfacts_c(tb)
        ans = T.answer_from_facts(fam, f, opts)
        assert ans == v, (fam, gseed, v, ans, f)
        if fam == "colour":
            assert set(f["decoy_colours"]) | {opts[v]} == set(opts)
        out.append({"family": fam, "group": gseed, "variant": v, "split": split, "tableau": tb,
                    "filler_seed": key, "prompt": prompt, "options": opts, "answer": ans, "facts": f})
    return out


def stream(it) -> dict:
    fs = it["filler_seed"]
    pre = render_filler(fs, PRE * FPS, 0)
    tab = render_tableau(it["tableau"], E * FPS)
    post = render_filler(fs, POST * FPS, (PRE + E) * FPS)
    win = render_filler(fs, WIN * FPS, (PRE + E + POST) * FPS)
    return {"pre": pre, "tab": tab, "post": post, "win": win,
            "full": np.concatenate([pre, tab, post, win])}


def contact(items, path, thumb=224, ncol=4):
    """One still per item (first tableau frame), 4 variants of a group per row."""
    font = ImageFont.truetype(G.FONT.replace("-Bold", ""), 13)
    rows = (len(items) + ncol - 1) // ncol
    pad = 34
    sheet = Image.new("RGB", (ncol * thumb, rows * (thumb + pad)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        fr = render_tableau(it["tableau"], 1)[0]
        x0, y0 = (i % ncol) * thumb, (i // ncol) * (thumb + pad)
        f = it["facts"]
        d.text((x0 + 3, y0 + 2), f"{it['family']} g{it['group']} v{it['variant']} ans={it['options'][it['answer']]}",
               fill=(0, 0, 0), font=font)
        d.text((x0 + 3, y0 + 17), f"n={f['n']} shirts={','.join(f['colours'])}"[:34], fill=(80, 0, 0), font=font)
        sheet.paste(Image.fromarray(fr).resize((thumb, thumb), Image.BILINEAR), (x0, y0 + pad))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", default="contact_v3c.png")
    ap.add_argument("--groups", default="0,1")
    a = ap.parse_args()
    items = []
    for fam in V3C:
        for g in a.groups.split(","):
            items += make_group(fam, int(g))
    contact(items, a.contact)
    print("saved", a.contact)


if __name__ == "__main__":
    main()
