"""v4 "composited" generator: the v3 facts drawn ONTO real FineVideo footage.

Why v4 (lead, 22/10): a cartoon room inside real footage is trivially detectable, so a
trained memory could learn "keep cartoon-looking frames" (an anomaly detector) instead of
relevance. In v4 the episode is the stream's own footage with agents/objects/sign/hat
composited on top: it differs from the surrounding stream by CONTENT, not by style.

Stream of one item (2 fps, 448 px square centre crop, all from ONE FineVideo clip):
    [PRE 4 s][EPISODE 6 s = footage + overlay][POST 8 s][WINDOW 8 s]   = 26 s
The footage keeps moving under the overlay. The overlay (agents breathe +-2 px) is
alpha-composited with a dark soft shadow under the feet and a light halo around every
drawn shape, so it stays legible on any background.

Counterfactual group = same clip, same start, same overlay layout parameters (agent size,
sign position, object slot set by the group), same question and options; only the tested
fact changes. Everything outside the overlay mask is pixel-identical across variants.

Splits (FineVideo):
  train backgrounds: stream_targets.is_eval(id) == False, minus the ladder's 120 filler ids
  eval  backgrounds: is_eval(id) == True, minus the ladder's 120 filler ids  (disjoint from
                     the graft_v3 training pool, which is is_eval False minus the same 120)
  + seed salt per split, held-out hat colours (pink/white eval only), sign font (sans train /
  serif eval), agent size range (train 105-125 px, eval 115-135 px), paraphrased templates.

Load cue (shared with the ladder, ladder3.py CUE_WALL): turquoise (64, 200, 200). Here the
target episode gets a turquoise border (10 px) and the question names it; distractor
episodes (other variants of the same group, as in the ladder) have no border.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import gen_scenes as G

FPS = 2
SIZE = 448
PRE, E, POST, WIN = 4, 6, 8, 8
TOTAL = PRE + E + POST + WIN
FV = os.path.expanduser("~/work/titans-mac/data/finevideo")      # read only
CNAMES = G.CNAMES
COLORS = G.COLORS
LET = "ABCD"
OBJECTS = ["ball", "box", "cone", "star"]
OBJ_COL = (90, 140, 200)
CUE = (64, 200, 200)                    # ladder's turquoise
HALO = (245, 245, 245)                  # v4c: (30, 30, 30), a white shirt merged with the light halo
OBJ_SCALE = 0.66                        # v4c: 0.9 (object height / agent height)
CUE_STYLE = "border"                    # "border" (10 px) | "banner" (24 px border + "TARGET" label)
FEET = 410                              # agents stand on the lower part of the frame
COUNT_SETS = [[1, 3, 5, 7], [2, 4, 6, 8]]   # v4b: [[2, 4, 6, 8]] (n = 1 read as 3 on footage)
FAMILIES = ["count", "colour", "big_object", "hat_color", "sign_number"]
GIST = ["count", "colour", "big_object"]
DETAIL = ["hat_color", "sign_number"]


# ------------------------------------------------------------------ FineVideo pools
_POOLS = {}


def _ladder_ids(n=120, seed=0, min_dur=200):
    rows = [json.loads(l) for l in open(os.path.join(FV, "manifest.jsonl"))]
    rows = [r for r in rows if r["duration"] >= min_dur]
    random.Random(seed).shuffle(rows)
    return {r["id"] for r in rows[:n]}           # == ladder core.fv_ids(120)


def is_eval(vid: str, frac: float = 0.1) -> bool:   # == titans.graft.stream_targets.is_eval
    return (int(hashlib.md5(vid.encode()).hexdigest()[:8], 16) % 10000) / 10000 < frac


def pool(split):
    if split not in _POOLS:
        rows = [json.loads(l) for l in open(os.path.join(FV, "manifest.jsonl"))]
        lad = _ladder_ids()
        ok = [(r["id"], r["duration"]) for r in rows
              if r["id"] not in lad and r["duration"] >= 60 and is_eval(r["id"]) == (split == "eval")
              and os.path.exists(os.path.join(FV, "videos", r["id"] + ".mp4"))]
        _POOLS[split] = sorted(ok)
    return _POOLS[split]


def square(fr, side=SIZE):
    T, H, W, _ = fr.shape
    s = min(H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    return np.stack([np.asarray(Image.fromarray(f[y0:y0 + s, x0:x0 + s]).resize((side, side), Image.BILINEAR))
                     for f in fr])


_CLIP_CACHE = {}


def read_clip(vid, start):
    key = (vid, start)
    if key not in _CLIP_CACHE:
        import decord
        vr = decord.VideoReader(os.path.join(FV, "videos", vid + ".mp4"))
        src = float(vr.get_avg_fps())
        idx = [min(len(vr) - 1, int(round((start + i / FPS) * src))) for i in range(TOTAL * FPS)]
        if len(_CLIP_CACHE) > 16:
            _CLIP_CACHE.pop(next(iter(_CLIP_CACHE)))
        _CLIP_CACHE[key] = square(vr.get_batch(idx).asnumpy())
    return _CLIP_CACHE[key]


# ------------------------------------------------------------------ overlay
@dataclass
class Overlay:
    agents: list              # [(colour, x, hat)]
    obj: str | None
    obj_x: int
    sign: int | None
    sign_xy: tuple
    fig_h: int
    split: str
    sign_font: int = 20


def draw_object(dr, kind, x, s, feet):
    y1 = feet
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


def overlay_rgba(ov: Overlay, f: int) -> Image.Image:
    """RGBA layer for frame f: halo + shadow + drawn content."""
    fig = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    dr = ImageDraw.Draw(fig)
    st = G.Style(fig_h=ov.fig_h, fig_w=int(ov.fig_h * 0.32))
    if ov.sign is not None:
        x, y = ov.sign_xy
        s = ov.sign_font
        font = ImageFont.truetype(G.FONT if ov.split == "train" else G.FONT_EVAL, s)
        dr.rectangle([x, y, x + 3.0 * s, y + 1.5 * s], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        dr.text((x + 1.5 * s, y + 0.75 * s), str(ov.sign), fill=(0, 0, 0), font=font, anchor="mm")
    if ov.obj:
        draw_object(dr, ov.obj, ov.obj_x, int(ov.fig_h * OBJ_SCALE), FEET)
    for i, (c, x, hat) in enumerate(ov.agents):
        bob = 2 * ((f + i) % 2)
        G.draw_person(dr, x, FEET - bob, c, hat, st)
    a = np.asarray(fig)[..., 3]
    halo = Image.fromarray(a).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.5))
    out = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    # soft shadow under each standing thing
    sh = Image.new("L", (SIZE, SIZE), 0)
    sd = ImageDraw.Draw(sh)
    for (_, x, _) in ov.agents:
        sd.ellipse([x - ov.fig_h * 0.25, FEET - 8, x + ov.fig_h * 0.25, FEET + 8], fill=120)
    if ov.obj:
        sd.ellipse([ov.obj_x - ov.fig_h * 0.4, FEET - 8, ov.obj_x + ov.fig_h * 0.4, FEET + 8], fill=120)
    sh = sh.filter(ImageFilter.GaussianBlur(4))
    out = Image.composite(Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 255)), out, sh)
    halo_l = Image.new("RGBA", (SIZE, SIZE), HALO + (255,))
    out = Image.composite(halo_l, out, halo.point(lambda v: int(v * 0.8)))
    out.alpha_composite(fig)
    return out


def composite(bg: np.ndarray, ov: Overlay, border=None) -> np.ndarray:
    out = []
    for f, frame in enumerate(bg):
        im = Image.fromarray(frame).convert("RGBA")
        im.alpha_composite(overlay_rgba(ov, f))
        im = im.convert("RGB")
        if border is not None:
            d = ImageDraw.Draw(im)
            if CUE_STYLE == "banner":
                d.rectangle([0, 0, SIZE - 1, SIZE - 1], outline=border, width=24)
                d.rectangle([SIZE // 2 - 80, 0, SIZE // 2 + 80, 40], fill=border)
                d.text((SIZE // 2, 20), "TARGET", fill=(0, 0, 0),
                       font=ImageFont.truetype(G.FONT, 26), anchor="mm")
            else:
                d.rectangle([0, 0, SIZE - 1, SIZE - 1], outline=border, width=10)
        out.append(np.asarray(im))
    return np.stack(out)


# ------------------------------------------------------------------ sampling
SLOTS = [44 + 51 * i for i in range(8)]


def sample_overlay(rng, grp, *, n=None, colours=None, obj="rand", sign=None, hat=None, pool_c=None):
    """grp: group-level layout (fig_h, sign_xy, split)."""
    if n is None:
        n = rng.randint(1, 4)
    pc = list(pool_c or CNAMES)
    cols = list(colours or [])
    rest = [c for c in pc if c not in cols]
    rng.shuffle(rest)
    cols = (cols + rest)[:n]
    rng.shuffle(cols)
    if obj == "rand":
        obj = rng.choice(OBJECTS + [None, None])
    oslot = rng.randrange(7) if obj else None
    avoid = {oslot, oslot + 1} if obj else set()
    slots = sorted(rng.sample([s for s in range(8) if s not in avoid], n))
    agents = [(c, SLOTS[s], None) for c, s in zip(cols, slots)]
    if hat is not None:
        i = rng.randrange(n)
        agents[i] = (agents[i][0], agents[i][1], hat)
    return Overlay(agents, obj, SLOTS[oslot] + 25 if obj else 0, sign, grp["sign_xy"], grp["fig_h"], grp["split"])


def ofacts(ov):
    return {"n": len(ov.agents), "colours": sorted(c for c, _, _ in ov.agents), "obj": ov.obj,
            "sign": ov.sign, "hat": next((h for _, _, h in ov.agents if h), None)}


Q = {
    "count": ("How many cartoon people were drawn over the footage?",
              "At one point cartoon figures appeared over the video. How many were there?"),
    "colour": ("Which of these shirt colours did one of the cartoon people wear?",
               "One of the cartoon figures wore a shirt of one of these colours. Which one?"),
    "big_object": ("What large cartoon object was drawn next to the cartoon people?",
                   "Which big drawn object appeared next to the cartoon figures?"),
    "hat_color": ("One cartoon person wore a hat. What colour was the hat?",
                  "One of the cartoon figures had a hat on. Which colour was that hat?"),
    "sign_number": ("A small sign with a number was drawn over the footage. What number was on it?",
                    "Which number was written on the small drawn sign?"),
}
INTRO = ("The video stream shows real footage. At one point, cartoon people, each wearing a "
         "coloured shirt, are drawn over the footage. ")
CUE_INTRO = ("The video stream shows real footage. At several points, cartoon people, each wearing a "
             "coloured shirt, are drawn over the footage. Only one of these clips has a turquoise "
             "border. The question is about the clip with the turquoise border. ")
TAIL = "Answer with the option's letter from the given choices directly."


def answer_from_facts(fam, f, opts):
    if fam == "count":
        return opts.index(str(f["n"]))
    if fam == "colour":
        hits = [i for i, o in enumerate(opts) if o in f["colours"]]
        assert len(hits) == 1, (opts, f)
        return hits[0]
    if fam == "big_object":
        return opts.index(f"a {f['obj']}")
    if fam == "hat_color":
        return opts.index(f["hat"])
    if fam == "sign_number":
        return opts.index(str(f["sign"]))
    raise ValueError(fam)


def make_group(fam, gseed, split="eval", qset=None):
    assert split in G.SPLITS
    qset = qset or (1 if split == "train" else 2)
    key = f"v4-{split}-{fam}-{gseed}"
    rng = random.Random(key)
    vids = pool(split)
    vid, dur = vids[rng.randrange(len(vids))]
    start = round(rng.uniform(10, dur - TOTAL - 5), 1)
    lo, hi = (105, 125) if split == "train" else (115, 135)
    grp = {"fig_h": rng.randint(lo, hi), "split": split,
           "sign_xy": (rng.randint(20, SIZE - 90), rng.randint(30, 150))}
    vr = lambda v: random.Random(f"{key}-{v}")       # noqa: E731
    if fam == "count":
        opts = [str(x) for x in rng.choice(COUNT_SETS)]
        ovs = [sample_overlay(vr(v), grp, n=int(o), obj=None) for v, o in enumerate(opts)]
    elif fam == "colour":
        cs = rng.sample(CNAMES, 4)
        opts = list(cs)
        rest = [c for c in CNAMES if c not in cs]
        ovs = []
        for v, c in enumerate(cs):
            r = vr(v)
            ovs.append(sample_overlay(r, grp, n=r.randint(1, 4), colours=[c], pool_c=[c] + rest))
    elif fam == "big_object":
        opts = [f"a {o}" for o in OBJECTS]
        ovs = [sample_overlay(vr(v), grp, obj=o) for v, o in enumerate(OBJECTS)]
    elif fam == "hat_color":
        hp = CNAMES if split == "eval" else [c for c in CNAMES if c not in G.HAT_HELDOUT]
        hats = rng.sample(hp, 4)
        opts = list(hats)
        rest = [c for c in CNAMES if c not in hats]
        ovs = [sample_overlay(vr(v), grp, hat=h, pool_c=rest) for v, h in enumerate(hats)]
    elif fam == "sign_number":
        nums = sorted(rng.sample(range(10, 100), 4))
        opts = [str(x) for x in nums]
        ovs = [sample_overlay(vr(v), grp, sign=n) for v, n in enumerate(nums)]
    else:
        raise ValueError(fam)
    prompt_q = Q[fam][qset - 1] + "\n" + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts)) + "\n" + TAIL
    out = []
    for v, ov in enumerate(ovs):
        f = ofacts(ov)
        ans = answer_from_facts(fam, f, opts)
        assert ans == v, (fam, gseed, v, ans, f)
        out.append({"family": fam, "group": gseed, "variant": v, "split": split, "overlay": ov,
                    "vid": vid, "start": start, "prompt": INTRO + prompt_q,
                    "prompt_cue": CUE_INTRO + prompt_q, "options": opts, "answer": ans, "facts": f})
    return out


def stream(it, border=None) -> dict:
    bg = read_clip(it["vid"], it["start"])
    a, b = PRE * FPS, (PRE + E) * FPS
    ep = composite(bg[a:b], it["overlay"], border)
    full = np.concatenate([bg[:a], ep, bg[b:]])
    return {"pre": bg[:a], "ep": ep, "ep_bg": bg[a:b], "post": bg[b:b + POST * FPS],
            "win": bg[-WIN * FPS:], "full": full}


def contact(items, path, every=2, thumb=200, ncol=13):
    rows, labels = [], []
    for it in items:
        rows.append(stream(it)["full"][::every])
        f = it["facts"]
        labels.append(f"{it['family']} v{it['variant']} [{it['vid']} @{it['start']}s]: ans={it['options'][it['answer']]} | "
                      f"n={f['n']} cols={f['colours']} obj={f['obj']} hat={f['hat']} sign={f['sign']}")
    pad = 24
    nl = [(len(r) + ncol - 1) // ncol for r in rows]
    sheet = Image.new("RGB", (ncol * thumb, sum(pad + n * thumb for n in nl)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(G.FONT.replace("-Bold", ""), 14)
    y = 0
    for r, lab, n in zip(rows, labels, nl):
        d.text((4, y + 4), lab, fill=(0, 0, 0), font=font)
        y += pad
        for i, fr in enumerate(r):
            x0, y0 = (i % ncol) * thumb, y + (i // ncol) * thumb
            sheet.paste(Image.fromarray(fr).resize((thumb, thumb), Image.BILINEAR), (x0, y0))
            d.text((x0 + 3, y0 + 2), f"{i * every / FPS:.0f}s", fill=(255, 255, 0), font=font)
        y += n * thumb
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", default="contact_v4.png")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--variants", default="0,1")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    items = []
    for fam in a.families.split(","):
        g = make_group(fam, a.group, a.split)
        items += [g[int(v)] for v in a.variants.split(",")]
    contact(items, a.contact, a.every)
    print("saved", a.contact, "pools:", {s: len(pool(s)) for s in G.SPLITS})


if __name__ == "__main__":
    main()
