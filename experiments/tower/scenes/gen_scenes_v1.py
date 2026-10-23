"""Procedural "gist vs detail" scene generator.

A static room (wall, floor, one door). Cartoon people (one shirt colour each) stand
in a row; during the episode some walk out through the side edges, some walk in, the
door may open or close, a numbered sign may appear briefly. Everything is seeded.

Design choices (see bilan.md):
  * one shirt colour per person, from 8 easily named colours  -> identity by colour;
  * people stand on fixed slots on a back line; to leave they step DOWN to a front
    lane then walk sideways out of frame (no occlusion of the row while counting);
  * a still hold at the start (HOLD0 s) and at the end (HOLD1 s >= 4 s), so the
    "last frames" condition shows only the final state;
  * questions come in COUNTERFACTUAL GROUPS of 4: the same background, the same
    question text and the same 4 options, but each variant's video makes a
    different option correct. A text-only model therefore scores exactly 1/4 on a
    group (it gives the same answer four times).

    python gen_scenes.py --contact out.png --n 3
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ constants
FPS = 2
SIZE = 448
COLORS = {"red": (220, 30, 30), "blue": (30, 70, 230), "green": (30, 160, 50),
          "yellow": (245, 215, 20), "purple": (135, 40, 175), "orange": (245, 130, 20),
          "pink": (250, 120, 200), "white": (250, 250, 250)}
CNAMES = list(COLORS)
SKIN = (235, 190, 150)
LEGS = (45, 45, 60)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LET = "ABCD"


@dataclass
class Style:
    """The knobs iterated on in step 3."""
    fig_h: int = 120          # person height, px
    fig_w: int = 38           # body width
    hat_scale: float = 1.0    # hat size multiplier (detail calibration)
    sign_font: int = 26       # sign digit size (detail calibration)
    hold0: int = 8            # frames of still start (2 fps)
    hold1: int = 10           # frames of still end (>= 8 for the 4 s last-frames cut)
    ev_len: int = 5           # frames per enter/leave walk
    gap: int = 1              # still frames between events
    sign_len: int = 4         # frames the sign is visible
    clock: bool = False       # burn a running clock (seconds) in the top-left corner
    qset: int = 1             # question wording version (1 = original, 2 = revised)


@dataclass
class Scene:
    wall: tuple
    floor: tuple
    door_x: int
    slots: list               # x centres of standing slots


@dataclass
class Agent:
    color: str
    slot: int
    hat: str | None = None
    initial: bool = True


@dataclass
class Episode:
    scene: Scene
    agents: list
    door0: bool               # door open at start?
    events: list              # ("leave", i) | ("enter", i) | ("door", open_bool) | ("sign", num)
    style: Style = field(default_factory=Style)
    # filled by layout()
    timeline: list = field(default_factory=list)   # (event, f0, f1)
    n_frames: int = 0


# ------------------------------------------------------------------ scene
def make_scene(rng: random.Random) -> Scene:
    base = rng.choice([(150, 162, 178), (165, 160, 150), (140, 165, 150), (170, 150, 160)])
    wall = tuple(int(c + rng.randint(-8, 8)) for c in base)
    floor = rng.choice([(105, 92, 80), (95, 95, 100), (110, 100, 70)])
    door_x = rng.choice([40, 184, 320])
    slots = [44 + 51 * i for i in range(8)]
    return Scene(wall, floor, door_x, slots)


def layout(ep: Episode) -> Episode:
    s = ep.style
    t = s.hold0
    tl = []
    for ev in ep.events:
        d = s.sign_len if ev[0] == "sign" else (2 if ev[0] == "door" else s.ev_len)
        tl.append((ev, t, t + d))
        t += d + s.gap
    ep.timeline = tl
    ep.n_frames = t + s.hold1
    return ep


# ------------------------------------------------------------------ drawing
WALL_Y = 250                  # wall/floor boundary
BACK_FEET = 372
FRONT_FEET = 440


def draw_person(dr: ImageDraw.ImageDraw, x: float, feet: float, color, hat, st: Style):
    h, w = st.fig_h, st.fig_w
    top = feet - h
    r = h * 0.11
    cx = x
    # legs
    lw = w * 0.32
    dr.rectangle([cx - w * 0.38, feet - h * 0.33, cx - w * 0.38 + lw, feet], fill=LEGS)
    dr.rectangle([cx + w * 0.38 - lw, feet - h * 0.33, cx + w * 0.38, feet], fill=LEGS)
    # body
    dr.rounded_rectangle([cx - w / 2, top + 2 * r + 2, cx + w / 2, feet - h * 0.31],
                         radius=int(w * 0.25), fill=COLORS[color], outline=(20, 20, 20), width=2)
    # head
    dr.ellipse([cx - r, top, cx + r, top + 2 * r], fill=SKIN, outline=(20, 20, 20), width=2)
    if hat:
        k = st.hat_scale
        bw, bh, cw, ch = 1.25 * r * k, 0.22 * r * k, 0.8 * r * k, 0.75 * r * k
        dr.rectangle([cx - cw, top - ch, cx + cw, top + 0.2 * r], fill=COLORS[hat],
                     outline=(20, 20, 20), width=1)
        dr.rectangle([cx - bw, top + 0.2 * r - bh, cx + bw, top + 0.2 * r + bh / 2],
                     fill=COLORS[hat], outline=(20, 20, 20), width=1)


def draw_room(img: Image.Image, sc: Scene, door_open: bool):
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, SIZE, WALL_Y], fill=sc.wall)
    dr.rectangle([0, WALL_Y, SIZE, SIZE], fill=sc.floor)
    dr.line([0, WALL_Y, SIZE, WALL_Y], fill=(60, 60, 60), width=3)
    x0, y0, x1, y1 = sc.door_x, 70, sc.door_x + 88, WALL_Y
    dr.rectangle([x0 - 6, y0 - 6, x1 + 6, y1], fill=(240, 240, 235))     # frame
    if door_open:
        dr.rectangle([x0, y0, x1, y1], fill=(15, 15, 25))                 # dark opening
        dr.polygon([(x0, y0), (x0 + 22, y0 + 14), (x0 + 22, y1 + 10), (x0, y1)],
                   fill=(125, 72, 30), outline=(40, 25, 10))               # swung panel
    else:
        dr.rectangle([x0, y0, x1, y1], fill=(125, 72, 30), outline=(40, 25, 10), width=2)
        dr.rectangle([x0 + 10, y0 + 12, x1 - 10, y0 + 70], outline=(80, 45, 15), width=2)
        dr.rectangle([x0 + 10, y0 + 90, x1 - 10, y1 - 12], outline=(80, 45, 15), width=2)
        dr.ellipse([x1 - 18, 160, x1 - 8, 170], fill=(230, 200, 60))
    return dr


def sign_box(sc: Scene):
    # on the wall, away from the door
    x = 300 if sc.door_x < 200 else 60
    return x, 110


def path_pos(p: float, xs: float, side: int):
    """Leave path: slot -> front lane (first 30 %) -> off the edge (side -1/+1)."""
    edge = -60 if side < 0 else SIZE + 60
    if p <= 0.3:
        q = p / 0.3
        return xs, BACK_FEET + q * (FRONT_FEET - BACK_FEET), True
    q = (p - 0.3) / 0.7
    return xs + q * (edge - xs), FRONT_FEET, True


def render(ep: Episode) -> np.ndarray:
    """[T, H, W, 3] uint8."""
    st, sc = ep.style, ep.scene
    frames = []
    font = ImageFont.truetype(FONT, st.sign_font)
    for f in range(ep.n_frames):
        door = ep.door0
        sign = None
        pos = {}   # agent -> (x, feet, front)
        for i, a in enumerate(ep.agents):
            if a.initial:
                pos[i] = (sc.slots[a.slot], BACK_FEET, False)
        for (ev, f0, f1) in ep.timeline:
            kind, arg = ev
            if kind == "door" and f >= f0:
                door = arg
            elif kind == "sign" and f0 <= f < f1:
                sign = arg
            elif kind in ("leave", "enter"):
                a = ep.agents[arg]
                xs = sc.slots[a.slot]
                side = -1 if xs < SIZE / 2 else 1
                if f < f0:
                    if kind == "enter":
                        pos.pop(arg, None)
                    continue
                p = min(1.0, (f - f0) / (f1 - f0 - 1))
                if kind == "leave":
                    if p >= 1.0:
                        pos.pop(arg, None)
                    else:
                        pos[arg] = path_pos(p, xs, side)
                else:
                    if p >= 1.0:
                        pos[arg] = (xs, BACK_FEET, False)
                    else:
                        pos[arg] = path_pos(1.0 - p, xs, side)
        img = Image.new("RGB", (SIZE, SIZE))
        dr = draw_room(img, sc, door)
        if sign is not None:
            x, y = sign_box(sc)
            dr.rectangle([x, y, x + 3.0 * st.sign_font, y + 1.5 * st.sign_font],
                         fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            dr.text((x + 1.5 * st.sign_font, y + 0.75 * st.sign_font), str(sign),
                    fill=(0, 0, 0), font=font, anchor="mm")
        if st.clock:
            cf = ImageFont.truetype(FONT, 22)
            dr.rectangle([4, 4, 70, 34], fill=(0, 0, 0))
            dr.text((37, 19), f"{f // FPS:02d}s", fill=(255, 255, 255), font=cf, anchor="mm")
        # back row first, front lane on top
        for front in (False, True):
            for i, (x, feet, fr) in sorted(pos.items(), key=lambda kv: kv[1][1]):
                if fr == front:
                    a = ep.agents[i]
                    draw_person(dr, x, feet, a.color, a.hat, st)
        frames.append(np.asarray(img))
    return np.stack(frames)


# ------------------------------------------------------------------ episode sampling
def sample_episode(rng: random.Random, scene: Scene, style: Style, *, n0: int, n_left: int,
                   n_enter: int, door_events: list | None = None, door0: bool | None = None,
                   colors: list | None = None, leave_order: list | None = None,
                   hat: str | None = None, sign: int | None = None,
                   shirt_pool: list | None = None) -> Episode:
    """n0 initial people, n_left of them leave, n_enter newcomers come in.

    ``colors`` forces the initial shirt colours (first entries); ``leave_order``
    forces which initial colours leave and in which order (prefix of the leaves).
    """
    pool = list(shirt_pool or CNAMES)
    assert n0 + n_enter <= min(len(pool), 8)
    init_cols = list(colors or [])
    rest = [c for c in pool if c not in init_cols]
    rng.shuffle(rest)
    init_cols += rest[: n0 - len(init_cols)]
    rng.shuffle(init_cols)
    rest = [c for c in pool if c not in init_cols]
    rng.shuffle(rest)
    enter_cols = rest[:n_enter]
    slots = rng.sample(range(8), n0 + n_enter)
    agents = [Agent(c, slots[i]) for i, c in enumerate(init_cols)]
    agents += [Agent(c, slots[n0 + i], initial=False) for i, c in enumerate(enter_cols)]
    # leavers
    lo = list(leave_order or [])
    others = [c for c in init_cols if c not in lo]
    rng.shuffle(others)
    leavers = (lo + others)[:n_left]
    idx = {a.color: i for i, a in enumerate(agents)}
    leave_ev = [("leave", idx[c]) for c in leavers]
    enter_ev = [("enter", n0 + i) for i in range(n_enter)]
    # interleave: keep the forced leave order, random interleave of enters
    evs = list(leave_ev)
    for e in enter_ev:
        evs.insert(rng.randint(0, len(evs)), e)
    if hat is not None:
        # the hat goes on an initial person who leaves (seen, then gone)
        hi = idx[leavers[rng.randrange(len(leavers))]] if leavers else 0
        agents[hi].hat = hat
    if door0 is None:
        door0 = rng.random() < 0.5
    for d in (door_events or []):
        evs.insert(rng.randint(0, len(evs)), ("door", d))
    if sign is not None:
        evs.insert(rng.randint(0, len(evs)), ("sign", sign))
    return layout(Episode(scene, agents, door0, evs, style))


def facts(ep: Episode) -> dict:
    n0 = sum(a.initial for a in ep.agents)
    left = [ep.agents[a].color for (k, a) in ep.events if k == "leave"]
    ent = [ep.agents[a].color for (k, a) in ep.events if k == "enter"]
    door = ep.door0
    for (k, a) in ep.events:
        if k == "door":
            door = a
    return {"n0": n0, "n_left": len(left), "n_enter": len(ent), "n_final": n0 - len(left) + len(ent),
            "left": left, "entered": ent, "door0": ep.door0, "door1": door,
            "hat": next((a.hat for a in ep.agents if a.hat), None),
            "sign": next((a for (k, a) in ep.events if k == "sign"), None)}


# ------------------------------------------------------------------ question families
INTRO = ("This video shows cartoon people, each wearing a differently coloured shirt, "
         "in a room with a door. ")
TAIL = "Answer with the option's letter from the given choices directly."
DOOR_OPTS = ["closed at the start, open at the end", "open at the start, closed at the end",
             "open at the start and open at the end", "closed at the start and closed at the end"]
FAMILIES = ["init_count", "final_count", "n_left", "first_left", "order", "door_state",
            "hat_color", "sign_number"]
GIST = FAMILIES[:6]
DOOR_OPTS2 = ["it was closed, then it opened", "it was open, then it closed",
              "it stayed open the whole time", "it stayed closed the whole time"]


def _prompt(q: str, opts: list) -> str:
    return "\n".join([INTRO + q] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL])


def make_group(family: str, gseed: int, style: Style | None = None) -> list:
    """A counterfactual group: same scene, same question, same options; the 4 videos
    make each option correct once (2 options x 2 videos for the yes/no family)."""
    style = style or Style()
    rng = random.Random(f"{family}-{gseed}")
    scene = make_scene(rng)
    out = []

    def add(ep, q, opts, ans, v):
        out.append({"family": family, "group": gseed, "variant": v, "episode": ep,
                    "prompt": _prompt(q, opts), "options": opts, "answer": ans,
                    "facts": facts(ep)})

    if family in ("init_count", "final_count", "n_left"):
        if family == "init_count":
            vals = sorted(rng.sample(range(1, 7), 4))
            q = "How many people were in the room at the very start of the video?"
        elif family == "final_count":
            vals = sorted(rng.sample(range(0, 7), 4))
            q = ("How many people are in the room at the end of the video?" if style.qset == 1 else
                 "How many people are still in the room in the last frame of the video?")
        else:
            vals = [0, 1, 2, 3]
            q = ("How many people left the room during the video?" if style.qset == 1 else
                 "How many people walked out of the room during the video?")
        for v, val in enumerate(vals):
            r = random.Random(f"{family}-{gseed}-{v}")
            while True:
                n0 = r.randint(1, 6)
                nl = r.randint(1 if family != "n_left" else 0, min(3, n0))
                ne = r.randint(0, min(2, 8 - n0))
                if family == "init_count":
                    n0 = val
                    nl = r.randint(1, min(3, n0))
                    ne = r.randint(0, min(2, 8 - n0))
                elif family == "n_left":
                    nl = val
                    if n0 < nl:
                        continue
                nf = n0 - nl + ne
                if family == "final_count" and nf != val:
                    continue
                if family == "init_count" and nl == ne:   # final count must differ from initial
                    continue
                break
            ep = sample_episode(r, scene, style, n0=n0, n_left=nl, n_enter=ne,
                                door_events=[r.random() < 0.5] if r.random() < 0.5 else [])
            add(ep, q, [str(x) for x in vals], v, v)
    elif family == "first_left":
        cols = rng.sample(CNAMES, 4)
        q = "Which person left the room first?"
        opts = [f"the person in {c}" for c in cols]
        for v, c in enumerate(cols):
            r = random.Random(f"{family}-{gseed}-{v}")
            n0 = r.randint(4, 6)
            nl = r.randint(1, 3)
            ne = r.randint(0, min(2, 8 - n0))
            ep = sample_episode(r, scene, style, n0=n0, n_left=nl, n_enter=ne, colors=cols,
                                leave_order=[c], shirt_pool=CNAMES)
            add(ep, q, opts, v, v)
    elif family == "order":
        c = rng.choice(CNAMES)
        q = f"Did the door open before the person in {c} left the room?"
        opts = ["yes", "no"]
        if style.qset == 2:
            q = "Which of these two events happened first?"
            opts = ["the door opened", f"the person in {c} left the room"]
        for v in range(4):
            r = random.Random(f"{family}-{gseed}-{v}")
            want_yes = v % 2 == 0
            while True:
                n0 = r.randint(2, 5)
                nl = r.randint(1, min(3, n0))
                ne = r.randint(0, 1)
                ep = sample_episode(r, scene, style, n0=n0, n_left=nl, n_enter=ne, colors=[c],
                                    leave_order=[c] if r.random() < 0.5 else None,
                                    door0=False, door_events=[True])
                kinds = [(k, a) for (k, a) in ep.events]
                ci = next((i for i, (k, a) in enumerate(kinds)
                           if k == "leave" and ep.agents[a].color == c), None)
                di = next(i for i, (k, a) in enumerate(kinds) if k == "door")
                if ci is None:
                    continue
                if (di < ci) == want_yes:
                    break
            add(ep, q, opts, 0 if want_yes else 1, v)
    elif family == "door_state":
        q = "What was the state of the door at the start and at the end of the video?"
        dopts = DOOR_OPTS
        if style.qset == 2:
            q = "What happened to the door during the video?"
            dopts = DOOR_OPTS2
        plans = [(False, [True]), (True, [False]), (True, None), (False, None)]
        for v, (d0, evs) in enumerate(plans):
            r = random.Random(f"{family}-{gseed}-{v}")
            if evs is None:   # unchanged: either no event or a double toggle
                evs = [] if (r.random() < 0.5 or style.qset == 2) else [not d0, d0]
            n0 = r.randint(1, 5)
            ep = sample_episode(r, scene, style, n0=n0, n_left=r.randint(1, min(2, n0)),
                                n_enter=r.randint(0, 1), door0=d0, door_events=evs)
            add(ep, q, dopts, v, v)
    elif family == "hat_color":
        hats = rng.sample(CNAMES, 4)
        pool = [c for c in CNAMES if c not in hats]
        q = "One person wore a hat. What colour was the hat?"
        for v, h in enumerate(hats):
            r = random.Random(f"{family}-{gseed}-{v}")
            n0 = r.randint(2, 4)
            ep = sample_episode(r, scene, style, n0=n0, n_left=r.randint(1, min(2, n0)),
                                n_enter=0, hat=h, shirt_pool=pool)
            add(ep, q, hats, v, v)
    elif family == "sign_number":
        nums = sorted(rng.sample(range(10, 100), 4))
        q = "A sign with a number appeared briefly on the wall. What number was on it?"
        for v, n in enumerate(nums):
            r = random.Random(f"{family}-{gseed}-{v}")
            n0 = r.randint(1, 5)
            ep = sample_episode(r, scene, style, n0=n0, n_left=r.randint(1, min(2, n0)),
                                n_enter=r.randint(0, 1), sign=n)
            add(ep, q, [str(x) for x in nums], v, v)
    else:
        raise ValueError(family)
    return out


# ------------------------------------------------------------------ contact sheet
def contact_sheet(eps: list, path: str, every: int = 2, thumb: int = 224, labels=None,
                  ncol: int = 8):
    """One block per episode, wrapped at ``ncol`` thumbnails per line."""
    blocks = []
    for ep in eps:
        blocks.append(render(ep)[::every])
    pad = 26
    nlines = [(len(b) + ncol - 1) // ncol for b in blocks]
    H = sum(pad + n * thumb for n in nlines) + 8 * len(blocks)
    sheet = Image.new("RGB", (ncol * thumb, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT.replace("-Bold", ""), 14)
    y = 0
    for bi, fr in enumerate(blocks):
        if labels:
            d.text((4, y + 5), labels[bi], fill=(0, 0, 0), font=font)
        y += pad
        for ci, f in enumerate(fr):
            im = Image.fromarray(f).resize((thumb, thumb), Image.BILINEAR)
            x0, y0 = (ci % ncol) * thumb, y + (ci // ncol) * thumb
            sheet.paste(im, (x0, y0))
            d.rectangle([x0, y0, x0 + thumb - 1, y0 + thumb - 1], outline=(255, 255, 255))
            d.text((x0 + 4, y0 + 3), f"{ci * every / FPS:.0f}s", fill=(255, 255, 0), font=font)
        y += nlines[bi] * thumb + 8
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", default="contact.png")
    ap.add_argument("--families", default="init_count,first_left,hat_color")
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    eps, labels = [], []
    for fam in a.families.split(","):
        g = make_group(fam, 0)
        it = g[1]
        eps.append(it["episode"])
        f = it["facts"]
        labels.append(f"{fam}: ans={it['options'][it['answer']]}  n0={f['n0']} left={f['left']} "
                      f"entered={f['entered']} door {f['door0']}->{f['door1']} hat={f['hat']} "
                      f"sign={f['sign']}  T={it['episode'].n_frames / FPS:.1f}s")
        print(labels[-1])
    contact_sheet(eps, a.contact, every=a.every, labels=labels)
    print("saved", a.contact)


if __name__ == "__main__":
    main()
