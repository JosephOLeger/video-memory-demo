"""Procedural "gist vs detail" scene generator, v2 (post-audit 22/10).

A static room (wall, floor, one door). Cartoon people (one shirt colour each) stand
on a back row; during the episode some walk out through the side edges, some walk
in, the door may open or close, a numbered sign may appear briefly. All seeded.

v2 invariants (each one answers an audit item, see ~/work/iclr/audit/audit.md):
  * CONSTANT DURATION: every episode is exactly T_FRAMES frames; events are placed
    at random times with random idle padding, so T carries nothing (audit #1).
  * SHARED END STATE: within a counterfactual group the 4 variants end in the SAME
    final state (same people on the same slots, same door). The end hold (last
    HOLD1 frames) is therefore pixel-identical across the group, so any rule that
    reads only the end of the episode scores exactly chance per item and 0 per group
    on every memory family (audit #1, #10). `final_count` is the one exception and
    is kept on purpose as a WINDOW CONTROL (answer visible at the end).
  * first_left: all 4 offered colours leave in every variant; only the order
    changes (audit #10).
  * door family asks only for the START state (audit #1); toggles are applied in
    sequence, and make_group asserts answer == f(facts) for every item (audit #2).
  * no clock overlay (audit #8).
  * SPLITS (audit #7): seeds are salted with the split; eval uses held-out decor
    (at least one of: wall base 3, floor 2, door x=320), hat colours pink/white
    appear only in eval, the sign sits at another place in another font in eval,
    eval uses a paraphrased template (qset 2), and `n_left` is an eval-only family.

    python gen_scenes.py --contact out.png
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
T_FRAMES = 56                 # 28 s, constant for every episode
COLORS = {"red": (220, 30, 30), "blue": (30, 70, 230), "green": (30, 160, 50),
          "yellow": (245, 215, 20), "purple": (135, 40, 175), "orange": (245, 130, 20),
          "pink": (250, 120, 200), "white": (250, 250, 250)}
CNAMES = list(COLORS)
HAT_HELDOUT = ["pink", "white"]          # never a hat colour in train
SKIN = (235, 190, 150)
LEGS = (45, 45, 60)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_EVAL = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
LET = "ABCD"
WALLS = [(150, 162, 178), (165, 160, 150), (140, 165, 150), (170, 150, 160)]  # 3 = held out
FLOORS = [(105, 92, 80), (95, 95, 100), (110, 100, 70)]                      # 2 = held out
DOORS = [40, 184, 320]                                                       # 320 = held out
SPLITS = ("train", "eval")


@dataclass
class Style:
    """Rendering knobs."""
    fig_h: int = 120          # person height, px
    fig_w: int = 38           # body width
    hat_scale: float = 1.0    # hat size multiplier (detail calibration)
    sign_font: int = 26       # sign digit size (detail calibration)
    hold0: int = 8            # frames of still start (2 fps)
    hold1: int = 10           # frames of still end (>= 8 for the 4 s last-frames cut)
    ev_len: int = 5           # frames per enter/leave walk
    sign_len: int = 4         # frames the sign is visible
    door_len: int = 2
    t_frames: int = T_FRAMES  # constant episode length (frames)


@dataclass
class Scene:
    wall: tuple
    floor: tuple
    door_x: int
    slots: list               # x centres of standing slots
    split: str = "eval"


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
    timeline: list = field(default_factory=list)   # (event, f0, f1)
    n_frames: int = T_FRAMES


# ------------------------------------------------------------------ scene
def make_scene(rng: random.Random, split: str) -> Scene:
    while True:
        w, f, d = rng.randrange(4), rng.randrange(3), rng.randrange(3)
        held = (w == 3) or (f == 2) or (d == 2)
        if held == (split == "eval"):
            break
    wall = tuple(int(c + rng.randint(-8, 8)) for c in WALLS[w])
    slots = [44 + 51 * i for i in range(8)]
    return Scene(wall, FLOORS[f], DOORS[d], slots, split)


def ev_dur(ev, s: Style) -> int:
    return {"sign": s.sign_len, "door": s.door_len}.get(ev[0], s.ev_len)


def layout(ep: Episode, rng: random.Random) -> Episode:
    """Place the (ordered) events at random times inside the constant-length episode:
    still start, events separated by >= 1 idle frame plus random padding, still end."""
    s = ep.style
    durs = [ev_dur(e, s) + 1 for e in ep.events]          # +1 = minimum gap
    span = s.t_frames - s.hold0 - s.hold1
    slack = span - sum(durs)
    assert slack >= 0, f"too many events for t_frames ({sum(durs)} > {span})"
    cuts = sorted(rng.randint(0, slack) for _ in range(len(durs)))
    pads = [b - a for a, b in zip([0] + cuts, cuts + [slack])]
    t = s.hold0
    tl = []
    for ev, d, p in zip(ep.events, durs, pads):
        t += p
        tl.append((ev, t, t + d - 1))
        t += d
    ep.timeline = tl
    ep.n_frames = s.t_frames
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
    lw = w * 0.32
    dr.rectangle([cx - w * 0.38, feet - h * 0.33, cx - w * 0.38 + lw, feet], fill=LEGS)
    dr.rectangle([cx + w * 0.38 - lw, feet - h * 0.33, cx + w * 0.38, feet], fill=LEGS)
    dr.rounded_rectangle([cx - w / 2, top + 2 * r + 2, cx + w / 2, feet - h * 0.31],
                         radius=int(w * 0.25), fill=COLORS[color], outline=(20, 20, 20), width=2)
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
    dr.rectangle([x0 - 6, y0 - 6, x1 + 6, y1], fill=(240, 240, 235))
    if door_open:
        dr.rectangle([x0, y0, x1, y1], fill=(15, 15, 25))
        dr.polygon([(x0, y0), (x0 + 22, y0 + 14), (x0 + 22, y1 + 10), (x0, y1)],
                   fill=(125, 72, 30), outline=(40, 25, 10))
    else:
        dr.rectangle([x0, y0, x1, y1], fill=(125, 72, 30), outline=(40, 25, 10), width=2)
        dr.rectangle([x0 + 10, y0 + 12, x1 - 10, y0 + 70], outline=(80, 45, 15), width=2)
        dr.rectangle([x0 + 10, y0 + 90, x1 - 10, y1 - 12], outline=(80, 45, 15), width=2)
        dr.ellipse([x1 - 18, 160, x1 - 8, 170], fill=(230, 200, 60))
    return dr


def sign_box(sc: Scene):
    """On the wall, away from the door. Eval uses another height (held-out placement)."""
    x = 300 if sc.door_x < 200 else 60
    y = 110 if sc.split == "train" else 160
    return x, y


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
    font = ImageFont.truetype(FONT if sc.split == "train" else FONT_EVAL, st.sign_font)
    for f in range(ep.n_frames):
        door = ep.door0
        sign = None
        pos = {}
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
                if f < f0:
                    continue
                a = ep.agents[arg]
                xs = sc.slots[a.slot]
                side = -1 if xs < SIZE / 2 else 1
                p = min(1.0, (f - f0) / (f1 - f0 - 1))
                if kind == "leave":
                    if p >= 1.0:
                        pos.pop(arg, None)
                    else:
                        pos[arg] = path_pos(p, xs, side)
                else:
                    pos[arg] = (xs, BACK_FEET, False) if p >= 1.0 else path_pos(1.0 - p, xs, side)
        img = Image.new("RGB", (SIZE, SIZE))
        dr = draw_room(img, sc, door)
        if sign is not None:
            x, y = sign_box(sc)
            dr.rectangle([x, y, x + 3.0 * st.sign_font, y + 1.5 * st.sign_font],
                         fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            dr.text((x + 1.5 * st.sign_font, y + 0.75 * st.sign_font), str(sign),
                    fill=(0, 0, 0), font=font, anchor="mm")
        for front in (False, True):
            for i, (x, feet, fr) in sorted(pos.items(), key=lambda kv: kv[1][1]):
                if fr == front:
                    a = ep.agents[i]
                    draw_person(dr, x, feet, a.color, a.hat, st)
        frames.append(np.asarray(img))
    return np.stack(frames)


# ------------------------------------------------------------------ episode building
@dataclass
class Final:
    """The end state shared by every variant of a group."""
    stay: list                # [(colour, slot)] present from the start and at the end
    enter: list               # [(colour, slot)] walk in during the episode
    door1: bool


def make_final(rng, n_stay, n_enter, door1=None, forbid=()):
    cols = [c for c in CNAMES if c not in forbid]
    rng.shuffle(cols)
    slots = rng.sample(range(8), n_stay + n_enter)
    stay = list(zip(cols[:n_stay], slots[:n_stay]))
    enter = list(zip(cols[n_stay:n_stay + n_enter], slots[n_stay:]))
    if door1 is None:
        door1 = rng.random() < 0.5
    return Final(stay, enter, door1)


def build_episode(rng, scene, style, fin: Final, leavers, *, door0=None, leave_order_fixed=False,
                  hat=None, sign=None, constraint=None) -> Episode:
    """leavers = [colour] of extra initial people who leave (they take free slots).

    Events: the leaves (random order unless leave_order_fixed), the enters, one door
    toggle if door0 != door1, the sign; shuffled together. `constraint(events, agents)`
    may reject an ordering (resampled)."""
    used = {s for _, s in fin.stay + fin.enter}
    free = [s for s in range(8) if s not in used]
    lslots = rng.sample(free, len(leavers))
    agents = [Agent(c, s) for c, s in fin.stay]
    li = []
    for c, s in zip(leavers, lslots):
        li.append(len(agents))
        agents.append(Agent(c, s))
    ei = []
    for c, s in fin.enter:
        ei.append(len(agents))
        agents.append(Agent(c, s, initial=False))
    if hat is not None:
        agents[li[rng.randrange(len(li))]].hat = hat
    if door0 is None:
        door0 = rng.random() < 0.5
    for _ in range(1000):
        lev = [("leave", i) for i in li]
        if not leave_order_fixed:
            rng.shuffle(lev)
        evs = list(lev)
        others = [("enter", i) for i in ei]
        if door0 != fin.door1:
            others.append(("door", fin.door1))
        if sign is not None:
            others.append(("sign", sign))
        for e in others:
            evs.insert(rng.randint(0, len(evs)), e)
        if constraint is None or constraint(evs, agents):
            break
    else:
        raise RuntimeError("constraint unsatisfiable")
    return layout(Episode(scene, agents, door0, evs, style), rng)


def facts(ep: Episode) -> dict:
    """Ground truth recomputed from the episode itself (events applied in time order)."""
    n0 = sum(a.initial for a in ep.agents)
    tl = sorted(ep.timeline, key=lambda x: x[1])
    left = [ep.agents[a].color for ((k, a), _, _) in tl if k == "leave"]
    ent = [ep.agents[a].color for ((k, a), _, _) in tl if k == "enter"]
    door = ep.door0
    door_open_t = None
    for ((k, a), f0, _) in tl:
        if k == "door":
            door = a
            if a and door_open_t is None:
                door_open_t = f0
    leave_t = {ep.agents[a].color: f0 for ((k, a), f0, _) in tl if k == "leave"}
    final = sorted([(a.color, a.slot) for a in ep.agents
                    if (a.initial and a.color not in left) or not a.initial])
    return {"n0": n0, "n_left": len(left), "n_enter": len(ent),
            "n_final": n0 - len(left) + len(ent), "left": left, "entered": ent,
            "door0": ep.door0, "door1": door, "door_open_t": door_open_t, "leave_t": leave_t,
            "hat": next((a.hat for a in ep.agents if a.hat), None),
            "sign": next((a for ((k, a), _, _) in tl if k == "sign"), None),
            "final": final}


# ------------------------------------------------------------------ question families
INTRO = ("This video shows cartoon people, each wearing a differently coloured shirt, "
         "in a room with a door. ")
TAIL = "Answer with the option's letter from the given choices directly."
FAMILIES = ["init_count", "n_left", "first_left", "order", "door_start", "hat_color",
            "sign_number", "final_count"]
MEMORY = ["init_count", "n_left", "first_left", "order", "door_start", "hat_color", "sign_number"]
GIST = ["init_count", "n_left", "first_left", "order", "door_start"]
DETAIL = ["hat_color", "sign_number"]
CONTROL = ["final_count"]          # answer visible at the end: window control
EVAL_ONLY = ["n_left"]             # held-out family

# (train template, eval paraphrase)
Q = {
    "init_count": ("How many people were in the room at the very start of the video?",
                   "When the video began, how many people were standing in the room?"),
    "n_left": ("How many people left the room during the video?",
               "How many people walked out of the room over the course of the video?"),
    "first_left": ("Which person left the room first?",
                   "Who was the first person to walk out of the room?"),
    "order": ("Did the door open before the person in {c} left the room?",
              "Was the door opened before the person in {c} walked out?"),
    "door_start": ("Was the door open or closed at the very start of the video?",
                   "When the video began, what was the state of the door?"),
    "hat_color": ("One person wore a hat. What colour was the hat?",
                  "Someone in the video had a hat on. Which colour was that hat?"),
    "sign_number": ("A sign with a number appeared briefly on the wall. What number was on it?",
                    "For a moment a sign showing a number was on the wall. Which number?"),
    "final_count": ("How many people are in the room at the end of the video?",
                    "When the video ends, how many people are in the room?"),
}


def _prompt(q: str, opts: list) -> str:
    return "\n".join([INTRO + q] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL])


def answer_from_facts(family, f, opts, meta):
    """Independent recomputation of the gold option from the episode facts."""
    if family == "init_count":
        return opts.index(str(f["n0"]))
    if family == "n_left":
        return opts.index(str(f["n_left"]))
    if family == "final_count":
        return opts.index(str(f["n_final"]))
    if family == "first_left":
        return opts.index(f"the person in {f['left'][0]}")
    if family == "order":
        before = f["door_open_t"] is not None and f["door_open_t"] < f["leave_t"][meta["c"]]
        return 0 if before else 1
    if family == "door_start":
        return opts.index("open" if f["door0"] else "closed")
    if family == "hat_color":
        return opts.index(f["hat"])
    if family == "sign_number":
        return opts.index(str(f["sign"]))
    raise ValueError(family)


def make_group(family: str, gseed: int, style: Style | None = None, split: str = "eval",
               qset: int | None = None) -> list:
    """A counterfactual group: same scene, same question, same options, same END STATE
    (except final_count); the 4 videos make each option correct once (each of the 2
    options twice for the binary families)."""
    assert split in SPLITS
    if split == "train" and family in EVAL_ONLY:
        raise ValueError(f"{family} is a held-out (eval-only) family")
    style = style or Style()
    qset = qset or (1 if split == "train" else 2)
    rng = random.Random(f"{split}-{family}-{gseed}")
    scene = make_scene(rng, split)
    q = Q[family][qset - 1]
    out = []

    def vr(v):
        return random.Random(f"{split}-{family}-{gseed}-{v}")

    meta = {}
    if family == "init_count":
        while True:
            vals = sorted(rng.sample(range(1, 7), 4))
            lo, hi = max(0, vals[-1] - 4), vals[0]
            if lo <= hi:
                break
        n_stay = rng.randint(lo, hi)
        fin = make_final(rng, n_stay, rng.randint(0, 1))
        opts = [str(x) for x in vals]
        spec = []
        for v, n0 in enumerate(vals):
            r = vr(v)
            pool = [c for c in CNAMES if c not in [c for c, _ in fin.stay + fin.enter]]
            spec.append(dict(r=r, leavers=r.sample(pool, n0 - n_stay)))
    elif family == "n_left":
        fin = make_final(rng, rng.randint(0, 3), rng.randint(0, 1))
        opts = ["0", "1", "2", "3"]
        pool = [c for c in CNAMES if c not in [c for c, _ in fin.stay + fin.enter]]
        spec = [dict(r=vr(v), leavers=vr(v).sample(pool, v)) for v in range(4)]
    elif family == "final_count":
        vals = sorted(rng.sample(range(0, 7), 4))
        opts = [str(x) for x in vals]
        spec = []
        for v, nf in enumerate(vals):     # control: the end state differs by design
            r = vr(v)
            ne = r.randint(0, min(1, nf))
            fin_v = make_final(r, nf - ne, ne)
            pool = [c for c in CNAMES if c not in [c for c, _ in fin_v.stay + fin_v.enter]]
            spec.append(dict(r=r, fin=fin_v, leavers=r.sample(pool, r.randint(1, min(3, len(pool))))))
        fin = None
    elif family == "first_left":
        cols = rng.sample(CNAMES, 4)
        fin = make_final(rng, rng.randint(0, 2), rng.randint(0, 1), forbid=cols)
        opts = [f"the person in {c}" for c in cols]
        spec = []
        for v, c in enumerate(cols):
            r = vr(v)
            rest = [x for x in cols if x != c]
            r.shuffle(rest)
            spec.append(dict(r=r, leavers=[c] + rest, leave_order_fixed=True))
    elif family == "order":
        c = rng.choice(CNAMES)
        meta["c"] = c
        q = q.format(c=c)
        fin = make_final(rng, rng.randint(0, 3), rng.randint(0, 1), door1=True, forbid=[c])
        opts = ["yes", "no"]
        pool = [x for x in CNAMES if x != c and x not in [y for y, _ in fin.stay + fin.enter]]
        spec = []
        for v in range(4):
            r = vr(v)
            want = v % 2 == 0          # 0, 2 -> yes ; 1, 3 -> no
            extra = r.sample(pool, r.randint(0, min(2, len(pool))))

            def cons(evs, agents, want=want, c=c):
                di = next(i for i, e in enumerate(evs) if e[0] == "door")
                ci = next(i for i, e in enumerate(evs) if e[0] == "leave" and agents[e[1]].color == c)
                return (di < ci) == want
            spec.append(dict(r=r, leavers=[c] + extra, door0=False, constraint=cons))
    elif family == "door_start":
        fin = make_final(rng, rng.randint(0, 3), rng.randint(0, 1))
        opts = ["open", "closed"]
        pool = [c for c in CNAMES if c not in [c for c, _ in fin.stay + fin.enter]]
        spec = []
        for v in range(4):
            r = vr(v)
            spec.append(dict(r=r, leavers=r.sample(pool, r.randint(1, 2)), door0=(v % 2 == 0)))
    elif family == "hat_color":
        hat_pool = CNAMES if split == "eval" else [c for c in CNAMES if c not in HAT_HELDOUT]
        hats = rng.sample(hat_pool, 4)
        fin = make_final(rng, rng.randint(0, 2), rng.randint(0, 1), forbid=hats)
        opts = list(hats)
        pool = [c for c in CNAMES if c not in hats and c not in [y for y, _ in fin.stay + fin.enter]]
        spec = []
        for v, h in enumerate(hats):
            r = vr(v)
            spec.append(dict(r=r, leavers=r.sample(pool, r.randint(1, min(2, len(pool)))) if pool else [],
                             hat=h))
        if not pool:      # no shirt colour left for the hat wearer: retry with a smaller final
            return make_group(family, gseed + 100000, style, split, qset)
    elif family == "sign_number":
        nums = sorted(rng.sample(range(10, 100), 4))
        fin = make_final(rng, rng.randint(0, 3), rng.randint(0, 1))
        opts = [str(x) for x in nums]
        pool = [c for c in CNAMES if c not in [c for c, _ in fin.stay + fin.enter]]
        spec = [dict(r=vr(v), leavers=vr(v).sample(pool, vr(v).randint(1, 2)), sign=n)
                for v, n in enumerate(nums)]
    else:
        raise ValueError(family)

    prompt = _prompt(q, opts)
    for v, sp in enumerate(spec):
        r = sp.pop("r")
        f_v = sp.pop("fin", fin)
        ep = build_episode(r, scene, style, f_v, **sp)
        fa = facts(ep)
        ans = answer_from_facts(family, fa, opts, meta)
        gold = v % 2 if len(opts) == 2 else v
        assert ans == gold, (family, gseed, v, ans, gold, fa)
        out.append({"family": family, "group": gseed, "variant": v, "split": split,
                    "episode": ep, "prompt": prompt, "options": opts, "answer": ans,
                    "facts": fa})
    if family in MEMORY:     # shared end state, checked
        assert len({str(it["facts"]["final"]) + str(it["facts"]["door1"]) for it in out}) == 1
    return out


# ------------------------------------------------------------------ contact sheet
def contact_sheet(eps: list, path: str, every: int = 2, thumb: int = 224, labels=None,
                  ncol: int = 7):
    blocks = [render(ep)[::every] for ep in eps]
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
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--variants", default="1")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    eps, labels = [], []
    for fam in a.families.split(","):
        g = make_group(fam, a.group, split=a.split)
        for v in map(int, a.variants.split(",")):
            it = g[v]
            eps.append(it["episode"])
            f = it["facts"]
            labels.append(f"{fam} v{v}: ans={it['options'][it['answer']]} | n0={f['n0']} "
                          f"left={f['left']} entered={f['entered']} door {f['door0']}->{f['door1']} "
                          f"hat={f['hat']} sign={f['sign']} | T={it['episode'].n_frames / FPS:.0f}s")
            print(labels[-1])
    contact_sheet(eps, a.contact, every=a.every, labels=labels)
    print("saved", a.contact)


if __name__ == "__main__":
    main()
