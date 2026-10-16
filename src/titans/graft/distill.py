"""Context distillation: the memory has to replace the context we took away.

The objective, and why it is this one
-------------------------------------
EgoLifeQA ships 101 usable questions on the day we have. A cross-entropy over one
letter out of four is 1.39 nats of signal per example -- nowhere near enough to
train half a million parameters. So the graft is trained on a signal that needs no
labels at all and that *is* the paper's claim:

    teacher = the frozen host, seeing the WHOLE window
    student = the grafted host, seeing only the LAST clip, plus the memory
    loss    = cross-entropy of the student on the teacher's own continuation

Anything the student gets right beyond what the last clip supports had to come
through the memory. The teacher's greedy continuation is free supervision, and the
supply is the length of the video rather than the number of annotations.

Three properties this buys that the alternatives do not:

* It trains ``up``. A self-supervised loss on ``z`` (predicting the next latent,
  the world model's objective) never touches the projection that writes back into
  the residual stream, so the branch would learn to remember and never learn to
  *speak*.
* It has an exact floor and an exact ceiling, in the same units as the loss:
  the same student with the branch disabled (``up`` is zero at init, so this is
  the untouched host) and the teacher itself. The reported number is the fraction
  of that gap the memory closes -- the graft's analogue of ``memory_gain``.
* Its horizon is honest about its limit. The teacher can only see as far as the
  host's context window (~6 min of video here), so distillation only ever teaches
  reach up to that. Reach *beyond* it is the architecture's job and is measured
  zero-shot on LongVideoBench's 600 s and 3600 s slices -- which is the experiment
  worth publishing, not a gap to hide.

The probe asks about the BEGINNING
----------------------------------
``--probe`` defaults to a question about the *start* of the window. Ask about the
video in general and the answer is dominated by the last clip, which the student
can already see: the loss gap collapses and the memory is never driven. Asking
about the beginning puts the answer squarely in the part that was removed.

Two-pass usage -- targets first (teacher, expensive, cached), then training:

    python -m titans.graft.distill --make-targets --z cache/jake_day1_z.pt \\
        --out-targets cache/jake_day1_tgt.pt
    python -m titans.graft.distill --z cache/jake_day1_z.pt \\
        --targets cache/jake_day1_tgt.pt --basis cache/qwen_l24_pca128.pt \\
        --muon unit --write-stride 2 --alpha-init 1.27e-4 --out runs/graft_unit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .branch import (MemoryBranch, add_memory_flags, memory_kwargs_from_args,  # noqa: F401
                     stride_for_reach)
from .qwen import branch_report, default_layer, find_decoder_layers, graft

PROBE = ("Describe what the camera wearer was doing at the very beginning of "
         "this video.")


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def windows(n_clips: int, span: int, hop: int) -> list:
    """``(start, end)`` clip ranges. The student sees ``end-1``; memory gets the rest."""
    if span < 2:
        raise ValueError("span must be >= 2: the memory needs at least one clip")
    return [(s, s + span) for s in range(0, max(0, n_clips - span) + 1, hop)]


def split_windows(wins: list, val_frac: float = 0.1, embargo: int = 0) -> tuple:
    """A **contiguous tail** is held out, not a random subset, with an embargo.

    Two things are wrong with the obvious split, and both inflate the score.

    A *random* split puts nearly the same video on both sides -- consecutive windows
    overlap by ``span - hop`` clips -- so the validation curve would measure
    memorisation of the training stream.

    A contiguous tail alone is not enough either: the last training window and the
    first validation window still share up to ``span - hop`` clips. With
    ``span=12, hop=4`` that is **8 clips out of 12**, i.e. two thirds of the
    context the memory is asked to carry. ``embargo`` drops that many windows at
    the seam, which is the purged/embargoed split standard in time-series
    evaluation. Pass ``embargo = ceil(span/hop) - 1`` for a strictly disjoint
    split; :func:`main` computes it from the flags rather than leaving it to the
    caller to remember.

    With several days the tail is taken **per day** and the pieces concatenated.
    Holding out one whole day instead would confound "unseen window" with "unseen
    day": a day of EgoLife is a different room, a different activity and a
    different set of people, so a whole-day split would measure domain transfer
    rather than memory.
    """
    def cut_one(sub):
        n = len(sub)
        cut = max(1, int(n * (1.0 - val_frac)))
        # La purge est prise cote ENTRAINEMENT, pas cote validation. A
        # val_frac=0.1 un jour n'a que ~5 fenetres de validation contre ~43
        # d'entrainement : retirer 2 fenetres coute 40 % de la mesure d'un cote
        # et 5 % du signal de l'autre. On coupe donc dans l'abondant.
        return sub[:max(1, cut - embargo)], sub[cut:]

    if wins and len(wins[0]) == 3:
        train, val = [], []
        for d in sorted({w[0] for w in wins}):
            t, v = cut_one([w for w in wins if w[0] == d])
            train += t
            val += v
        return train, val
    return cut_one(wins)


# --------------------------------------------------------------------------- #
# teacher
# --------------------------------------------------------------------------- #
def load_window_frames(paths, lo: int, hi: int, num_frames: int, max_side: int):
    from ..worldmodel.data import load_video_frames
    from .fit_basis import resize_frames
    return torch.cat([resize_frames(load_video_frames(p, num_frames=num_frames),
                                    max_side) for p in paths[lo:hi]], dim=0)


@torch.no_grad()
def teacher_continuation(model, processor, frames, probe: str, *, fps: float,
                         max_new: int, device: str) -> torch.Tensor:
    """The frozen host's greedy answer, seeing the whole window. Returns token ids."""
    from .fit_basis import video_inputs
    inputs = video_inputs(processor, frames, probe, fps)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return out[0, inputs["input_ids"].shape[1]:].detach().cpu()


# --------------------------------------------------------------------------- #
# student
# --------------------------------------------------------------------------- #
#: Value to pad each per-token tensor with when ``y`` is appended. ``y`` is plain
#: text, so it attends (1) and is not a multimodal placeholder (0). Anything the
#: processor grows in the future defaults to 0 and trips the assertion below if
#: that is wrong, rather than being silently left short.
_PAD_PER_TOKEN = {"attention_mask": 1, "mm_token_type_ids": 0, "token_type_ids": 0}


def text_only_inputs(processor, prompt: str):
    """The prompt with no video at all -- the diagnostic's zero.

    How much of the teacher's answer is reachable from the question alone? If a
    lot, the task is mostly a language prior and no memory architecture can show
    up in it, whatever it retains.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    return processor(text=[text], return_tensors="pt")


def forced_inputs(processor, frames, probe: str, y: torch.Tensor, fps: float,
                  device: str):
    """``[prompt | y]`` teacher-forced, plus the index where ``y`` starts.

    Every tensor the processor emits **per token** has to grow with ``input_ids``,
    not just the attention mask. Qwen2.5-VL also returns ``mm_token_type_ids``, and
    leaving it at the prompt's length makes ``get_rope_index`` index a 684-long
    tensor with a 706-long mask -- an ``IndexError`` a hundred lines deep in
    ``transformers``, with nothing in it naming the cause.
    """
    from .fit_basis import video_inputs
    inp = (text_only_inputs(processor, probe) if frames is None
           else video_inputs(processor, frames, probe, fps))
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    n_prompt = int(inp["input_ids"].shape[1])
    m = int(y.shape[0])
    for k, v in list(inp.items()):
        if not torch.is_tensor(v) or v.dim() != 2 or v.shape[1] != n_prompt:
            continue                       # pixels / grids are not per text token
        if k == "input_ids":
            inp[k] = torch.cat([v, y.to(device).unsqueeze(0)], dim=1)
        else:
            fill = _PAD_PER_TOKEN.get(k, 0)
            inp[k] = torch.cat([v, torch.full((1, m), fill, dtype=v.dtype,
                                              device=v.device)], dim=1)
    grown = {k: tuple(v.shape) for k, v in inp.items()
             if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n_prompt}
    assert not grown, f"tenseurs par token restes a la longueur du prompt: {grown}"
    return inp, n_prompt


def forced_loss(model, inputs: dict, n_prompt: int, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of ``y`` given the prompt. Position ``n_prompt-1`` predicts ``y[0]``.

    ``use_cache=False`` is not an optimisation: the grafted layer advances a
    recurrence in its forward, so anything that replays or reuses a forward makes
    the memory step twice.
    """
    logits = model(**inputs, use_cache=False).logits
    pred = logits[:, n_prompt - 1: n_prompt - 1 + y.shape[0]].float()
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                           y.to(pred.device).reshape(-1))


def write_context(branch: MemoryBranch, z: torch.Tensor, lo: int, hi: int,
                  stride: int, device: str):
    """Phase B: run the memory over the cached context, keeping the graph.

    ``z`` is the frozen host's own output, so nothing upstream of here needs a
    gradient -- but everything the *memory* does (its gates, its key/value maps)
    does, and the graph is ``d_mem``-wide, which is why the whole context fits in
    it instead of a truncation window.
    """
    zc = z[lo:hi:stride].float().to(device).unsqueeze(0)          # [1, n_w, k]
    state, S = branch.init_carry(1)
    state, S = branch.memory.update(zc, state, S,
                                    min(branch.chunk_size, zc.shape[1]))
    return state, S, zc.shape[1]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--z", required=True, nargs="+",
                    help="un ou plusieurs caches produits par titans.graft.cache_z. "
                         "Plusieurs jours sont concatenes en une seule liste de "
                         "fenetres (--make-targets n'en accepte qu'un).")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--span", type=int, default=12,
                    help="clips par fenetre (12 x 30 s = 6 min, la limite du "
                         "professeur)")
    ap.add_argument("--hop", type=int, default=4, help="decalage entre fenetres")
    ap.add_argument("--student-clips", type=int, default=1, dest="student_clips",
                    help="clips que l'eleve VOIT; le reste ne lui parvient que par "
                         "la memoire")
    ap.add_argument("--max-new", type=int, default=48, dest="max_new")
    # --- memoire ---
    add_memory_flags(ap, write_stride=2)
    # --- optimisation ---
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=200, dest="eval_every")
    ap.add_argument("--eval-n", type=int, default=16, dest="eval_n")
    ap.add_argument("--seed", type=int, default=0)
    # --- modes / plomberie ---
    ap.add_argument("--make-targets", action="store_true", dest="make_targets")
    ap.add_argument("--targets", default=None, nargs="+",
                    help="fichiers de cibles, dans le MEME ordre que --z. Omis, "
                         "ils sont deduits en remplacant '_z.pt' par '_tgt.pt'.")
    ap.add_argument("--out-targets", default=None, dest="out_targets")
    ap.add_argument("--out", default=None, help="prefixe des sorties (.json, .pt)")
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    from transformers import AutoProcessor

    from .check_qwen import load_host

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    if a.make_targets and len(a.z) != 1:
        raise SystemExit("--make-targets ne prend qu'un seul --z (une carte par jour)")

    # Un "jour" est un cache z autonome : ses propres clips, ses propres offsets,
    # sa propre resolution. Rien n'est fusionne -- une fenetre porte l'indice de
    # son jour et tout le reste s'indexe dessus.
    days = []
    for path in a.z:
        c = torch.load(path, map_location="cpu", weights_only=False)
        days.append({"src": path, "z": c["z"], "offsets": c["offsets"].tolist(),
                     "paths": c["clips"], "fpc": c["frames_per_clip"],
                     "fps": c["frames_per_clip"] / 30.0,
                     "max_side": c["max_side"]})
        print(f"cache {os.path.basename(path)} : {len(c['clips'])} clips, "
              f"{c['z'].shape[0]} tokens, {c['frames_per_clip']} images/clip",
              flush=True)

    wins = [(d, lo, hi) for d, day in enumerate(days)
            for lo, hi in windows(len(day["paths"]), a.span, a.hop)]
    print(f"{len(wins)} fenetres de {a.span} clips ({a.span * 0.5:.1f} min), "
          f"pas {a.hop}, sur {len(days)} jour(s)")

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)

    # ------------------------------------------------------------ professeur
    if a.make_targets:
        day = days[0]
        paths, fpc, fps = day["paths"], day["fpc"], day["fps"]
        out = a.out_targets or a.z[0].replace("_z.pt", "_tgt.pt")
        tgt, t0 = {}, time.time()
        for i, (_, lo, hi) in enumerate(wins):
            try:
                frames = load_window_frames(paths, lo, hi, fpc, day["max_side"])
                y = teacher_continuation(model, processor, frames, a.probe, fps=fps,
                                         max_new=a.max_new, device=a.device)
                tgt[(lo, hi)] = y
            except Exception as e:
                print(f"  fenetre {lo}-{hi} sautee : {type(e).__name__}: {e}", flush=True)
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{i + 1}/{len(wins)}] {el / (i + 1):.1f} s/fenetre, "
                      f"reste {(len(wins) - i - 1) * el / (i + 1) / 60:.0f} min", flush=True)
        torch.save({"targets": {f"{k[0]}-{k[1]}": v for k, v in tgt.items()},
                    "probe": a.probe, "span": a.span, "hop": a.hop,
                    "max_new": a.max_new, "z": a.z[0]}, out)
        print(f"\n{len(tgt)}/{len(wins)} cibles -> {out}")
        return

    tgt_paths = a.targets or [p.replace("_z.pt", "_tgt.pt") for p in a.z]
    if len(tgt_paths) != len(days):
        raise SystemExit(f"{len(tgt_paths)} fichiers de cibles pour {len(days)} "
                         f"caches z -- ils doivent etre dans le meme ordre")
    tgt = {}
    for d, tp in enumerate(tgt_paths):
        if not os.path.exists(tp):
            raise SystemExit(f"cibles manquantes : {tp} (lancer --make-targets)")
        tb = torch.load(tp, map_location="cpu", weights_only=False)
        # La sonde est verifiee par jour : melanger deux sondes donnerait un
        # entrainement ou la meme fenetre a deux cibles differentes selon le jour.
        if tb["probe"] != a.probe:
            raise SystemExit(f"{tp} : sonde {tb['probe']!r} != --probe")
        if tb.get("span") != a.span or tb.get("hop") != a.hop:
            raise SystemExit(f"{tp} : genere avec span/hop "
                             f"{tb.get('span')}/{tb.get('hop')}, pas {a.span}/{a.hop}")
        for k, v in tb["targets"].items():
            lo, hi = (int(x) for x in k.split("-"))
            tgt[(d, lo, hi)] = v
        print(f"  {os.path.basename(tp)} : {len(tb['targets'])} cibles", flush=True)

    wins = [w for w in wins if w in tgt]
    # Purge la couture : sans elle la premiere fenetre de validation partage
    # span-hop clips avec la derniere d'entrainement (8 sur 12 aux valeurs par
    # defaut), donc le score de validation serait flatte des deux tiers du
    # contexte que la memoire est censee porter.
    embargo = max(0, math.ceil(a.span / a.hop) - 1)
    train_w, val_w = split_windows(wins, embargo=embargo)
    print(f"{len(train_w)} fenetres d'entrainement, {len(val_w)} de validation "
          f"(queue contigue par jour, purge de {embargo} fenetres a la couture), "
          f"{len(tgt)} cibles au total")

    # ------------------------------------------------------------------ greffe
    pca = (torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
           if a.basis else None)
    mem_kw = memory_kwargs_from_args(a)
    d_model = int(model.config.text_config.hidden_size
                  if hasattr(model.config, "text_config") else model.config.hidden_size)
    branch = MemoryBranch(d_model, a.d_mem, chunk_size=a.chunk,
                          write_stride=1, down_pca=pca, **mem_kw).to(a.device)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False            # l'eleve LIT la memoire, il ne l'ecrit pas
    rep = branch_report(branch, sum(p.numel() for p in model.parameters()))
    print(f"greffe couche {site} : {rep['branch_trainable']} parametres entrainables "
          f"({rep['pct_of_host']:.3f} % de l'hote), identite a l'init : "
          f"{branch.is_identity()}")

    params = [p for p in branch.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    def lr_at(step):
        if a.warmup and step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    def one(win, floor: bool = False):
        """Loss of the student on one window. Returns ``(mem, short_or_None, n_writes)``.

        ``floor`` runs the *same* input through the branch-disabled model, i.e. the
        untouched host on the student's clip alone. It is a second full forward, so
        it is only paid where it is read -- at eval, and on the printed steps.
        """
        d, lo, hi = win
        day = days[d]
        y = tgt[win]
        s_lo = hi - a.student_clips
        frames = load_window_frames(day["paths"], s_lo, hi, day["fpc"],
                                    day["max_side"])
        inputs, n_prompt = forced_inputs(processor, frames, a.probe, y, day["fps"],
                                         a.device)
        # contexte = ce que l'eleve NE voit pas, dans les offsets de SON jour
        tok_lo, tok_hi = day["offsets"][lo], day["offsets"][s_lo]
        wrapper.reset()
        state, S, nw = write_context(branch, day["z"], tok_lo, tok_hi,
                                     a.write_stride, a.device)
        wrapper.carry, wrapper.token_offset = (state, S), 0
        loss = forced_loss(model, inputs, n_prompt, y)
        short = None
        if floor:
            with torch.no_grad():
                wrapper.enabled = False
                short = float(forced_loss(model, inputs, n_prompt, y))
                wrapper.enabled = True
        return loss, short, nw

    hist, best = [], float("inf")
    rng = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        win = train_w[int(torch.randint(len(train_w), (1,), generator=rng))]
        show = step % 20 == 0
        try:
            loss, short, nw = one(win, floor=show)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip_grad)
        opt.step()

        if show:
            print(f"  {step:5d}  loss {float(loss):.4f}  sans-memoire {short:.4f}  "
                  f"|g| {float(gn):.3f}  ecritures {nw}  "
                  f"{(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)

        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            branch.eval()
            with torch.no_grad():
                ms, ss = [], []
                for w in val_w[:a.eval_n]:
                    try:
                        l, s, _ = one(w, floor=True)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache(); continue
                    ms.append(float(l)); ss.append(s)
            branch.train()
            if ms:
                m, s = sum(ms) / len(ms), sum(ss) / len(ss)
                rec = (s - m) / s if s else float("nan")
                hist.append({"step": step + 1, "val_mem": m, "val_short": s,
                             "recovered": rec})
                print(f"  == val {step + 1}: memoire {m:.4f} | sans {s:.4f} | "
                      f"recupere {100 * rec:+.1f} %", flush=True)
                if m < best and a.out:
                    best = m
                    torch.save({"branch": branch.state_dict(), "step": step + 1,
                                "val_mem": m, "val_short": s, "args": vars(a)},
                               a.out + ".pt")

    if a.out:
        with open(a.out + ".json", "w") as f:
            json.dump({"args": vars(a), "report": rep, "history": hist,
                       "elapsed_s": time.time() - t0}, f, indent=1)
        print("\necrit:", a.out + ".json", "/", a.out + ".pt")


if __name__ == "__main__":
    main()
