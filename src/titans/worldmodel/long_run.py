"""Is the memory stable when the sequence never ends?

Every result in this project comes from sequences of 64 to 144 latents. The scalar
momentum has a fixed point at that length and the cascade at ``cascade_leak="none"``
is a pure accumulator (``rho(B) = 1``, mass-conserving), so a linear ramp and a fixed
point are indistinguishable over 144 steps. This module removes the length limit.

Everything outside the memory is frozen -- the model is in ``eval()`` under
``no_grad``, so the *only* thing evolving is the memory carry ``(state, S)``,
threaded across repetitions via ``WorldModel(carry=...)``. That is the dynamical
system we want to observe, isolated from the outer optimiser. Forward-only also
means no activation graph, so length costs nothing: 500 repetitions of a 144-frame
clip is 72000 steps at the memory footprint of one.

Three streams, because stability alone is not the question:

``loop``     one held-out clip repeated R times. A working memory should predict
             repetition 2 better than repetition 1 and hold that advantage. This is
             the [V;V] echo test taken to its limit.
``cycle``    a pool of R_p random clips round-robin. Repetition still exists, but the
             memory has to hold several contents at once and keep them apart.
``shuffle``  R *independent* random clips. Nothing to re-use, so the echo advantage
             must be zero -- what is left is purely the drift of the state.

Each stream is paired with a **memory-off control on the identical clip order** (the
carry dropped between clips). The gain is read against that, never against another
stream: "loop is better than shuffle" would confound the memory's contribution with
one clip happening to be easier than another.

**Both a divergence and a collapse must be caught.** A run whose loss goes NaN and a
run whose memory quietly stops carrying information both fail, and only the first
shows up in a finiteness check. So each
repetition records a *functional* metric (next-step L1, and the echo advantage over
repetition 1) alongside the state diagnostics ``pr`` / ``cos_ret`` / ``||w1||`` /
``||S||``. A memory that has collapsed to a constant function of the query shows
``cos_ret -> 1`` and ``pr -> 1`` long before anything goes non-finite.

    python -m titans.worldmodel.frozen --ckpt ckpt/full_nocas_s0.pt \\
        --out frozen/nocas.pt --dim 128
    python -m titans.worldmodel.long_run --frozen frozen/nocas.pt \\
        --val cache/egolife_full_pca128_val.pt --reps 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .frozen import load_frozen
from .train import load_latent_cache


@torch.no_grad()
def state_diagnostics(model, carry, n_probe: int = 8) -> dict:
    """Rank / magnitude / query-dependence of every block's memory, from the carry.

    Same quantities as :func:`~titans.worldmodel.harness.memory_health`, but read off
    a carry that already exists instead of re-running a fresh segment loop -- here the
    state *is* the thing under test and re-deriving it would discard the history.
    """
    out = {}
    for i, (blk, (state, S)) in enumerate(zip(model.blocks, carry)):
        mem = blk.memory
        w = state["w1"][0].double()
        finite = bool(torch.isfinite(w).all())
        if finite:
            sv = torch.linalg.svdvals(w)
            pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
            g = torch.Generator(device=w.device).manual_seed(7)
            q = torch.randn(n_probe, model.dim, device=state["w1"].device, generator=g)
            B = state["w1"].shape[0]
            r = mem.retrieve(state, q.unsqueeze(0).expand(B, -1, -1))[0].double()
            rn = r / (r.norm(dim=-1, keepdim=True) + 1e-9)
            c = rn @ rn.T
            off = c[~torch.eye(c.shape[0], dtype=bool, device=c.device)]
            cos_ret = float(off.mean())
        else:
            pr, cos_ret = float("nan"), float("nan")
        s_norm = math.sqrt(sum(float(t.double().pow(2).sum()) for t in S.values()))
        out[f"b{i}"] = {
            "pr": pr, "cos_ret": cos_ret, "w1_norm": float(w.norm()),
            "S_norm": s_norm, "finite": finite,
        }
    return out


def build_stream(va: torch.Tensor, reps: int, mode: str, seed: int = 0, pool: int = 4):
    """A list of ``reps`` clips of shape ``[1, T, dim]``, fed one after another.

    A list rather than one concatenated tensor: the stream is processed clip by clip
    so the per-repetition metrics fall out naturally, and a 72000-step tensor never
    has to exist.

    ``loop``     one clip, ``reps`` times. Maximal repetition.
    ``cycle``    ``pool`` random clips round-robin. Repetition still exists but the
                 memory must hold several contents at once and keep them apart --
                 the regime a real deployment is in.
    ``shuffle``  ``reps`` independent random clips. No repetition at all, so any echo
                 advantage must be zero and only the state's drift is left.
    """
    g = torch.Generator().manual_seed(seed)
    if mode == "loop":
        idx = [int(torch.randint(va.shape[0], (1,), generator=g))] * reps
    elif mode == "cycle":
        base = torch.randint(va.shape[0], (min(pool, va.shape[0]),), generator=g).tolist()
        idx = [base[r % len(base)] for r in range(reps)]
    elif mode == "shuffle":
        idx = torch.randint(va.shape[0], (reps,), generator=g).tolist()
    else:
        raise ValueError(f"mode must be 'loop', 'cycle' or 'shuffle', got {mode!r}")
    return [va[i:i + 1] for i in idx]


@torch.no_grad()
def run_stream(model, clips, reset_each: bool = False,
               diag_every: int = 1) -> list[dict]:
    """Forward the clips in order, threading the memory carry, one record per clip.

    ``reset_each`` drops the carry between clips: the memory-off control, which says
    what the same model predicts with no history at all. The gap between the two is
    the memory's actual contribution at that point in the stream.

    ``diag_every`` subsamples the *state* diagnostics only -- the L1 is recorded every
    repetition regardless. At 5000 repetitions the per-repetition SVD is the dominant
    cost and the state moves far too slowly for consecutive samples to differ; the
    last repetition and any non-finite one are always diagnosed.
    """
    carry = None
    out = []
    n = len(clips)
    for r, z in enumerate(clips):
        if reset_each:
            carry = None
        pred, carry = model(z, carry=carry, return_carry=True)
        # position t predicts t+1, so the last position's target is in the next clip;
        # dropping it costs one step per repetition and keeps every clip comparable.
        l1 = float((pred[:, :-1] - z[:, 1:]).abs().mean())
        finite = bool(torch.isfinite(pred).all())
        rec = {"rep": r, "l1": l1, "finite": finite}
        if not finite or r % diag_every == 0 or r == n - 1:
            rec.update(state_diagnostics(model, carry))
        out.append(rec)
        if not finite or any(isinstance(v, dict) and not v["finite"]
                             for v in rec.values()):
            rec["note"] = "non-finite; stream stopped"
            break
    return out


@torch.no_grad()
def memoff_baseline(model, clips, diag_every: int = 1) -> list[dict]:
    """The memory-off control, computed once per *distinct* clip.

    With the carry dropped, repetition ``r`` is a function of clip ``r`` alone -- so
    ``loop_memoff`` repeats one identical number 5000 times and ``cycle_memoff``
    repeats ``pool`` of them. Computing each distinct clip once and mapping back is
    exact, not an approximation, and it is what makes 5000 repetitions affordable.
    """
    seen: dict[int, dict] = {}
    out = []
    for r, z in enumerate(clips):
        key = int(z.data_ptr())
        if key not in seen:
            rec = run_stream(model, [z], reset_each=True, diag_every=diag_every)[0]
            seen[key] = rec
        rec = dict(seen[key])
        rec["rep"] = r
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", required=True,
                    help="frozen artifact from titans.worldmodel.frozen")
    ap.add_argument("--val", required=True, help="PCA-reduced val cache")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="loop,cycle,shuffle")
    ap.add_argument("--pool", type=int, default=4, help="clips in the cycle pool")
    ap.add_argument("--diag-every", type=int, default=10, dest="diag_every",
                    help="record state diagnostics every N reps (L1 is always recorded)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    va, meta = load_latent_cache(a.val)
    if "pca" not in meta:
        raise SystemExit(f"{a.val} is not a PCA cache; the model runs in latent space.")
    dim = va.shape[-1]

    t0 = time.time()
    # The architecture comes from the artifact, not from flags: an inference probe
    # that lets you mis-specify the model is a probe that can silently measure a
    # different one (see frozen.py).
    model, info = load_frozen(a.frozen, a.device)
    cfg, arch = info["cfg"], info["arch"]
    if arch["dim"] != dim:
        raise SystemExit(f"frozen model dim {arch['dim']} != cache dim {dim}")
    va = va.to(a.device)

    streams = {}
    for mode in a.modes.split(","):
        clips = build_stream(va, a.reps, mode, seed=a.seed, pool=a.pool)
        streams[mode] = run_stream(model, clips, diag_every=a.diag_every)
        # Memory-off control for EVERY mode, on the identical clip order. Comparing
        # loop against shuffle would confound the memory's contribution with "clip 7
        # happens to be easier than clip 312"; comparing each stream against itself
        # with the carry dropped does not.
        streams[mode + "_memoff"] = memoff_baseline(model, clips,
                                                    diag_every=a.diag_every)

    def summary(tr):
        fin = [r for r in tr if r["finite"]]
        if not fin:
            return {}
        diag = [r for r in fin if "b0" in r]
        tail = fin[max(0, len(fin) - 50):]
        d0, dN = (diag[0], diag[-1]) if diag else ({}, {})

        def q(rec, k):
            return rec.get("b0", {}).get(k) if rec else None

        out = {
            "reps_finite": len(fin), "reps_total": len(tr),
            "l1_rep0": fin[0]["l1"], "l1_rep1": fin[1]["l1"] if len(fin) > 1 else None,
            "l1_tail": sum(r["l1"] for r in tail) / len(tail),
            "l1_max": max(r["l1"] for r in fin),
            "all_finite": len(fin) == len(tr),
        }
        # State trajectory, first diagnosed rep -> last. The four quantities that
        # separate "still working" from "quietly dead": effective rank, magnitude,
        # query-dependence of the retrieval, and the momentum's own norm.
        for k in ("pr", "cos_ret", "w1_norm", "S_norm"):
            out[k + "_first"], out[k + "_last"] = q(d0, k), q(dN, k)
        return out

    summ = {k: summary(v) for k, v in streams.items()}
    # The headline number: how much the threaded memory is worth at the END of the
    # stream, on the same content. Positive = the memory still helps; ~0 = it has
    # stopped carrying anything, whether or not it went non-finite.
    for mode in a.modes.split(","):
        on, off = summ.get(mode), summ.get(mode + "_memoff")
        if on and off:
            on["memory_gain_tail"] = off["l1_tail"] - on["l1_tail"]
            on["memory_gain_rep0"] = off["l1_rep0"] - on["l1_rep0"]

    out = {
        "label": a.label or info["meta"].get("label") or os.path.basename(a.frozen),
        "frozen": os.path.basename(a.frozen), "meta": info["meta"],
        "cascade": bool(cfg.get("memory_use_cascade")),
        "leak": cfg.get("cascade_leak"), "leak_rate": cfg.get("cascade_leak_rate"),
        "eta_init": cfg.get("eta_init"), "theta_scale": cfg.get("theta_scale"),
        "reps": a.reps, "T": va.shape[1], "dim": dim, "seg": arch["segment_size"],
        "pool": a.pool, "diag_every": a.diag_every,
        "tokens_per_stream": a.reps * va.shape[1],
        "summary": summ,
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps(out), flush=True)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({**out, "streams": streams}, f)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
