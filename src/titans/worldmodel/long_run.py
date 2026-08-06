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
shows up in a finiteness check (see README on the MQAR theta ladder). So each
repetition records a *functional* metric (next-step L1, and the echo advantage over
repetition 1) alongside the state diagnostics ``pr`` / ``cos_ret`` / ``||w1||`` /
``||S||``. A memory that has collapsed to a constant function of the query shows
``cos_ret -> 1`` and ``pr -> 1`` long before anything goes non-finite.

    python -m titans.worldmodel.long_run --ckpt ckpt/full_nocas_s0.pt \\
        --val cache/egolife_full_pca128_val.pt --reps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .atlas import CONFIGS
from .model import WorldModel
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
def run_stream(model, clips, reset_each: bool = False) -> list[dict]:
    """Forward the clips in order, threading the memory carry, one record per clip.

    ``reset_each`` drops the carry between clips: the memory-off control, which says
    what the same model predicts with no history at all. The gap between the two is
    the memory's actual contribution at that point in the stream.
    """
    carry = None
    out = []
    for r, z in enumerate(clips):
        if reset_each:
            carry = None
        pred, carry = model(z, carry=carry, return_carry=True)
        # position t predicts t+1, so the last position's target is in the next clip;
        # dropping it costs one step per repetition and keeps every clip comparable.
        l1 = float((pred[:, :-1] - z[:, 1:]).abs().mean())
        rec = {"rep": r, "l1": l1, "finite": bool(torch.isfinite(pred).all())}
        rec.update(state_diagnostics(model, carry))
        out.append(rec)
        if not rec["finite"] or not all(v["finite"] for k, v in rec.items()
                                        if isinstance(v, dict)):
            rec["note"] = "non-finite; stream stopped"
            break
    return out


def build_model(ckpt: str, dim: int, seg: int, cfg: dict, device: str):
    model = WorldModel(latent_dim=dim, dim=dim, depth=2, segment_size=seg, **cfg)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model = model.to(device).eval()
    for p in model.parameters():          # explicit: nothing outside the memory moves
        p.requires_grad_(False)
    return model, blob.get("step")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True, help="PCA-reduced val cache")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="loop,cycle,shuffle")
    ap.add_argument("--pool", type=int, default=4, help="clips in the cycle pool")
    ap.add_argument("--eta-init", type=float, default=None, dest="eta_init")
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    ap.add_argument("--leak", default="none")
    ap.add_argument("--leak-rate", type=float, default=1.0, dest="leak_rate")
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--gate", default="surprise")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    va, meta = load_latent_cache(a.val)
    if "pca" not in meta:
        raise SystemExit(f"{a.val} is not a PCA cache; the model runs in latent space.")
    dim = va.shape[-1]

    cfg = dict(CONFIGS["sane_ns"])
    if a.eta_init is not None:
        cfg["eta_init"] = a.eta_init
    if a.cascade:
        cfg.update(memory_use_cascade=True, cascade_reservoirs=a.reservoirs,
                   cascade_leak=a.leak, cascade_gate_signal=a.gate,
                   cascade_leak_rate=a.leak_rate)
        if a.readout != "none":
            cfg["cascade_readout"] = a.readout
        if a.selective:
            cfg["cascade_selective"] = True

    t0 = time.time()
    model, step = build_model(a.ckpt, dim, a.seg, cfg, a.device)
    va = va.to(a.device)

    streams = {}
    for mode in a.modes.split(","):
        clips = build_stream(va, a.reps, mode, seed=a.seed, pool=a.pool)
        streams[mode] = run_stream(model, clips)
        # Memory-off control for EVERY mode, on the identical clip order. Comparing
        # loop against shuffle would confound the memory's contribution with "clip 7
        # happens to be easier than clip 312"; comparing each stream against itself
        # with the carry dropped does not.
        streams[mode + "_memoff"] = run_stream(model, clips, reset_each=True)

    def summary(tr):
        fin = [r for r in tr if r["finite"]]
        if not fin:
            return {}
        first, last = fin[0], fin[-1]
        tail = fin[max(0, len(fin) - 10):]
        return {
            "reps_finite": len(fin), "reps_total": len(tr),
            "l1_rep0": first["l1"], "l1_rep1": fin[1]["l1"] if len(fin) > 1 else None,
            "l1_tail": sum(r["l1"] for r in tail) / len(tail),
            "l1_max": max(r["l1"] for r in fin),
            "pr_rep0": first["b0"]["pr"], "pr_tail": tail[-1]["b0"]["pr"],
            "cos_ret_rep0": first["b0"]["cos_ret"], "cos_ret_tail": tail[-1]["b0"]["cos_ret"],
            "w1_rep0": first["b0"]["w1_norm"], "w1_tail": tail[-1]["b0"]["w1_norm"],
            "S_rep0": first["b0"]["S_norm"], "S_tail": tail[-1]["b0"]["S_norm"],
            "all_finite": len(fin) == len(tr),
        }

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
        "label": a.label or os.path.basename(a.ckpt),
        "ckpt": os.path.basename(a.ckpt), "train_step": step,
        "cascade": a.cascade, "leak": a.leak if a.cascade else None,
        "leak_rate": a.leak_rate if a.cascade else None,
        "eta_init": cfg.get("eta_init"), "theta_scale": cfg.get("theta_scale"),
        "reps": a.reps, "T": va.shape[1], "dim": dim, "seg": a.seg,
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
