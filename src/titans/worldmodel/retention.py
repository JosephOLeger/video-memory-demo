"""How far apart can the two halves of the echo be before the memory loses them?

``[V;V]`` asks whether the memory retains ``V`` across ``T`` latents. That is the
right *kind* of question -- episodic recall of specific content, not statistical
predictability -- but it fixes the distance at one block. The claim the project
actually makes is "the model can use distant events **without widening the
context**", and that is a claim about a *distance*. So sweep it::

    repeat :  [ V ; D_1 ; ... ; D_n ; V ]
    cross  :  [ U ; D_1 ; ... ; D_n ; V ]

Only the **first** block differs; the distractors and the scored final block are
identical between the two. The gap is therefore attributable to what the memory kept
of ``V`` across ``n*T`` intervening latents, not to one arm being handed easier
content::

    retention_gain(n) = l1(cross) - l1(repeat)      > 0  =>  V is still held

At ``n = 0`` this reduces exactly to the existing ``memory_gain``, so the curve
extends the benchmark rather than replacing it.

**The attention window never moves.** ``segment_size`` is fixed across every arm, so
attention sees the same 8 latents at every ``n`` and only the memory can carry
anything across the gap. A longer context window is the trivial competing baseline;
holding it fixed is what makes the measurement about memory.

Evaluation only -- no training. It runs on the frozen artifacts a sweep already
produced.

    python -m titans.worldmodel.retention --frozen 'frozen_vv/*_s0.pt' \\
        --val cache/egolife_pca128_val.pt --out retention.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re

import torch

from .frozen import load_frozen
from .harness import memory_health
from .train import load_latent_cache

# A seed that reaches a distance without going NaN is not thereby a seed whose
# memory still works: the recurring failure here is the state collapsing to a
# near-rank-1, near-constant map, which stays perfectly finite. Counting those as
# survivors is what let "positive out to 2048 latents" be reported. Both criteria
# are relative to the *same artifact at gap 0*, so they do not assume a healthy
# value that depends on the memory's width.
COLLAPSE_PR_FRAC = 0.25    # participation ratio fallen to a quarter of its gap-0 value
COLLAPSE_COS = 0.95        # retrievals for distinct queries nearly parallel
# Absolute floor, as a fraction of the memory width (min dim of W1). The
# relative rule alone is blind to an artifact TRAINED into collapse: its
# degenerate gap-0 pr becomes the reference and the whole walk is tagged
# healthy. Measured basis (campaign 2026-08-28, width 128): collapsed unit
# seeds sit at pr 1.5-13, healthy arms at 20-70 -- 0.10 x width separates
# them (one borderline seed at 13.4 documented in the predictions file).
# Applies from gap 0 itself, so a collapsed artifact stops the reach walk
# immediately. Rows without ``pr_width`` (older results) skip this rule.
COLLAPSE_PR_ABS_FRAC = 0.10


def classify(curve: list[dict]) -> list[dict]:
    """Tag each gap ``sain`` / ``collapse`` / ``diverge``, in place, and return it.

    ``diverge`` dominates ``collapse``: a non-finite state is not a collapsed one.
    """
    ref = None
    for c in curve:
        pr, cos = c.get("pr"), c.get("cos_ret")
        width = c.get("pr_width")
        finite = c.get("health_finite", True) and all(
            isinstance(v, float) and v == v for v in (c["l1_repeat"], c["l1_cross"]))
        if ref is None and pr is not None and pr == pr:
            ref = pr
        if not finite:
            c["status"] = "diverge"
        elif pr is not None and (
                (ref and pr < COLLAPSE_PR_FRAC * ref)
                or (width and pr < COLLAPSE_PR_ABS_FRAC * width)
                or (cos is not None and cos > COLLAPSE_COS)):
            c["status"] = "collapse"
        else:
            c["status"] = "sain"
    return curve


def spaced(v: torch.Tensor, first: torch.Tensor, distractors) -> torch.Tensor:
    """``[first ; D_1 ; ... ; D_n ; v]`` along time."""
    return torch.cat([first, *distractors, v], dim=1)


@torch.no_grad()
def last_block_l1(model, seq: torch.Tensor, T: int, K: int, reset: bool,
                  batch: int = 8, margin: int = 4) -> float:
    """K-step-ahead L1 over the **final** ``T``-block only.

    ``margin`` skips the first positions of the block, where the prediction is
    dominated by the segment boundary rather than by memory -- same convention as
    ``echo.per_copy_l1_K``, so ``n = 0`` here is comparable to ``second`` there.
    """
    model.eval()
    for b in model.blocks:
        b.reset_memory_each_segment = reset
    tot = num = 0.0
    N = seq.shape[1]
    lo, hi = N - T + margin, N - K
    try:
        for i in range(0, seq.shape[0], batch):
            z = seq[i:i + batch]
            pred = model(z)
            err = (pred[:, :-K] - z[:, K:]).abs().mean(-1)
            e = err[:, lo:hi]
            tot += e.sum().item()
            num += e.numel()
    finally:
        for b in model.blocks:
            b.reset_memory_each_segment = False
    return tot / num


@torch.no_grad()
def retention_curve(model, va: torch.Tensor, T: int, K: int, gaps,
                    batch: int = 8, margin: int = 4, health_seqs: int = 4) -> list[dict]:
    """``retention_gain`` at each gap, plus the memory-off floor and the state's health.

    Distractors come from ``torch.roll`` at distinct offsets, so every row of the
    batch gets different intervening videos and none of them is that row's ``V`` or
    ``U``. They are built once per gap and shared by both conditions.

    ``mem_effect = l1_repeat_memoff - l1_repeat`` is the headline: it says whether
    carrying a memory across this stream is worth anything at all, where
    ``retention_gain`` only says whether a trace of ``V`` specifically survived --
    both conditions can be degraded and the difference still positive.

    The health triple (``pr``, ``norm``, ``cos_ret``) is measured on the *same*
    spaced sequence, so it reports the state the scored forward actually ran with.
    It costs one extra block-0 pass over ``health_seqs`` sequences per gap, against
    four full-model passes over ``n_eval``.
    """
    out = []
    w1_0 = model.blocks[0].memory.init_memory(1)[0]["w1"]
    pr_width = int(min(w1_0.shape[-2:]))
    for n in gaps:
        if n + 2 > va.shape[0]:
            continue
        distr = [torch.roll(va, k, 0) for k in range(2, n + 2)]
        rep = spaced(va, va, distr)                       # [V ; D... ; V]
        crs = spaced(va, torch.roll(va, 1, 0), distr)     # [U ; D... ; V]
        l_rep = last_block_l1(model, rep, T, K, False, batch, margin)
        l_crs = last_block_l1(model, crs, T, K, False, batch, margin)
        # memory reset each segment: nothing can cross the gap, so the gain must
        # collapse to ~0. It is the floor the curve has to be read against.
        l_rep_off = last_block_l1(model, rep, T, K, True, batch, margin)
        l_crs_off = last_block_l1(model, crs, T, K, True, batch, margin)
        try:
            h = memory_health(model, rep[:health_seqs])
        except Exception:                    # a NaN state can trip svdvals
            h = {"pr": float("nan"), "norm": float("nan"),
                 "cos_ret": float("nan"), "finite": False}
        out.append({
            "gap_blocks": int(n), "gap_latents": int(n * T),
            "seq_len": int((n + 2) * T),
            "l1_repeat": l_rep, "l1_cross": l_crs,
            "retention_gain": l_crs - l_rep,
            "l1_repeat_memoff": l_rep_off, "l1_cross_memoff": l_crs_off,
            "gain_memoff": l_crs_off - l_rep_off,
            "mem_effect": l_rep_off - l_rep,
            "pr": h["pr"], "w1_norm": h["norm"], "cos_ret": h["cos_ret"],
            "pr_width": pr_width,
            "health_finite": bool(h["finite"]),
        })
    return classify(out)


def arm_of(path: str) -> str:
    return re.sub(r"_s\d+\.pt$", "", os.path.basename(path)).replace(".pt", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", required=True, help="artifact, or a glob over several")
    ap.add_argument("--val", required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=96, dest="n_eval")
    ap.add_argument("--gaps", default="0,1,2,4,8,16,32")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha-override", type=float, default=None,
                    dest="alpha_override",
                    help="pin the forget gate to this constant at read time, "
                         "bypassing the learned alpha_head. The reach law "
                         "alpha* ~ 1/(2d) is about a constant test-time alpha, "
                         "and training erases alpha_init (all inits end at "
                         "realised ~1e-4), so the law is probed by overriding "
                         "alpha on frozen artifacts, not by initialising it.")
    ap.add_argument("--theta-override", type=float, default=None,
                    dest="theta_override",
                    help="pin the write step to this constant at read time. "
                         "0.0 disables writing entirely: if mem_effect survives "
                         "it, the memory's measured benefit does not come from "
                         "test-time writes -- the discriminating probe for "
                         "artefacts whose margin audit reads theta_t ~ 1e-6 "
                         "while mem_effect stays positive.")
    a = ap.parse_args()

    va, meta = load_latent_cache(a.val)
    if "pca" not in meta:
        raise SystemExit(f"{a.val} is not a PCA cache.")
    T = va.shape[1]
    va = va[:a.n_eval].to(a.device)
    gaps = [int(x) for x in a.gaps.split(",")]

    rows = []
    for p in sorted(globmod.glob(a.frozen)) or [a.frozen]:
        try:
            model, info = load_frozen(p, a.device)
            if a.alpha_override is not None:
                for b in model.blocks:
                    b.memory.alpha_override = a.alpha_override
            if a.theta_override is not None:
                for b in model.blocks:
                    b.memory.theta_override = a.theta_override
            curve = retention_curve(model, va, T, a.horizon, gaps, a.batch)
            rec = {"arm": arm_of(p), "file": os.path.basename(p),
                   "cascade": bool(info["cfg"].get("memory_use_cascade")),
                   "leak": info["cfg"].get("cascade_leak"),
                   "leak_rate": info["cfg"].get("cascade_leak_rate"),
                   "eta_init": info["cfg"].get("eta_init"),
                   "alpha_override": a.alpha_override,
                   "theta_override": a.theta_override,
                   "T": T, "horizon": a.horizon, "n_eval": int(va.shape[0]),
                   "curve": curve}
        except Exception as e:
            rec = {"arm": arm_of(p), "file": os.path.basename(p),
                   "error": repr(e)[:200]}
        rows.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "curve"}), flush=True)

    ok = [r for r in rows if "curve" in r]
    if ok:
        gaps_seen = [c["gap_latents"] for c in ok[0]["curve"]]
        print()
        print("mem_effect = l1(memoire coupee) - l1(repeat) ; > 0 => porter la memoire "
              "vaut mieux que ne pas en avoir")
        print("statut : . sain / c collapse (pr ou cos_ret) / X diverge")
        print("%-18s" % "bras" + "".join("%11s" % f"+{g}" for g in gaps_seen))
        print("-" * (18 + 11 * len(gaps_seen)))
        for r in sorted(ok, key=lambda r: (not r["cascade"], r["arm"])):
            line = "%-18s" % r["arm"]
            for c in r["curve"]:
                tag = {"sain": ".", "collapse": "c", "diverge": "X"}[c["status"]]
                line += "%10.4f%s" % (c.get("mem_effect", float("nan")), tag)
            print(line)
        print()
        print("retention_gain = l1(cross) - l1(repeat) ; > 0 => une trace de V survit")
        print("%-18s" % "bras" + "".join("%11s" % f"+{g}" for g in gaps_seen))
        print("-" * (18 + 11 * len(gaps_seen)))
        for r in sorted(ok, key=lambda r: (not r["cascade"], r["arm"])):
            print("%-18s" % r["arm"] + "".join(
                "%11.4f" % c["retention_gain"] for c in r["curve"]))
        print()
        print("%-18s" % "(memoire coupee)" + "".join(
            "%10.4f" % c["gain_memoff"] for c in ok[0]["curve"]))
        print("colonnes = latents intercales entre les deux presentations de V")

    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
