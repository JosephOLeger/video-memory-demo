"""Named world-model configurations, from plain Titans to the full Atlas stack.

The point of this module is that a run is reproducible from a *name*. Every
architecture flag this project has accumulated is opt-in and defaults off, which
is right for keeping old results comparable but makes an experiment's actual
configuration easy to get wrong -- four flags were found silently inert exactly
because they were only ever set ad hoc. Configs live here, are diffed against
each other by construction (each stage adds to the previous), and are printed
with every run.

Ladder (each row adds one ATLAS ingredient to the row above):

    titans        the paper's original memory: scalar momentum, identity feature
                  map, window c=1, first-order inner update
    phi           + polynomial feature map (ATLAS §3.1, capacity)
    table7        + the paper's memory MLP recipe (expansion 4, GELU, residual)
    omega         + sliding-window objective, c=4 (ATLAS Eq 9)
    atlas         + Muon/Newton-Schulz inner optimiser -> Eq 57-58 complete
    atlas_cascade + the metaplastic reservoir momentum (this project's own idea,
                  not from either paper)

Measured cost on an RTX A5000 at ``dim=128, depth=2, T=64, batch=8``:
``titans`` 0.380 s/step at 20% GPU utilisation (launch-bound -- its batch can go
to 64 for free), ``atlas_cascade`` 1.546 s/step at 91% and 13.5 GB. Newton-Schulz
is what costs: ~3.6x time and ~4x memory, because it runs five matmul iterations
per token per weight tensor with autograd retaining every intermediate. Batch 16
OOMs at 23 GB; batch 8 is the practical ceiling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .harness import compare_variants, format_comparison, run_config

# --- the ingredients, so each config below reads as a delta ------------------
PHI = dict(feature_degree=2)
TABLE7 = dict(memory_expansion=4, memory_activation="gelu", memory_residual=True)
OMEGA = dict(memory_window_size=4)
MUON = dict(memory_orthogonalize_momentum=True, memory_update_bias=False)
# Same Newton-Schulz, the two other magnitude conventions. "unit" is ATLAS Eq 32
# taken literally (theta multiplies the orthogonalised momentum, whose singular
# values are ~1, so the per-step move of M is bounded by theta whatever the
# gradient did); "clip" keeps theta on the gradient and only caps the tail.
MUON_UNIT = dict(memory_orthogonalize_momentum="unit", memory_update_bias=False)
MUON_CLIP = dict(memory_orthogonalize_momentum="clip", memory_update_bias=False)
# clip's hard corner, smoothed: bounded by the constant c, monotone in the surprise
# everywhere. c is a cap on theta_t*||S||, so pick it from a trace, not by feel.
MUON_SOFTCLIP = dict(memory_orthogonalize_momentum="softclip",
                     memory_orthogonalize_clip=8.0, memory_update_bias=False)
CASCADE = dict(memory_use_cascade=True, cascade_leak="none", cascade_readout="mean")
# The gate/normalisation set that follows ATLAS's own bounds (p25) rather than
# patching around them. Each piece is inert or harmful alone; together they took
# out-of-attention-window recall from 0.23 to 0.99 on the synthetic recall benchmark
# this repo used to carry (retired 2026-08; the numbers below are that provenance, not
# something you can re-run here). SANE is the base of every arm we still train.
SANE = dict(alpha_init=0.002,          # forget-gate half-life 1 token -> ~346
            theta_scale=0.03,          # inner step under the 2/lambda_max bound
            normalize_qk="l2",         # ||k|| = 1 -> lambda_max ~ m, bound ~0.25
            normalize_output=False,    # the output norm empties the inner loss
            normalize_retrieval=True)  # stands in for the paper's chunk-end norm

CONFIGS: dict[str, dict] = {
    "titans": {},
    "phi": {**PHI},
    "table7": {**PHI, **TABLE7},
    "omega": {**PHI, **TABLE7, **OMEGA},
    "atlas": {**PHI, **TABLE7, **OMEGA, **MUON},
    "atlas_cascade": {**PHI, **TABLE7, **OMEGA, **MUON, **CASCADE},
    # SANE alone. Kept for the ladder; every run we launch uses sane_ns below.
    "sane": {**SANE},
    "sane_ns": {**SANE, **MUON, "learned_qk_scale": True},
    "sane_ns_cascade": {**SANE, **MUON, "learned_qk_scale": True, **CASCADE},
    # The bound ATLAS has and we threw away: our newton_schulz5 restores ||G||,
    # so orthogonalisation was directional only and bounded nothing. Measured
    # 2026-08-17 on the graft, the divergence is a curvature feedback -- ||h_0||
    # runs away and drags the effective step with it -- which is exactly what a
    # gradient-independent bound on ||M_t - M_{t-1}|| would cut. Deltas on sane_ns.
    "sane_ns_unit": {**SANE, **MUON_UNIT, "learned_qk_scale": True},
    "sane_ns_clip": {**SANE, **MUON_CLIP, "learned_qk_scale": True},
    "sane_ns_softclip": {**SANE, **MUON_SOFTCLIP, "learned_qk_scale": True},
    # SANE with the inner step halved. normalize_output=False makes the inner
    # loop diverge to NaN on ~13% of seeds -- abruptly, inside a single forward, with
    # no precursor, and deterministically per seed. theta_scale=0.01 prevented that on
    # the synthetic benchmark without restoring the output norm, which is what SANE
    # deliberately removes. On PCA video latents the divergence has never been seen.
    # Note theta already sat ~10x INSIDE ATLAS's 2/lambda_max bound when it
    # diverged, so this margin is empirical, not the paper's condition.
    "sane_ns_theta01": {**SANE, **MUON, "learned_qk_scale": True, "theta_scale": 0.01},
    # The theory note's fix for the same divergence, from the other direction.
    # theta_scale=0.01 shrinks the step; this bounds the curvature the step is
    # multiplied by -- lambda_max(H_GN^(W1)) = 2||h_0||^2, which normalize_qk does
    # NOT bound because h_0 is computed from online-updated weights. Unlike
    # normalize_output it leaves M(k)'s scale free, so the associative loss keeps
    # its signal. theta_scale=0.01 was measured to cost memory_gain (p=0.015);
    # whether this one is free is the point of measuring it.
    "sane_ns_nh": {**SANE, **MUON, "learned_qk_scale": True, "normalize_hidden": True},
    # Union of both tasks' winners: the full ATLAS stack, the cascade, and the
    # gate/normalisation set. Note SANE removes the output RMSNorm, which killed
    # one seed in four on the synthetic benchmark -- on continuous regression that
    # is more exposed, not less, so this arm may simply diverge. That is itself
    # worth measuring.
    # SANE minus normalize_output=False: on continuous video latents, dropping
    # the output norm on top of phi+cascade sends ||M|| to inf at step 0 (NaN,
    # measured). Keep the output norm here; alpha_init + L2 keys + theta_scale +
    # retrieval-norm are what carry over. Documented deviation from the literal
    # "everything" reading, forced by the task.
    "atlas_cascade_sane": {**PHI, **TABLE7, **OMEGA, **MUON, **CASCADE,
                           **{k: v for k, v in SANE.items() if k != "normalize_output"},
                           "learned_qk_scale": True},
}

# Newton-Schulz dominates VRAM, so configs that use it need the smaller batch.
# Kept explicit rather than inferred: a silently different batch would make the
# comparison meaningless.
SAFE_BATCH = {name: (8 if cfg.get("memory_orthogonalize_momentum") else 8)
              for name, cfg in CONFIGS.items()}


def describe(name: str) -> str:
    """One-line description of what a named config turns on."""
    cfg = CONFIGS[name]
    return f"{name}: " + (", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) or "(paper default)")


def run(
    train_path: str | Path,
    val_path: str | Path,
    names: list[str],
    *,
    seeds=range(2),
    epochs: float = 5,
    batch_size: int = 8,
    dim: int = 128,
    depth: int = 2,
    segment_size: int = 8,
    lr: float = 1e-3,
    out: str | Path | None = None,
    log_every: int = 0,
    health_every: int = 0,
) -> dict:
    """Train the named configs on a latent cache and compare them.

    Batch size is deliberately *shared* across configs: it changes the
    optimisation, so an unmatched batch would confound the architecture
    comparison even though ``titans`` alone would happily run at batch 64.
    """
    variants = {
        n: dict(dim=dim, depth=depth, segment_size=segment_size, **CONFIGS[n])
        for n in names
    }
    for n in names:
        print(f"  {describe(n)}", flush=True)
    t0 = time.time()
    results = compare_variants(
        train_path, val_path, variants, seeds=list(seeds), epochs=epochs, lr=lr,
        batch_size=batch_size, eval_batch_size=batch_size, log_every=log_every,
        health_every=health_every,
    )
    for r in results.values():
        r["elapsed_s"] = time.time() - t0
    if out:
        Path(out).write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="path to the train latent cache (.pt)")
    ap.add_argument("--val", required=True, help="path to the val latent cache (.pt)")
    ap.add_argument("--configs", default="titans,atlas_cascade",
                    help="comma-separated names from CONFIGS")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--health-every", type=int, default=0,
                    help="probe memory rank/magnitude/query-dependence every N steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = [n.strip() for n in args.configs.split(",")]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; available: {sorted(CONFIGS)}")

    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"dim={args.dim} depth={args.depth} batch={args.batch_size} "
          f"epochs={args.epochs} seeds={args.seeds}", flush=True)
    results = run(
        args.train, args.val, names,
        seeds=[int(s) for s in args.seeds.split(",")],
        epochs=args.epochs, batch_size=args.batch_size, dim=args.dim,
        depth=args.depth, lr=args.lr, out=args.out, log_every=args.log_every,
        health_every=args.health_every,
    )
    print()
    print(format_comparison(results), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
