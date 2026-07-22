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
CASCADE = dict(memory_use_cascade=True, cascade_leak="none", cascade_readout="mean")
# The gate/normalisation set that follows ATLAS's own bounds (p25) rather than
# patching around them. Each piece is inert or harmful alone; together they take
# MQAR's out-of-attention-window memory from 0.23 to 0.99 (see README).
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
    # Best configuration found on MQAR. Bimodal there: ~50% of seeds reach
    # memory ~0.99, the rest sit at the Titans baseline ~0.27, and ~12% die.
    # Never yet run on video -- the world model's winner is atlas_cascade, and
    # the two have no ingredient in common beyond Muon.
    "sane_ns": {**SANE, **MUON, "learned_qk_scale": True},
    "sane_ns_cascade": {**SANE, **MUON, "learned_qk_scale": True, **CASCADE},
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
    )
    print()
    print(format_comparison(results), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
