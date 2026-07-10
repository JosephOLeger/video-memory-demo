"""Difficulty sweep: 3 memory-update configs at a matched parameter budget.

Compares three ways the neural memory carries its momentum, all sharing the *same*
memory-MLP geometry (``dim``/``hidden_dim``/``num_layers``) so the parameter
budget is matched (see the honest note below on the one small, disclosed delta):

1. ``baseline``  -- plain scalar momentum ``S_t`` (``use_cascade_momentum=False``).
2. ``uncoupled`` -- cascade enabled but coupling betas forced to 0
   (``cascade_beta1=0.0``): the K reservoirs never diffuse, only reservoir 0 is
   ever populated, and the readout is reservoir 0. NOTE (verified numerically):
   with betas == 0 this recurrence is *bit-identical* to ``baseline`` -- the
   cascade collapses exactly onto the scalar rule -- so this config is included as
   a control that isolates "what does the coupling buy?" and is expected to match
   ``baseline`` cell-for-cell.
3. ``coupled``   -- full data-dependent, non-commuting cascade
   (``use_cascade_momentum=True, cascade_selective=True``): the actual proposed
   architecture.

Parameter budget. Configs 1 and 2 have *identical* parameter counts (the fixed
cascade adds only a non-trained buffer). Config 3 adds exactly one small head,
``cascade.beta_head = Linear(dim, K-1)`` (``dim*(K-1)+(K-1)`` params), because
selective mode needs a per-edge gate. This is a small, disclosed delta -- reported
in the results, not hidden -- and every config keeps ``dim``/``hidden_dim``
identical so it is not a confound.

Tasks swept (all reuse / extend ``membench.tasks``):

* ``mqar``          -- ``associative_recall`` (MQAR-style), swept over ``num_pairs``.
* ``selective_copy``-- MAD selective copying, swept over ``num_tokens``.
* ``fuzzy_recall``  -- MAD fuzzy (near-duplicate) recall, swept over ``num_pairs``.

Run:  ``python -m titans.membench.sweep``            (full small sweep)
      ``python -m titans.membench.sweep --smoke``     (tiny, for the smoke test)
      ``python -m titans.membench.sweep --out FILE``  (results JSON path)
Results are written as JSON (list of per-cell records). No plotting -- raw numbers
only; a follow-up pass can visualise once the numbers exist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from .run import train_recall
from .tasks import associative_recall, fuzzy_recall, selective_copy

# ------------------------------------------------------------------- configs
# Each entry is the extra block_kwargs handed to the MAC block / memory.
CONFIGS: dict[str, dict] = {
    "baseline": dict(memory_use_cascade=False),
    "uncoupled": dict(
        memory_use_cascade=True,
        cascade_reservoirs=4,
        cascade_beta1=0.0,       # zero coupling -> reservoirs never diffuse
        cascade_selective=False,
    ),
    "coupled": dict(
        memory_use_cascade=True,
        cascade_reservoirs=4,
        cascade_beta1=0.5,
        cascade_g=2.0,
        cascade_selective=True,  # data-dependent, non-commuting B_t
    ),
}


# ----------------------------------------------------------------- task grid
def _task_grid(smoke: bool) -> list[tuple[str, str, int, Callable[[int], object], dict]]:
    """Return ``(task, difficulty_label, difficulty_value, task_fn, meta)`` cells.

    ``task_fn`` maps a seed -> Batch (matching ``run.train_recall``'s contract).
    ``meta`` carries fixed generator settings echoed into the results file.
    """
    seg = 8
    cells = []

    if smoke:
        mqar_pairs = [4]
        selcopy_tokens = [4]
        fuzzy_pairs = [4]
        batch = 32
    else:
        # Reduced-but-real grid (see run_sweep docstring): kept small so the whole
        # sweep finishes in ~20 min on CPU. The vmap-over-batch inner gradient of
        # the memory MLP is the cost bottleneck, so batch and step counts are the
        # main levers; harder cells (num_pairs=32) are dropped rather than run at a
        # step budget too small to mean anything.
        mqar_pairs = [4, 8, 16]
        selcopy_tokens = [4, 8]
        fuzzy_pairs = [4, 8]
        batch = 32

    # MQAR (associative recall): harder + longer with more pairs.
    for n in mqar_pairs:
        kv = max(16, n)  # key_vocab >= num_pairs
        meta = dict(num_pairs=n, key_vocab=kv, val_vocab=16, segment_size=seg, batch=batch)
        fn = (lambda n=n, kv=kv, b=batch: (
            lambda s: associative_recall(batch=b, num_pairs=n, segment_size=seg,
                                         key_vocab=kv, val_vocab=16, seed=s)))()
        cells.append(("mqar", "num_pairs", n, fn, meta))

    # MAD selective copying: harder with more marked tokens (slots scale with it).
    for k in selcopy_tokens:
        slots = 4 * k
        meta = dict(num_tokens=k, num_slots=slots, content_vocab=16,
                    segment_size=seg, batch=batch)
        fn = (lambda k=k, slots=slots, b=batch: (
            lambda s: selective_copy(batch=b, num_tokens=k, num_slots=slots,
                                     segment_size=seg, content_vocab=16, seed=s)))()
        cells.append(("selective_copy", "num_tokens", k, fn, meta))

    # MAD fuzzy recall: harder with more pairs (more near-duplicate interference).
    for n in fuzzy_pairs:
        kv = max(32, 2 * n)  # key_vocab >= 2*num_pairs
        meta = dict(num_pairs=n, key_vocab=kv, val_vocab=16, segment_size=seg, batch=batch)
        fn = (lambda n=n, kv=kv, b=batch: (
            lambda s: fuzzy_recall(batch=b, num_pairs=n, segment_size=seg,
                                   key_vocab=kv, val_vocab=16, seed=s)))()
        cells.append(("fuzzy_recall", "num_pairs", n, fn, meta))

    return cells


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def run_sweep(
    *,
    smoke: bool = False,
    steps: int = 200,
    dim: int = 32,
    depth: int = 2,
    memory_hidden_dim: int = 32,
    memory_num_layers: int = 2,
    segment_size: int = 8,
    lr: float = 1e-3,
    eval_seeds=range(10_000, 10_010),
    seed: int = 0,
    log: bool = True,
) -> list[dict]:
    """Train a matched-budget model for every (task, difficulty, config) cell.

    Returns a flat list of records with the final held-out accuracy, chance level,
    parameter count, and train loss per cell. Keeps everything small (CPU-feasible)
    -- see the module docstring for the scientific setup.
    """
    if smoke:
        steps = 3
        dim = 16
        memory_hidden_dim = 16
        eval_seeds = range(10_000, 10_002)

    cells = _task_grid(smoke)
    records: list[dict] = []
    t0 = time.time()
    for task, dlabel, dval, task_fn, meta in cells:
        for cfg_name, cfg_kw in CONFIGS.items():
            torch.manual_seed(seed)  # identical init across configs (matched start)
            res = train_recall(
                task_fn,
                steps=steps,
                lr=lr,
                dim=dim,
                depth=depth,
                segment_size=segment_size,
                memory_hidden_dim=memory_hidden_dim,
                memory_num_layers=memory_num_layers,
                eval_seeds=eval_seeds,
                **cfg_kw,
            )
            rec = dict(
                task=task,
                difficulty_label=dlabel,
                difficulty=dval,
                config=cfg_name,
                acc_mean=res["acc_mean"],
                acc_std=res["acc_std"],
                chance=res["chance"],
                train_loss=res["train_loss"],
                num_params=_count_params(res["model"]),
                steps=steps,
                dim=dim,
                memory_hidden_dim=memory_hidden_dim,
                meta=meta,
            )
            records.append(rec)
            if log:
                print(
                    f"[{time.time()-t0:6.1f}s] {task:14s} {dlabel}={dval:<3d} "
                    f"{cfg_name:9s} acc={rec['acc_mean']:.3f} "
                    f"(chance {rec['chance']:.3f}) params={rec['num_params']}",
                    flush=True,
                )
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-config matched-budget memory sweep")
    ap.add_argument("--smoke", action="store_true", help="tiny sweep (harness test)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=str, default=None, help="results JSON path")
    args = ap.parse_args(argv)

    records = run_sweep(smoke=args.smoke, steps=args.steps)

    out = args.out
    if out is None:
        results_dir = Path(__file__).resolve().parents[3] / "results"
        results_dir.mkdir(exist_ok=True)
        tag = "smoke" if args.smoke else "sweep"
        out = results_dir / f"{tag}_results.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        configs={k: v for k, v in CONFIGS.items()},
        records=records,
    )
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(records)} records -> {out}")
    return out


if __name__ == "__main__":
    main()
