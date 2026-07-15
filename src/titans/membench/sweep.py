"""Difficulty sweep: 3 memory-update configs at a matched parameter budget.

Compares three ways the neural memory carries its momentum, all sharing the *same*
memory-MLP geometry (``dim``/``hidden_dim``/``num_layers``) so the parameter
budget is matched (see the honest note below on the one small, disclosed delta):

1. ``baseline``  -- plain scalar momentum ``S_t`` (``use_cascade_momentum=False``).
2. ``uncoupled`` -- the genuine null hypothesis: ``K`` **independent** exponential
   momentum filters with **no** coupling between them
   (``cascade_topology="independent"``). Every reservoir receives the same injected
   surprise ``-theta_t grad`` directly, each decays at its own fixed geometric rate
   (:func:`~titans.metaplastic.geometric_decays`), and the readout is the *mean*
   over the K filters (mean, not sum, so the effective inner learning rate stays
   matched to ``baseline`` -- see ``MetaplasticCascade.readout``). This is the
   diagonal-``B`` degenerate case (``B = diag(decays)``), so it tests "would any
   K-way mixture of exponential filters do as well as the cascade?" -- isolating
   what the *coupling* specifically buys. It is NOT bit-identical to ``baseline``
   (it mixes K distinct decay timescales), unlike the earlier flawed
   ``cascade_beta1=0.0`` config, which degenerated to 1 live filter + K-1 dead
   reservoirs and *was* bit-identical to ``baseline``.
3. ``coupled``   -- full data-dependent, non-commuting cascade
   (``use_cascade_momentum=True, cascade_selective=True``): the actual proposed
   architecture (tridiagonal, gradient into reservoir 0, diffusion between depths).

Parameter budget. Configs 1 and 2 have *identical* parameter counts (the
independent filter bank adds only non-trained buffers). Config 3 adds exactly one
small head,
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
from typing import Callable, Optional

import torch

from .run import train_recall, train_recall_converge
from .tasks import associative_recall, fuzzy_recall, selective_copy

# Per-task-family safety cap on the convergence budget (see the diagnostic:
# ``results/baseline_diagnostic.py``). MQAR converges fastest (~6k steps at lr=1e-3
# for the easy cell) so 10k leaves headroom for the harder num_pairs; selective_copy
# needed ~11.5k for baseline and fuzzy_recall carries extra interference, so both get
# 15k. These are ceilings, not targets -- the plateau/ceiling rule normally stops
# well before them; a cell that hits its cap is reported as ``stopped="max_steps"``
# (possibly still improving) rather than silently truncated.
CONVERGE_MAX_STEPS: dict[str, int] = {
    "mqar": 10_000,
    "selective_copy": 15_000,
    "fuzzy_recall": 15_000,
}

# ------------------------------------------------------------------- configs
# Each entry is the extra block_kwargs handed to the MAC block / memory.
CONFIGS: dict[str, dict] = {
    "baseline": dict(memory_use_cascade=False),
    "uncoupled": dict(
        memory_use_cascade=True,
        cascade_reservoirs=4,
        cascade_beta1=0.5,          # decays = 1 - 0.5*2^-k = [0.5,0.75,0.875,0.9375]
        cascade_g=2.0,
        cascade_topology="independent",  # diagonal B: K uncoupled exp. filters
        cascade_selective=False,
    ),
    "coupled": dict(
        memory_use_cascade=True,
        cascade_reservoirs=4,
        cascade_beta1=0.5,
        cascade_g=2.0,
        cascade_selective=True,  # data-dependent, non-commuting B_t (chain)
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
        # Focused-but-real grid. The previous wide/low-step grid left even the
        # baseline near chance everywhere (under-training, not an architecture
        # signal), so this version reallocates the wall-clock budget toward *more
        # steps on fewer, lower-difficulty cells*: the immediate priority is a
        # sanity check that the baseline config demonstrably learns the task before
        # any 3-way comparison is meaningful. The vmap-over-batch inner gradient of
        # the memory MLP is the cost bottleneck; steps (see ``run_sweep`` default
        # 1200) are the main lever. Harder cells are dropped rather than run at a
        # step budget too small to mean anything. 4 cells x 3 configs.
        mqar_pairs = [4, 8]
        selcopy_tokens = [4]
        fuzzy_pairs = [4]
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


def _rec_key(r: dict) -> tuple:
    return (r["task"], r["difficulty"], r["config"])


def _load_records(path: Optional[Path]) -> list[dict]:
    """Read the ``records`` list from an existing checkpoint file (``[]`` if none)."""
    if path is None or not Path(path).exists():
        return []
    try:
        return json.loads(Path(path).read_text()).get("records", [])
    except (json.JSONDecodeError, OSError):
        return []


def _merge_write(path: Path, records: list[dict]) -> None:
    """Merge ``records`` into any existing checkpoint (keyed by task/difficulty/config)
    and write back, so independent invocations sharing one ``--out`` accumulate
    instead of clobbering each other. Idempotent: re-writing the same cell replaces it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {_rec_key(r): r for r in _load_records(path)}
    for r in records:
        merged[_rec_key(r)] = r
    payload = dict(configs={k: v for k, v in CONFIGS.items()}, records=list(merged.values()))
    path.write_text(json.dumps(payload, indent=2))


def run_sweep(
    *,
    smoke: bool = False,
    steps: int = 1200,
    dim: int = 32,
    depth: int = 2,
    memory_hidden_dim: int = 32,
    memory_num_layers: int = 2,
    segment_size: int = 8,
    lr: float = 1e-3,
    eval_seeds=range(10_000, 10_010),
    seed: int = 0,
    log: bool = True,
    tasks: Optional[set] = None,
    configs: Optional[set] = None,
    checkpoint_path: Optional[Path] = None,
    converge: bool = False,
    eval_every: int = 500,
    min_steps: int = 2000,
    ceil: float = 0.98,
    ceil_patience: int = 3,
    plateau_patience: int = 12,
    max_steps: Optional[dict] = None,
    resume: bool = False,
) -> list[dict]:
    """Train a matched-budget model for every (task, difficulty, config) cell.

    Returns a flat list of records with the held-out accuracy, chance level,
    parameter count, and train loss per cell -- see the module docstring for the
    scientific setup.

    Two training modes:

    * ``converge=False`` (default): fixed ``steps`` budget (the original behaviour,
      used by the smoke test).
    * ``converge=True``: train each cell to a plateau/ceiling using the diagnostic's
      rule (:func:`~titans.membench.run.train_recall_converge`), the *same* rule and
      params applied identically to every config so the comparison stays fair. The
      per-task-family step cap comes from ``CONVERGE_MAX_STEPS`` unless overridden via
      ``max_steps`` (a ``{task: cap}`` dict). ``eval_every``/``min_steps``/``ceil``/
      ``ceil_patience``/``plateau_patience`` are the shared stopping-rule knobs.

    ``tasks`` / ``configs``, if given, restrict the grid to those task names /
    config names -- lets a long sweep be split into wall-clock-bounded chunks.
    ``checkpoint_path``, if given, merges the accumulated records into that path
    after *every* cell (see :func:`_merge_write`), so a killed/timed-out run doesn't
    lose already-completed cells. ``resume=True`` additionally skips any cell already
    present in ``checkpoint_path`` -- so re-launching after a kill only trains the
    missing cells.
    """
    if smoke:
        steps = 3
        dim = 16
        memory_hidden_dim = 16
        eval_seeds = range(10_000, 10_002)
        if converge:
            eval_every = 3
            min_steps = 0
            plateau_patience = 1

    caps = dict(CONVERGE_MAX_STEPS)
    if max_steps:
        caps.update(max_steps)
    if smoke and converge:
        caps = {k: 6 for k in caps}

    cells = _task_grid(smoke)
    if tasks is not None:
        cells = [c for c in cells if c[0] in tasks]
    cfg_names = [c for c in CONFIGS if configs is None or c in configs]

    done = {_rec_key(r) for r in _load_records(checkpoint_path)} if resume else set()

    records: list[dict] = []
    t0 = time.time()
    for task, dlabel, dval, task_fn, meta in cells:
        cap = caps.get(task, 10_000)
        for cfg_name in cfg_names:
            cfg_kw = CONFIGS[cfg_name]
            if (task, dval, cfg_name) in done:
                if log:
                    print(f"[{time.time()-t0:6.1f}s] skip (already done) "
                          f"{task} {dlabel}={dval} {cfg_name}", flush=True)
                continue
            torch.manual_seed(seed)  # identical init across configs (matched start)
            if converge:
                res = train_recall_converge(
                    task_fn, lr=lr, dim=dim, depth=depth, segment_size=segment_size,
                    memory_hidden_dim=memory_hidden_dim,
                    memory_num_layers=memory_num_layers, eval_seeds=eval_seeds,
                    eval_every=eval_every, max_steps=cap, min_steps=min_steps,
                    ceil=ceil, ceil_patience=ceil_patience,
                    plateau_patience=plateau_patience, **cfg_kw,
                )
                cell_steps = res["final_step"]
            else:
                res = train_recall(
                    task_fn, steps=steps, lr=lr, dim=dim, depth=depth,
                    segment_size=segment_size, memory_hidden_dim=memory_hidden_dim,
                    memory_num_layers=memory_num_layers, eval_seeds=eval_seeds,
                    **cfg_kw,
                )
                cell_steps = steps
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
                steps=cell_steps,
                dim=dim,
                memory_hidden_dim=memory_hidden_dim,
                meta=meta,
            )
            if converge:
                rec.update(
                    converge=True,
                    lr=lr,
                    stopped=res["stopped"],
                    best_acc=res["best"],
                    best_step=res["best_step"],
                    converge_steps=res["final_step"],
                    max_steps_cap=cap,
                    trajectory=res["trajectory"],
                )
            records.append(rec)
            if log:
                extra = (f" stopped={rec['stopped']}@{rec['converge_steps']} "
                         f"best={rec['best_acc']:.3f}") if converge else ""
                print(
                    f"[{time.time()-t0:6.1f}s] {task:14s} {dlabel}={dval:<3d} "
                    f"{cfg_name:9s} acc={rec['acc_mean']:.3f} "
                    f"(chance {rec['chance']:.3f}) params={rec['num_params']}{extra}",
                    flush=True,
                )
            if checkpoint_path is not None:
                _merge_write(checkpoint_path, records)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-config matched-budget memory sweep")
    ap.add_argument("--smoke", action="store_true", help="tiny sweep (harness test)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", type=str, default=None, help="results JSON path")
    ap.add_argument(
        "--tasks", type=str, default=None,
        help="comma-separated task names to restrict the grid to (e.g. 'mqar'), "
             "for splitting a long sweep into wall-clock-bounded chunks",
    )
    ap.add_argument(
        "--configs", type=str, default=None,
        help="comma-separated config names to restrict to (e.g. 'coupled'), for "
             "splitting a long converged sweep into per-config chunks",
    )
    ap.add_argument(
        "--converge", action="store_true",
        help="train each cell to a plateau/ceiling (diagnostic rule) instead of a "
             "fixed --steps budget; the same rule is applied to every config",
    )
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--min-steps", type=int, default=2000)
    ap.add_argument("--ceil", type=float, default=0.98)
    ap.add_argument("--ceil-patience", type=int, default=3)
    ap.add_argument("--plateau-patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--checkpoint", action="store_true",
        help="merge-write --out after every cell (not just at the end), so a killed "
             "run keeps its completed cells",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="skip cells already present in --out (re-launch after a kill only trains "
             "the missing cells); implies reading the existing --out",
    )
    args = ap.parse_args(argv)

    out = args.out
    if out is None:
        results_dir = Path(__file__).resolve().parents[3] / "results"
        results_dir.mkdir(exist_ok=True)
        tag = "smoke" if args.smoke else "sweep"
        out = results_dir / f"{tag}_results.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks = set(args.tasks.split(",")) if args.tasks else None
    configs = set(args.configs.split(",")) if args.configs else None
    # For a converged run we always want incremental persistence + resume safety.
    checkpoint = args.checkpoint or args.resume or args.converge
    records = run_sweep(
        smoke=args.smoke, steps=args.steps, lr=args.lr, tasks=tasks, configs=configs,
        checkpoint_path=out if checkpoint else None,
        converge=args.converge, eval_every=args.eval_every, min_steps=args.min_steps,
        ceil=args.ceil, ceil_patience=args.ceil_patience,
        plateau_patience=args.plateau_patience, resume=args.resume,
    )

    # Merge (not clobber): preserves cells written by other chunked invocations.
    _merge_write(out, records)
    print(f"\nwrote/merged {len(records)} records -> {out}")
    return out


if __name__ == "__main__":
    main()
