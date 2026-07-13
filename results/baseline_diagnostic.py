"""Baseline-only training diagnostic for the two easy membench cells.

Purpose (see the sweep-3config diagnostic task): the 3-config sweep at steps=1200
left even the *baseline* config well below ceiling on the easiest cells
(MQAR num_pairs=4 -> 0.503, selective_copy num_tokens=4 -> 0.628). Published
MQAR-style benchmarks saturate near ~1.0 at this difficulty, so this script
asks: is 1200 steps simply too early on a still-rising curve (undertraining),
or has training plateaued at a mediocre level (a real bottleneck)?

It trains *baseline only* (memory_use_cascade=False), matching the sweep's model
geometry (dim=32, depth=2, memory_hidden_dim=32, memory_num_layers=2,
segment_size=8, batch=32), and logs held-out eval accuracy + train loss every
`eval_every` steps so we see the whole trajectory, not just a final number.
A small LR sweep is included to check whether the default lr=1e-3 is off.

Not committed as production code -- a diagnostic harness kept alongside its
output (results/baseline_diagnostic.json) as evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.titans.membench.tasks import associative_recall, selective_copy
from src.titans.membench.model import RecallModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The two easy cells, exactly as the sweep configures them.
SEG = 8
BATCH = 32
CELLS = {
    "mqar_np4": lambda s: associative_recall(
        batch=BATCH, num_pairs=4, segment_size=SEG, key_vocab=16, val_vocab=16, seed=s
    ),
    "selcopy_nt4": lambda s: selective_copy(
        batch=BATCH, num_tokens=4, num_slots=16, segment_size=SEG,
        content_vocab=16, seed=s
    ),
}


def _masked(logits, y, mask):
    sel = mask
    loss = F.cross_entropy(logits[sel], y[sel])
    acc = (logits[sel].argmax(-1) == y[sel]).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate(model, task_fn, eval_seeds):
    model.eval()
    accs = []
    for s in eval_seeds:
        b = task_fn(s)
        logits = model(b.x.to(DEVICE))
        _, acc = _masked(logits, b.y.to(DEVICE), b.mask.to(DEVICE))
        accs.append(acc.item())
    model.train()
    t = torch.tensor(accs)
    return t.mean().item(), t.std(unbiased=False).item()


def run_one(task_fn, *, lr, max_steps, eval_every, seed,
            ceil=0.98, ceil_patience=3, plateau_patience=12, min_steps=2000):
    """Train baseline-only, logging eval acc over training. Early stop on
    ceiling (>=ceil for ceil_patience consecutive checks) or plateau (no best
    improvement for plateau_patience checks after min_steps)."""
    torch.manual_seed(seed)
    sample = task_fn(0)
    model = RecallModel(
        sample.vocab_size, dim=32, depth=2, segment_size=SEG, num_heads=1,
        memory_hidden_dim=32, memory_num_layers=2, memory_use_cascade=False,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eval_seeds = range(10_000, 10_020)

    traj = []  # (step, train_loss, eval_acc_mean, eval_acc_std)
    best = 0.0
    best_step = 0
    ceil_hits = 0
    chance = 1.0 / sample.answer_classes

    # eval at step 0 (random init)
    m, sd = evaluate(model, task_fn, eval_seeds)
    traj.append((0, float("nan"), m, sd))

    model.train()
    last_loss = float("nan")
    for step in range(1, max_steps + 1):
        b = task_fn(step)
        logits = model(b.x.to(DEVICE))
        loss, _ = _masked(logits, b.y.to(DEVICE), b.mask.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

        if step % eval_every == 0 or step == max_steps:
            m, sd = evaluate(model, task_fn, eval_seeds)
            traj.append((step, last_loss, m, sd))
            if m > best + 1e-4:
                best, best_step = m, step
            if m >= ceil:
                ceil_hits += 1
                if ceil_hits >= ceil_patience:
                    return dict(traj=traj, stopped="ceiling", best=best,
                                best_step=best_step, chance=chance, lr=lr,
                                final_step=step)
            else:
                ceil_hits = 0
            if step >= min_steps and (step - best_step) >= plateau_patience * eval_every:
                return dict(traj=traj, stopped="plateau", best=best,
                            best_step=best_step, chance=chance, lr=lr,
                            final_step=step)
    return dict(traj=traj, stopped="max_steps", best=best, best_step=best_step,
                chance=chance, lr=lr, final_step=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=str, default="1e-3")
    ap.add_argument("--max-steps", type=int, default=15000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", type=str, default="mqar_np4,selcopy_nt4")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "baseline_diagnostic.json"))
    args = ap.parse_args()

    lrs = [float(x) for x in args.lrs.split(",")]
    cells = args.cells.split(",")
    out = Path(args.out)

    results = dict(
        meta=dict(device=DEVICE, max_steps=args.max_steps, eval_every=args.eval_every,
                  seed=args.seed, batch=BATCH, segment_size=SEG,
                  model="dim32 depth2 memhid32 memlayers2 baseline(no-cascade)"),
        runs=[],
    )
    t0 = time.time()
    for cell in cells:
        task_fn = CELLS[cell]
        for lr in lrs:
            print(f"[{time.time()-t0:7.1f}s] === cell={cell} lr={lr:g} ===", flush=True)
            r = run_one(task_fn, lr=lr, max_steps=args.max_steps,
                        eval_every=args.eval_every, seed=args.seed)
            r["cell"] = cell
            results["runs"].append(r)
            for (st, ls, ac, sd) in r["traj"]:
                print(f"    step {st:6d}  loss {ls:8.4f}  acc {ac:.3f}+-{sd:.3f}", flush=True)
            print(f"    -> stopped={r['stopped']} best={r['best']:.3f}@{r['best_step']} "
                  f"chance={r['chance']:.3f}", flush=True)
            out.write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:7.1f}s] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
