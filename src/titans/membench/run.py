"""Train + accuracy evaluation for the memory benchmark.

Data is synthetic and freshly sampled every step (train) and from unseen seeds
(eval), so accuracy measures a *learned algorithm*, not memorisation. The headline
is retrieval accuracy vs the chance level; :func:`accuracy_by_distance` further
shows whether recall degrades with how far back the answer lives -- the direct
signature of long-term memory.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .model import RecallModel
from .tasks import Batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TaskFn = Callable[[int], Batch]  # seed -> Batch


def _masked(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    """Cross-entropy loss and accuracy over the masked (answer) positions."""
    sel = mask
    loss = F.cross_entropy(logits[sel], y[sel])
    acc = (logits[sel].argmax(-1) == y[sel]).float().mean()
    return loss, acc


def train_recall(
    task_fn: TaskFn,
    *,
    steps: int = 2000,
    lr: float = 1e-3,
    device: str = DEVICE,
    dim: int = 64,
    depth: int = 2,
    segment_size: int = 8,
    num_heads: int = 1,
    max_len: int = 1024,
    eval_seeds=range(10_000_000, 10_000_020),
    log_every: int = 0,
    stop_on_nan: bool = False,
    **block_kwargs,
) -> dict:
    """Train a :class:`RecallModel` on ``task_fn`` and eval accuracy on unseen seeds.

    ``segment_size`` must match the value baked into ``task_fn``. Returns final
    train loss, held-out accuracy (mean/std over ``eval_seeds``), and the chance
    level, plus the trained ``model`` for further probing.

    ``eval_seeds`` must not overlap the training range. Training draws
    ``task_fn(step)`` for ``step`` in ``[0, steps)``, and the old default of
    ``range(10_000, 10_020)`` sat inside that range for any run past 10k steps --
    so those runs trained on their own eval batches, once each. Held out here by
    putting the eval seeds at 1e7, far beyond any step count we use.
    """
    sample = task_fn(0)
    model = RecallModel(sample.vocab_size, dim=dim, depth=depth,
                        segment_size=segment_size, num_heads=num_heads,
                        max_len=max_len, **block_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    diverged_at = None
    for step in range(steps):
        b = task_fn(step)
        logits = model(b.x.to(device))
        loss, _ = _masked(logits, b.y.to(device), b.mask.to(device))
        # ``stop_on_nan`` records *when* the inner loop blew up, and stops. The
        # divergence is abrupt -- healthy at step n, all-NaN at n+1, inside a
        # single forward (README 8.5) -- so the step index is the measurement,
        # and running on afterwards only burns a card producing NaN.
        if stop_on_nan and not torch.isfinite(loss):
            diverged_at = step
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:5d}  train loss {loss.item():.4f}", flush=True)

    model.eval()
    accs = []
    with torch.no_grad():
        for s in eval_seeds:
            b = task_fn(s)
            logits = model(b.x.to(device))
            _, acc = _masked(logits, b.y.to(device), b.mask.to(device))
            accs.append(acc.item())
    accs = torch.tensor(accs)
    return {
        "acc_mean": accs.mean().item(),
        "acc_std": accs.std(unbiased=False).item(),
        "chance": 1.0 / sample.answer_classes,
        "train_loss": loss.item(),
        "diverged_at": diverged_at,
        "model": model,
        "device": device,
    }


def train_recall_converge(
    task_fn: TaskFn,
    *,
    lr: float = 1e-3,
    device: str = DEVICE,
    dim: int = 64,
    depth: int = 2,
    segment_size: int = 8,
    num_heads: int = 1,
    max_len: int = 1024,
    eval_seeds=range(10_000_000, 10_000_020),
    eval_every: int = 500,
    max_steps: int = 8000,
    min_steps: int = 2000,
    ceil: float = 0.98,
    ceil_patience: int = 3,
    plateau_patience: int = 12,
    log_every: int = 0,
    **block_kwargs,
) -> dict:
    """Train a :class:`RecallModel` until held-out accuracy converges.

    Same model geometry / ``block_kwargs`` contract as :func:`train_recall`, but
    the fixed step budget is replaced by the convergence rule from the baseline
    diagnostic (``results/baseline_diagnostic.py``): eval held-out accuracy every
    ``eval_every`` steps and stop early when accuracy is at ``ceil`` for
    ``ceil_patience`` consecutive evals (``stopped="ceiling"``), or has not beaten
    its running best for ``plateau_patience`` evals once past ``min_steps``
    (``stopped="plateau"``); otherwise run to ``max_steps`` (``stopped="max_steps"``).

    The caller is responsible for seeding (``torch.manual_seed``) *before* calling
    so that matched-init comparisons hold; this mirrors :func:`train_recall`. The
    same rule/params must be handed to every config for a fair comparison.

    Returns the same keys as :func:`train_recall` (``acc_mean``/``acc_std`` are the
    *final* held-out eval, at the stopping point) plus ``best``/``best_step``,
    ``stopped``, ``final_step``, and the full ``trajectory`` of
    ``(step, train_loss, eval_acc_mean, eval_acc_std)``.
    """
    sample = task_fn(0)
    model = RecallModel(sample.vocab_size, dim=dim, depth=depth,
                        segment_size=segment_size, num_heads=num_heads,
                        max_len=max_len, **block_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    chance = 1.0 / sample.answer_classes

    def _eval() -> tuple[float, float]:
        model.eval()
        accs = []
        with torch.no_grad():
            for s in eval_seeds:
                b = task_fn(s)
                logits = model(b.x.to(device))
                _, acc = _masked(logits, b.y.to(device), b.mask.to(device))
                accs.append(acc.item())
        model.train()
        t = torch.tensor(accs)
        return t.mean().item(), t.std(unbiased=False).item()

    traj: list[tuple] = []
    m, sd = _eval()  # step 0 (random init)
    traj.append((0, float("nan"), m, sd))

    best, best_step, ceil_hits = 0.0, 0, 0
    last_loss = float("nan")
    stopped, final_step = "max_steps", 0
    model.train()
    for step in range(1, max_steps + 1):
        b = task_fn(step)
        logits = model(b.x.to(device))
        loss, _ = _masked(logits, b.y.to(device), b.mask.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
        final_step = step
        if log_every and (step % log_every == 0):
            print(f"  step {step:5d}  train loss {last_loss:.4f}", flush=True)

        if step % eval_every == 0 or step == max_steps:
            m, sd = _eval()
            traj.append((step, last_loss, m, sd))
            if m > best + 1e-4:
                best, best_step = m, step
            if m >= ceil:
                ceil_hits += 1
                if ceil_hits >= ceil_patience:
                    stopped = "ceiling"
                    break
            else:
                ceil_hits = 0
            if step >= min_steps and (step - best_step) >= plateau_patience * eval_every:
                stopped = "plateau"
                break

    final_m, final_sd = traj[-1][2], traj[-1][3]
    return {
        "acc_mean": final_m,
        "acc_std": final_sd,
        "best": best,
        "best_step": best_step,
        "stopped": stopped,
        "final_step": final_step,
        "chance": chance,
        "train_loss": last_loss,
        "trajectory": [list(t) for t in traj],
        "model": model,
        "device": device,
    }


@torch.no_grad()
def accuracy_by_distance(
    model: RecallModel,
    task_fn: TaskFn,
    *,
    device: str = DEVICE,
    n_batches: int = 40,
    num_bins: int = 6,
    seed_base: int = 50_000,
) -> list[tuple[float, float, int]]:
    """Recall accuracy bucketed by answer distance (single-answer tasks only).

    Returns ``[(distance_mid, accuracy, count), ...]``. A flat curve means the
    memory holds equally over distance; a downward slope means it forgets with
    depth -- the key long-term-memory diagnostic.
    """
    model.eval()
    corrects, dists = [], []
    for i in range(n_batches):
        b = task_fn(seed_base + i)
        if b.distance is None:
            raise ValueError("accuracy_by_distance needs a task with a distance field")
        preds = model(b.x.to(device)).argmax(-1).cpu()
        # exactly one masked answer per row -> per-row correctness
        row_correct = ((preds == b.y) & b.mask).any(dim=1).float()
        corrects.append(row_correct)
        dists.append(b.distance.float())
    corrects = torch.cat(corrects)
    dists = torch.cat(dists)

    lo, hi = dists.min().item(), dists.max().item()
    edges = torch.linspace(lo, hi + 1e-6, num_bins + 1)
    out = []
    for k in range(num_bins):
        m = (dists >= edges[k]) & (dists < edges[k + 1])
        if m.any():
            out.append((((edges[k] + edges[k + 1]) / 2).item(),
                        corrects[m].mean().item(), int(m.sum())))
    return out
