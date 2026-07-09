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
    eval_seeds=range(10_000, 10_020),
    log_every: int = 0,
    **block_kwargs,
) -> dict:
    """Train a :class:`RecallModel` on ``task_fn`` and eval accuracy on unseen seeds.

    ``segment_size`` must match the value baked into ``task_fn``. Returns final
    train loss, held-out accuracy (mean/std over ``eval_seeds``), and the chance
    level, plus the trained ``model`` for further probing.
    """
    sample = task_fn(0)
    model = RecallModel(sample.vocab_size, dim=dim, depth=depth,
                        segment_size=segment_size, num_heads=num_heads,
                        max_len=max_len, **block_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for step in range(steps):
        b = task_fn(step)
        logits = model(b.x.to(device))
        loss, _ = _masked(logits, b.y.to(device), b.mask.to(device))
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
