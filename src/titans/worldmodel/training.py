"""Training-loop infrastructure for runs long enough to be worth protecting.

Two mechanisms, both absent before and both needed to scale:

**Checkpointing.** Nothing in this repo ever saved model state. At ~1.5 h per run
that was tolerable; a multi-hour run that dies loses everything, and this project's
memory diverges to NaN on ~13% of seeds, so dying is not hypothetical. Writes are
**atomic** (temp file + rename): a process killed mid-write leaves the previous
checkpoint intact rather than a truncated one. RNG state is saved too, so a resumed
run draws the same batches it would have drawn.

**Truncated BPTT.** The memory scan retains every step for autograd, so VRAM grows
**linearly in sequence length** -- measured at ``dim=128, B=4``: 3.19 GiB at
``T=128``, 6.36 at 256, ~12.7 at 512. For a *long-term memory* model that is the
binding constraint on the axis that matters most. ``WorldModel.forward`` has
accepted a ``carry`` since Phase 4 and it was covered by a test, but no training
loop ever used it -- it was dead code. :func:`tbptt_step` processes a long sequence
in windows, threading the memory carry across them while **detaching** the graph
between windows, so VRAM is bounded by the window instead of by ``T``.

The tradeoff is explicit: gradients no longer flow across window boundaries, so the
outer loss cannot credit the memory for information carried further back than one
window. The memory *state* still crosses (that is the point); only the gradient
path is cut. With ``window >= T`` the function reduces exactly to a plain
forward/backward, which is what :func:`tbptt_step`'s test asserts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------- checkpointing
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer],
    step: int,
    **meta,
) -> None:
    """Atomically write ``model``/``opt``/RNG state and ``step`` to ``path``.

    Atomicity matters more than speed here: the failure mode being guarded against
    is the process dying, and a half-written checkpoint is worse than none because
    it destroys the previous one. Write to ``<path>.tmp``, fsync, then rename --
    rename is atomic within a filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "meta": meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    *,
    restore_rng: bool = True,
) -> tuple[int, dict]:
    """Restore state from ``path``; returns ``(step, meta)``.

    ``step`` is the number of steps already completed, so a caller resumes at
    ``range(step, total)``. Restoring the RNG makes the resumed run draw the same
    batches the uninterrupted one would have -- without it a resume silently
    becomes a different experiment.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    if opt is not None and blob.get("opt") is not None:
        opt.load_state_dict(blob["opt"])
    if restore_rng:
        torch.set_rng_state(blob["rng_cpu"].to(torch.uint8).cpu())
        if blob.get("rng_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(blob["rng_cuda"])
            except (RuntimeError, ValueError):
                # a checkpoint written on a different GPU count: the model and
                # optimiser are still valid, only exact batch order is lost
                pass
    return int(blob["step"]), dict(blob.get("meta") or {})


def maybe_resume(path: str | Path | None, model, opt) -> int:
    """Resume from ``path`` if it exists, else start at step 0."""
    if path and Path(path).exists():
        step, _ = load_checkpoint(path, model, opt)
        return step
    return 0


# ------------------------------------------------------------- truncated BPTT
def _detach_carry(carry):
    """Cut the autograd graph at a window boundary, keeping the memory *state*."""
    out = []
    for state, S in carry:
        ds = None if state is None else {k: v.detach() for k, v in state.items()}
        dS = None if S is None else {k: v.detach() for k, v in S.items()}
        out.append((ds, dS))
    return out


def tbptt_step(
    model,
    z: torch.Tensor,
    *,
    horizon: int = 1,
    window: Optional[int] = None,
) -> torch.Tensor:
    """One optimisation step's worth of loss over a long sequence, in windows.

    ``z`` is ``[B, T, d]``; the objective is the ``horizon``-step-ahead L1, i.e.
    position ``t`` predicts ``z[:, t + horizon]``. ``window`` is how much sequence
    is held in the graph at once (must be a multiple of the blocks' segment size,
    which ``WorldModel`` enforces); ``None`` means the whole sequence, which makes
    this exactly a plain forward/backward.

    Calls ``backward()`` per window and returns the total loss as a detached
    scalar. Windows are weighted by how many positions they score, so the gradient
    equals the full-sequence mean-L1 gradient *within* each window -- what differs
    from full BPTT is only that no gradient crosses a boundary.

    The caller owns ``opt.zero_grad()`` before and ``opt.step()`` after.
    """
    B, T, _ = z.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if T <= horizon:
        raise ValueError(f"sequence length {T} must exceed horizon {horizon}")
    w = T if window is None else window
    if w <= horizon:
        raise ValueError(f"window {w} must exceed horizon {horizon}")

    # Positions 0 .. T-horizon-1 are scorable; weight each window by its share so
    # the summed gradient matches a single mean over all of them.
    total_scored = T - horizon
    carry = None
    total = 0.0
    for start in range(0, T, w):
        end = min(start + w, T)
        chunk = z[:, start:end]
        if chunk.shape[1] <= horizon:
            break                       # tail too short to score anything
        pred, carry = model(chunk, carry=carry, return_carry=True)
        tgt = chunk[:, horizon:]
        n = tgt.shape[1] * tgt.shape[0] * tgt.shape[2]
        loss = F.l1_loss(pred[:, :-horizon], tgt) * (tgt.shape[1] / total_scored)
        loss.backward()
        total += float(loss.detach())
        carry = _detach_carry(carry)
    return torch.tensor(total)


def train_loop(
    model,
    z: torch.Tensor,
    *,
    steps: int,
    opt: torch.optim.Optimizer,
    batch_size: Optional[int] = None,
    horizon: int = 1,
    window: Optional[int] = None,
    transform=None,
    ckpt_path: str | Path | None = None,
    ckpt_every: int = 0,
    log_every: int = 0,
    **ckpt_meta,
) -> dict:
    """Train ``model`` on ``z`` with optional truncated BPTT and checkpointing.

    Resumes from ``ckpt_path`` when it exists, so re-running an interrupted job is
    the same command. Returns ``{"steps", "last_loss", "resumed_from"}``.

    ``transform`` is applied to each mini-batch *after* indexing. The echo
    benchmark uses it to build ``[V;V]``: doubling per batch rather than doubling
    ``z`` up front keeps the full doubled tensor out of VRAM (196 MB saved on the
    EgoLife k=128 cache).
    """
    B = z.shape[0]
    bs = batch_size or B
    start = maybe_resume(ckpt_path, model, opt)
    if start >= steps:
        return {"steps": start, "last_loss": float("nan"), "resumed_from": start}
    last = float("nan")
    model.train()
    for step in range(start, steps):
        idx = torch.randperm(B, device=z.device)[:bs] if bs < B else slice(None)
        batch = z[idx]
        if transform is not None:
            batch = transform(batch)
        opt.zero_grad()
        last = float(tbptt_step(model, batch, horizon=horizon, window=window))
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:6d}/{steps}  L1 {last:.4f}", flush=True)
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_path, model, opt, step + 1, last_loss=last, **ckpt_meta)
    if ckpt_path:
        save_checkpoint(ckpt_path, model, opt, steps, last_loss=last, **ckpt_meta)
    return {"steps": steps, "last_loss": last, "resumed_from": start}
