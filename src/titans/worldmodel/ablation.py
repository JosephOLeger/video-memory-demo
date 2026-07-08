"""Long-term-memory ablation experiment (Phase 4).

Goal: measure what the *long-range* neural memory contributes, controlling for
two confounds raised in review:

1. **Parameter count.** We do NOT compare against a smaller model with the memory
   branch removed. Instead the control keeps the entire memory branch (identical
   weights) but resets it at every segment (``reset_memory_each_segment``), so it
   carries no information across segments. Since attention is confined to one
   segment window, the threaded memory is the only long-range pathway -- so
   "threaded" vs "reset" isolates long-range memory at *zero* parameter cost.

2. **A single final loss hides where memory helps.** We report L1 as a function
   of temporal position, not just one scalar: if the memory earns its keep on
   long context, the threaded-vs-control gap should *grow* with position (more
   accumulated history to exploit), not sit flat.

Train/val come from the dataset's separate ``train``/``val`` clip folders (no
video-level leakage); standardisation stats are computed on train and applied to
both (no distribution leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from .model import WorldModel
from .train import load_latents


def _standardize(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                 eps: float = 1e-5) -> torch.Tensor:
    return (z - mean) / (std + eps)


def train_stats(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over ``(videos, time)`` of the train latents."""
    return z.mean(dim=(0, 1), keepdim=True), z.std(dim=(0, 1), keepdim=True)


def train_model(
    train_z: torch.Tensor,
    *,
    reset_control: bool,
    steps: int = 300,
    lr: float = 1e-3,
    seed: int = 0,
    dim: int = 64,
    depth: int = 2,
    segment_size: int = 8,
    **model_kwargs,
) -> WorldModel:
    """Train one WorldModel (threaded or reset-control) on ``train_z``.

    ``reset_control`` selects the parameter-matched control. The seed fixes the
    initial weights, so both variants start from *identical* parameters.
    """
    torch.manual_seed(seed)
    model = WorldModel(
        latent_dim=train_z.shape[-1], dim=dim, depth=depth,
        segment_size=segment_size,
        reset_memory_each_segment=reset_control, **model_kwargs,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = model(train_z)
        loss = F.l1_loss(pred[:, :-1], train_z[:, 1:])
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def per_position_l1(model: WorldModel, val_z: torch.Tensor) -> torch.Tensor:
    """Next-step L1 on ``val_z`` per temporal position -> tensor ``[T-1]``.

    Position ``t`` is the mean absolute error of predicting ``val_z[:, t+1]`` from
    the past, averaged over videos and channels.
    """
    pred = model(val_z)
    err = (pred[:, :-1] - val_z[:, 1:]).abs().mean(dim=(0, 2))  # [T-1]
    return err


def run_ablation(
    train_path: str | Path,
    val_path: str | Path,
    *,
    steps: int = 300,
    seed: int = 0,
    segment_size: int = 8,
    **model_kwargs,
) -> dict:
    """Train threaded vs reset-control, evaluate per-position L1 on val.

    Returns a dict with per-position curves and segment-bucketed means so the
    caller can see whether the memory's advantage grows with temporal position.
    """
    train_raw = load_latents(train_path)
    val_raw = load_latents(val_path)
    mean, std = train_stats(train_raw)
    train_z = _standardize(train_raw, mean, std)
    val_z = _standardize(val_raw, mean, std)

    threaded = train_model(train_z, reset_control=False, steps=steps, seed=seed,
                           segment_size=segment_size, **model_kwargs)
    control = train_model(train_z, reset_control=True, steps=steps, seed=seed,
                          segment_size=segment_size, **model_kwargs)

    l1_threaded = per_position_l1(threaded, val_z)  # [T-1]
    l1_control = per_position_l1(control, val_z)

    # Bucket positions by segment index to read the trend cheaply.
    T1 = l1_threaded.shape[0]
    seg_idx = torch.arange(T1) // segment_size
    buckets = []
    for s in range(int(seg_idx.max()) + 1):
        m = seg_idx == s
        buckets.append((s,
                        l1_control[m].mean().item(),
                        l1_threaded[m].mean().item()))

    return {
        "l1_threaded": l1_threaded,
        "l1_control": l1_control,
        "final_threaded": l1_threaded.mean().item(),
        "final_control": l1_control.mean().item(),
        "by_segment": buckets,  # (segment, control_L1, threaded_L1)
    }


def format_report(res: dict) -> str:
    """Human-readable table: control vs threaded L1 per segment, and the gap."""
    lines = ["seg |  control |  threaded |   gap (ctrl-thr)  |  rel%",
             "----+----------+-----------+-------------------+------"]
    for s, ctrl, thr in res["by_segment"]:
        gap = ctrl - thr
        rel = 100 * gap / ctrl if ctrl else 0.0
        lines.append(f"{s:3d} | {ctrl:8.4f} | {thr:9.4f} | {gap:+.4f}           | {rel:+5.1f}%")
    lines.append("----+----------+-----------+-------------------+------")
    fc, ft = res["final_control"], res["final_threaded"]
    lines.append(f"ALL | {fc:8.4f} | {ft:9.4f} | {fc-ft:+.4f}           | "
                 f"{100*(fc-ft)/fc:+5.1f}%")
    return "\n".join(lines)
