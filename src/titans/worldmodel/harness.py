"""Experiment harness: GPU training, multi-seed, persistence baseline.

Built to make architecture-variant comparisons *fast* and *statistically honest*:

* **GPU** -- the WorldModel (not just the encoder) trains on CUDA when available,
  so many seeds/variants are cheap.
* **Multi-seed** -- every config is trained from several seeds and reported as
  ``mean ± std`` val L1; a single run is not trustworthy at this scale.
* **Persistence baseline** -- ``z[t+1] ~= z[t]`` gives a reference L1, so a
  variant's number means something (does it even beat copying the last latent?).

Latents are precomputed once and cached (encoding is the slow part); every
experiment here just loads a cache, so iterating on variants is fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .ablation import train_stats
from .model import WorldModel
from .train import load_latents

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _standardize(z, mean, std, eps=1e-5):
    return (z - mean) / (std + eps)


def persistence_baseline_l1(val_z: torch.Tensor) -> float:
    """Next-step L1 of the trivial predictor ``z_hat[t+1] = z[t]``."""
    return (val_z[:, 1:] - val_z[:, :-1]).abs().mean().item()


def train_eval_once(
    train_z: torch.Tensor,
    val_z: torch.Tensor,
    *,
    seed: int,
    steps: int = 400,
    lr: float = 1e-3,
    device: str = DEVICE,
    batch_size: int | None = None,
    **model_kwargs,
) -> dict:
    """Train one WorldModel (given seed/config) and return train & val next-step L1.

    ``batch_size`` mini-batches the training videos each step (default: full
    batch). Mini-batching both bounds VRAM (needed for larger ``dim``) and adds
    SGD noise that helps generalisation in the data-limited regime. Returns
    ``{"train_l1", "val_l1"}`` -- the gap between them tells overfitting (data-
    limited) from underfitting (architecture/optimisation).
    """
    torch.manual_seed(seed)
    model = WorldModel(latent_dim=train_z.shape[-1], **model_kwargs).to(device)
    tz, vz = train_z.to(device), val_z.to(device)
    B = tz.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        if batch_size is not None and batch_size < B:
            idx = torch.randperm(B, device=device)[:batch_size]
            batch = tz[idx]
        else:
            batch = tz
        opt.zero_grad()
        loss = F.l1_loss(model(batch)[:, :-1], batch[:, 1:])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():  # no_grad forward is light -> full-set eval fits even at large dim
        train_l1 = F.l1_loss(model(tz)[:, :-1], tz[:, 1:]).item()
        val_l1 = F.l1_loss(model(vz)[:, :-1], vz[:, 1:]).item()
    return {"train_l1": train_l1, "val_l1": val_l1}


def run_config(
    train_path: str | Path,
    val_path: str | Path,
    *,
    seeds: Iterable[int] = range(5),
    steps: int = 400,
    lr: float = 1e-3,
    device: str = DEVICE,
    standardize: bool = True,
    batch_size: int | None = None,
    **model_kwargs,
) -> dict:
    """Train one config across ``seeds`` and report mean/std train & val L1.

    Standardisation stats come from train only (applied to both), so there is no
    val leakage. Returns per-seed train/val L1s, their mean/std, the persistence
    baseline, and the config used. ``val_mean - train_mean`` is the
    generalisation gap.
    """
    train_raw, val_raw = load_latents(train_path), load_latents(val_path)
    if standardize:
        mean, std = train_stats(train_raw)
        train_z, val_z = _standardize(train_raw, mean, std), _standardize(val_raw, mean, std)
    else:
        train_z, val_z = train_raw, val_raw

    baseline = persistence_baseline_l1(val_z)
    per_seed = [
        train_eval_once(train_z, val_z, seed=s, steps=steps, lr=lr,
                        device=device, batch_size=batch_size, **model_kwargs)
        for s in seeds
    ]
    tr = torch.tensor([r["train_l1"] for r in per_seed])
    va = torch.tensor([r["val_l1"] for r in per_seed])
    return {
        "val_l1_mean": va.mean().item(),
        "val_l1_std": va.std(unbiased=False).item(),
        "train_l1_mean": tr.mean().item(),
        "train_l1_std": tr.std(unbiased=False).item(),
        "gen_gap": va.mean().item() - tr.mean().item(),
        "per_seed": per_seed,
        "baseline_l1": baseline,
        "beats_baseline": va.mean().item() < baseline,
        "config": model_kwargs,
        "device": device,
        "n_seeds": len(per_seed),
    }


def compare_variants(
    train_path: str | Path,
    val_path: str | Path,
    variants: dict[str, dict],
    *,
    seeds: Iterable[int] = range(5),
    steps: int = 400,
    lr: float = 1e-3,
    device: str = DEVICE,
) -> dict:
    """Run several named configs and return their results keyed by name."""
    seeds = list(seeds)
    return {
        name: run_config(train_path, val_path, seeds=seeds, steps=steps, lr=lr,
                         device=device, **cfg)
        for name, cfg in variants.items()
    }


def format_comparison(results: dict) -> str:
    """Table of variant -> train & val L1, gen gap, vs the persistence baseline."""
    baseline = next(iter(results.values()))["baseline_l1"]
    dev = next(iter(results.values()))["device"]
    n = next(iter(results.values()))["n_seeds"]
    lines = [
        f"device={dev}  seeds={n}  persistence baseline (val) L1={baseline:.4f}",
        "variant              |  train L1  |   val L1 (mean±std)  | gap  | vs base",
        "---------------------+------------+----------------------+------+--------",
    ]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["val_l1_mean"]):
        delta = 100 * (baseline - r["val_l1_mean"]) / baseline
        beat = " *" if r["beats_baseline"] else ""
        lines.append(
            f"{name:20.20s} | {r['train_l1_mean']:.4f}    | "
            f"{r['val_l1_mean']:.4f} ± {r['val_l1_std']:.4f}    | "
            f"{r['gen_gap']:+.3f}| {delta:+5.1f}%{beat}"
        )
    lines.append("(* beats baseline; gap = val - train, large gap => data-limited/overfit)")
    return "\n".join(lines)
