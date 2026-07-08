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
    **model_kwargs,
) -> float:
    """Train one WorldModel (given seed/config) on GPU and return val next-step L1."""
    torch.manual_seed(seed)
    model = WorldModel(latent_dim=train_z.shape[-1], **model_kwargs).to(device)
    tz, vz = train_z.to(device), val_z.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = F.l1_loss(model(tz)[:, :-1], tz[:, 1:])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return F.l1_loss(model(vz)[:, :-1], vz[:, 1:]).item()


def run_config(
    train_path: str | Path,
    val_path: str | Path,
    *,
    seeds: Iterable[int] = range(5),
    steps: int = 400,
    lr: float = 1e-3,
    device: str = DEVICE,
    standardize: bool = True,
    **model_kwargs,
) -> dict:
    """Train one config across ``seeds`` and report mean/std val L1 + baseline.

    Standardisation stats come from train only (applied to both), so there is no
    val leakage. Returns a dict with per-seed L1s, mean/std, the persistence
    baseline, and the config used.
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
                        device=device, **model_kwargs)
        for s in seeds
    ]
    t = torch.tensor(per_seed)
    return {
        "val_l1_mean": t.mean().item(),
        "val_l1_std": t.std(unbiased=False).item(),
        "per_seed": per_seed,
        "baseline_l1": baseline,
        "beats_baseline": t.mean().item() < baseline,
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
    """Table of variant -> val L1 mean±std, vs the (shared) persistence baseline."""
    baseline = next(iter(results.values()))["baseline_l1"]
    dev = next(iter(results.values()))["device"]
    n = next(iter(results.values()))["n_seeds"]
    lines = [
        f"device={dev}  seeds={n}  persistence baseline L1={baseline:.4f}",
        "variant                    |  val L1 (mean±std)  | vs baseline",
        "---------------------------+---------------------+------------",
    ]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["val_l1_mean"]):
        delta = 100 * (baseline - r["val_l1_mean"]) / baseline
        lines.append(
            f"{name:26.26s} | {r['val_l1_mean']:.4f} ± {r['val_l1_std']:.4f}    "
            f"| {delta:+5.1f}%"
        )
    return "\n".join(lines)
