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

from .bottleneck import pca_reduce_splits
from .model import WorldModel
from .train import load_latent_cache, load_latents, train_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _standardize(z, mean, std, eps=1e-5):
    return (z - mean) / (std + eps)


def persistence_baseline_l1(val_z: torch.Tensor) -> float:
    """Next-step L1 of the trivial predictor ``z_hat[t+1] = z[t]``."""
    return (val_z[:, 1:] - val_z[:, :-1]).abs().mean().item()


@torch.no_grad()
def memory_health(model, z: torch.Tensor, n_probe: int = 8) -> dict:
    """Rank / magnitude / query-dependence of the online memory, on real latents.

    Measured by running the first block's own segment loop, so the state is the
    one training actually produces -- not a fresh block on synthetic input, which
    is how every earlier probe in this project was run and which was shown to
    overestimate intermediate configurations by ~40%.

      pr        participation ratio of w1's singular values. 1.0 = rank-1
                (collapsed); the healthy value is set by the memory's width.
      norm      ||w1||_F. Collapsing to 0 is a distinct pathology from rank.
      cos_ret   mean pairwise cosine of retrievals for distinct probe queries.
                1.0 = M has become a constant function of the query.
    """
    blk = model.blocks[0]
    mem = blk.memory
    B = z.shape[0]
    h = z
    C = blk.segment_size
    state, S = mem.init_memory(B)
    for i in range(h.shape[1] // C):
        seg = h[:, i * C:(i + 1) * C, :]
        p = blk.persistent(B)
        hi = mem.retrieve(state, mem.q_proj(seg))
        if getattr(blk, "norm_hist", None) is not None:
            hi = blk.norm_hist(hi)
        y = blk.attn(torch.cat([p, hi, seg], dim=1))[:, -C:, :]
        state, S = mem.update(y, state, S, blk.memory_chunk_size)
    w = state["w1"][0].double()
    sv = torch.linalg.svdvals(w)
    q = torch.randn(n_probe, model.dim, device=z.device,
                    generator=torch.Generator(device=z.device).manual_seed(7))
    r = mem.retrieve(state, q.unsqueeze(0).expand(B, -1, -1))[0].double()
    rn = r / (r.norm(dim=-1, keepdim=True) + 1e-9)
    c = rn @ rn.T
    off = c[~torch.eye(c.shape[0], dtype=bool, device=c.device)]
    return {"pr": float((sv ** 2).sum() ** 2 / (sv ** 4).sum()),
            "norm": float(w.norm()),
            "cos_ret": float(off.mean()),
            "finite": bool(torch.isfinite(w).all())}


@torch.no_grad()
def _eval_l1(model, z: torch.Tensor, batch_size: int, max_seqs: int | None = None) -> float:
    """Next-step L1 over ``z``, evaluated in mini-batches.

    A full-set forward was fine when a cache held ~160 sequences and the model was
    small. It is not: on the 5999-sequence cache with the ATLAS memory (which
    already needs ~13.5 GB to *train* at batch 8) a single forward over the whole
    split is an immediate OOM. Batches are averaged weighted by sequence count, so
    the result equals the full-set mean exactly.
    """
    model.eval()
    n = z.shape[0] if max_seqs is None else min(max_seqs, z.shape[0])
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        b = z[i:i + batch_size]
        total += F.l1_loss(model(b)[:, :-1], b[:, 1:], reduction="sum").item()
        count += b[:, 1:].numel()
    return total / max(count, 1)


def train_eval_once(
    train_z: torch.Tensor,
    val_z: torch.Tensor,
    *,
    seed: int,
    steps: int | None = None,
    epochs: float | None = None,
    lr: float = 1e-3,
    device: str = DEVICE,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    eval_max_seqs: int | None = None,
    log_every: int = 0,
    health_every: int = 0,
    **model_kwargs,
) -> dict:
    """Train one WorldModel (given seed/config) and return train & val next-step L1.

    ``batch_size`` mini-batches the training videos each step (default: full
    batch). Mini-batching both bounds VRAM (needed for larger ``dim``) and adds
    SGD noise that helps generalisation in the data-limited regime.

    Give either ``steps`` (a fixed count, the original behaviour) or ``epochs``
    (converted to ``ceil(N / batch_size)`` steps per epoch, which is the meaningful
    unit once the cache is large enough that one pass is thousands of steps).

    Evaluation is mini-batched at ``eval_batch_size`` (defaults to ``batch_size``);
    ``eval_max_seqs`` caps how many sequences are scored, for cheap periodic checks.
    Returns ``{"train_l1", "val_l1"}`` -- the gap between them tells overfitting
    (data-limited) from underfitting (architecture/optimisation).
    """
    if (steps is None) == (epochs is None):
        raise ValueError("give exactly one of steps= or epochs=")
    torch.manual_seed(seed)
    model = WorldModel(latent_dim=train_z.shape[-1], **model_kwargs).to(device)
    tz, vz = train_z.to(device), val_z.to(device)
    B = tz.shape[0]
    bs = batch_size if batch_size is not None else B
    ebs = eval_batch_size or bs
    if epochs is not None:
        steps = max(1, int(round(epochs * -(-B // bs))))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    health: list[dict] = []
    probe = vz[: min(4, vz.shape[0])]
    model.train()
    for step in range(steps):
        if health_every and step % health_every == 0:
            model.eval()
            hh = memory_health(model, probe)
            hh["step"] = step
            health.append(hh)
            if log_every:
                print(f"    [health] step {step:6d}  PR={hh['pr']:8.3f}  "
                      f"||w1||={hh['norm']:.3e}  cos_ret={hh['cos_ret']:+.4f}",
                      flush=True)
            model.train()
        idx = torch.randperm(B, device=device)[:bs] if bs < B else slice(None)
        batch = tz[idx]
        opt.zero_grad()
        loss = F.l1_loss(model(batch)[:, :-1], batch[:, 1:])
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:6d}/{steps}  train L1 {loss.item():.4f}", flush=True)
        model.train()

    if health_every:
        hh = memory_health(model, probe); hh["step"] = steps
        health.append(hh)
    return {
        "train_l1": _eval_l1(model, tz, ebs, eval_max_seqs),
        "val_l1": _eval_l1(model, vz, ebs, eval_max_seqs),
        "steps": steps,
        "health": health,
        "model": model,
    }


def run_config(
    train_path: str | Path,
    val_path: str | Path,
    *,
    seeds: Iterable[int] = range(5),
    steps: int | None = None,
    epochs: float | None = None,
    lr: float = 1e-3,
    device: str = DEVICE,
    standardize: bool = True,
    pca_k: int | None = None,
    pca_whiten: bool = True,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    eval_max_seqs: int | None = None,
    log_every: int = 0,
    health_every: int = 0,
    **model_kwargs,
) -> dict:
    """Train one config across ``seeds`` and report mean/std train & val L1.

    Standardisation stats come from train only (applied to both), so there is no
    val leakage. Returns per-seed train/val L1s, their mean/std, the persistence
    baseline, and the config used. ``val_mean - train_mean`` is the
    generalisation gap.

    ``pca_k`` reduces the latents to a **fixed** ``k``-dim PCA basis fitted on train
    (:func:`~titans.worldmodel.bottleneck.pca_reduce_splits`) and, unless the caller
    overrides it, sets ``dim=k``. The model has no learned input/output projection,
    so the latents must arrive already reduced to ``dim``; a raw (unreduced) cache is
    therefore rejected. The PCA centres (and, whitened, rescales) the data, so it
    *replaces* ``standardize``.

    **Losses under ``pca_k`` live in the PCA space and are not comparable to runs
    without it** -- including the persistence baseline, which is recomputed in the
    same space so within-run comparisons stay meaningful.
    """
    train_raw, pre = load_latent_cache(train_path)
    val_raw = load_latents(val_path)
    pca_info = None
    if pca_k is None and "pca" in pre:
        # Cache already reduced by make_pca_cache: it is centred (and, whitened,
        # unit-variance), so standardising again would rescale it a second time.
        # Configure the model for it rather than letting the defaults misread it.
        pca_info = {"k": int(pre["k"]), "whiten": bool(pre["whiten"]),
                    "explained": float(pre["explained"]), "prereduced": True}
        train_z, val_z = train_raw, val_raw
        model_kwargs.setdefault("dim", pca_info["k"])
    elif pca_k is not None:
        train_z, val_z, pca = pca_reduce_splits(train_raw, val_raw, pca_k,
                                                whiten=pca_whiten)
        pca_info = {"k": pca_k, "whiten": pca_whiten, "explained": pca["explained"],
                    "prereduced": False}
        model_kwargs.setdefault("dim", pca_k)
    else:
        # No PCA anywhere: the model cannot absorb a width mismatch any more, so this
        # is only viable when the cache is already at the model width. Standardisation
        # (a per-channel affine) does not change the width, so it stays available.
        if standardize:
            mean, std = train_stats(train_raw)
            train_z = _standardize(train_raw, mean, std)
            val_z = _standardize(val_raw, mean, std)
        else:
            train_z, val_z = train_raw, val_raw
        model_kwargs.setdefault("dim", train_z.shape[-1])

    baseline = persistence_baseline_l1(val_z)
    per_seed = []
    for s in seeds:
        r = train_eval_once(train_z, val_z, seed=s, steps=steps, epochs=epochs, lr=lr,
                            device=device, batch_size=batch_size,
                            eval_batch_size=eval_batch_size,
                            eval_max_seqs=eval_max_seqs, log_every=log_every,
                            health_every=health_every, **model_kwargs)
        r.pop("model", None)   # don't pin every seed's weights in memory
        per_seed.append(r)
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
        "pca": pca_info,
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
    steps: int | None = None,
    epochs: float | None = None,
    lr: float = 1e-3,
    device: str = DEVICE,
    **run_kwargs,
) -> dict:
    """Run several named configs and return their results keyed by name."""
    seeds = list(seeds)
    return {
        name: run_config(train_path, val_path, seeds=seeds, steps=steps,
                         epochs=epochs, lr=lr, device=device, **run_kwargs, **cfg)
        for name, cfg in variants.items()
    }


def format_comparison(results: dict) -> str:
    """Table of variant -> train & val L1, gen gap, vs the persistence baseline."""
    first = next(iter(results.values()))
    baseline = first["baseline_l1"]
    dev = first["device"]
    n = first["n_seeds"]
    pca = first.get("pca")
    lines = []
    if pca:
        # Loud, because an L1 in PCA space silently looks like an L1 in latent space.
        lines.append(f"SPACE: PCA k={pca['k']} whiten={pca['whiten']} "
                     f"(explains {100 * pca['explained']:.1f}% of train variance) "
                     "-- L1 values are NOT comparable to non-PCA runs")
    lines += [
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
