"""Phase 3: a small Titans-based world model over latent sequences.

* :class:`WorldModel` -- next-latent predictor stacking MAC blocks (core, no
  extra dependencies).
* :func:`train_demo` / :func:`next_step_loss` / :func:`synthetic_latents` -- an
  L1-trained, runnable-standalone demo (``python -m titans.worldmodel.train``).
* :mod:`echo` -- the [V;V]/[V;W] memory benchmark (``memory_gain``), how memory
  variants are compared. Run as a module (``python -m titans.worldmodel.echo``);
  deliberately not re-exported here, so running it is a single import.
* :class:`VJEPA2Encoder` -- optional frozen video encoder (needs ``transformers``,
  imported lazily) turning real clips into the latents the model consumes.
"""

from .model import WorldModel
from .bottleneck import fit_pca, apply_pca, inverse_pca, pca_reduce_splits
from .train import (
    train_stats,
    latent_cache_meta,
    load_latent_cache,
    load_latents,
    next_step_loss,
    synthetic_latents,
    train_demo,
)
from .encoder import VJEPA2Encoder
from .precompute_latents import (
    precompute,
    precompute_egolife,
    video_to_latents_sliding,
)
from .data import load_video_frames
from .harness import run_config, compare_variants, format_comparison, persistence_baseline_l1

__all__ = [
    "WorldModel",
    "fit_pca",
    "apply_pca",
    "inverse_pca",
    "pca_reduce_splits",
    "latent_cache_meta",
    "train_stats",
    "load_latent_cache",
    "next_step_loss",
    "synthetic_latents",
    "train_demo",
    "load_latents",
    "VJEPA2Encoder",
    "precompute",
    "precompute_egolife",
    "video_to_latents_sliding",
    "load_video_frames",
    "run_config",
    "compare_variants",
    "format_comparison",
    "persistence_baseline_l1",
]
