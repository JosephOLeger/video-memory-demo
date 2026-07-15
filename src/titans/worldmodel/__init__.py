"""Phase 3: a small Titans-based world model over latent sequences.

* :class:`WorldModel` -- next-latent predictor stacking MAC blocks (core, no
  extra dependencies).
* :class:`GMMHead` / :func:`gmm_nll` -- Gaussian-mixture (MDN) output head + loss
  for a *distributional* next-latent objective (vs the unimodal L1).
* :func:`train_demo` / :func:`next_step_loss` / :func:`synthetic_latents` -- an
  L1-trained, runnable-standalone demo (``python -m titans.worldmodel.train``).
* :class:`VJEPA2Encoder` -- optional frozen video encoder (needs ``transformers``,
  imported lazily) turning real clips into the latents the model consumes.
"""

from .model import WorldModel
from .gmm import GMMHead, gmm_nll, gmm_log_prob, mixture_mean, dominant_mean
from .bottleneck import fit_pca, apply_pca, inverse_pca
from .train import load_latents, next_step_loss, synthetic_latents, train_demo
from .encoder import VJEPA2Encoder
from .precompute_latents import (
    precompute,
    precompute_egolife,
    video_to_latents_sliding,
)
from .data import load_video_frames
from .ablation import run_ablation, format_report
from .harness import run_config, compare_variants, format_comparison, persistence_baseline_l1

__all__ = [
    "WorldModel",
    "GMMHead",
    "gmm_nll",
    "gmm_log_prob",
    "mixture_mean",
    "dominant_mean",
    "fit_pca",
    "apply_pca",
    "inverse_pca",
    "next_step_loss",
    "synthetic_latents",
    "train_demo",
    "load_latents",
    "VJEPA2Encoder",
    "precompute",
    "precompute_egolife",
    "video_to_latents_sliding",
    "load_video_frames",
    "run_ablation",
    "format_report",
    "run_config",
    "compare_variants",
    "format_comparison",
    "persistence_baseline_l1",
]
