"""Phase 3: a small Titans-based world model over latent sequences.

* :class:`WorldModel` -- next-latent predictor stacking MAC blocks (core, no
  extra dependencies).
* :func:`train_demo` / :func:`next_step_loss` / :func:`synthetic_latents` -- an
  L1-trained, runnable-standalone demo (``python -m titans.worldmodel.train``).
* :class:`VJEPA2Encoder` -- optional frozen video encoder (needs ``transformers``,
  imported lazily) turning real clips into the latents the model consumes.
"""

from .model import WorldModel
from .train import load_latents, next_step_loss, synthetic_latents, train_demo
from .encoder import VJEPA2Encoder
from .frames import aggregate_frames, split_into_clips
from .precompute_latents import (
    precompute,
    precompute_kinetics,
    video_to_latents,
    video_to_latents_sliding,
)
from .data import kinetics_frames, list_clips, load_video_frames
from .ablation import run_ablation, format_report

__all__ = [
    "WorldModel",
    "next_step_loss",
    "synthetic_latents",
    "train_demo",
    "load_latents",
    "VJEPA2Encoder",
    "aggregate_frames",
    "split_into_clips",
    "precompute",
    "precompute_kinetics",
    "video_to_latents",
    "video_to_latents_sliding",
    "kinetics_frames",
    "list_clips",
    "load_video_frames",
    "run_ablation",
    "format_report",
]
