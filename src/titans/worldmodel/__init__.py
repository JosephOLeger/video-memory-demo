"""Phase 3: a small Titans-based world model over latent sequences.

* :class:`WorldModel` -- next-latent predictor stacking MAC blocks (core, no
  extra dependencies).
* :func:`train_demo` / :func:`next_step_loss` / :func:`synthetic_latents` -- an
  L1-trained, runnable-standalone demo (``python -m titans.worldmodel.train``).
* :class:`VJEPA2Encoder` -- optional frozen video encoder (needs ``transformers``,
  imported lazily) turning real clips into the latents the model consumes.
"""

from .model import WorldModel
from .train import next_step_loss, synthetic_latents, train_demo
from .encoder import VJEPA2Encoder

__all__ = [
    "WorldModel",
    "next_step_loss",
    "synthetic_latents",
    "train_demo",
    "VJEPA2Encoder",
]
