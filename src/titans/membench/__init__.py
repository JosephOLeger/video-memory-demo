"""Synthetic memory benchmark: does the Titans memory actually *remember*?

Next-step prediction loss (our world-model setup) is nearly blind to long-term
memory -- predicting the next latent is dominated by local dynamics, so a working
memory contributes little measurable signal. The Titans/ATLAS papers therefore
demonstrate memory on tasks with *explicit* long-range dependency (associative
recall, needle-in-haystack), scored by retrieval **accuracy**, not perplexity.

This package provides those probes, isolated from all video/world-model
confounds:

* :mod:`tasks` -- ``associative_recall``, ``copy_task``, ``needle`` generators.
* :mod:`model` -- :class:`RecallModel`, a token model built on ``TitansMACBlock``.
* :mod:`run` -- train + accuracy eval (optionally accuracy vs distance).

The design forces reliance on the long-term memory: sequences are much longer
than ``segment_size`` (the attention window), so the only pathway for
early-to-late information is the memory threaded across segments.
"""

from .tasks import associative_recall, copy_task, needle
from .model import RecallModel
from .run import train_recall, accuracy_by_distance

__all__ = [
    "associative_recall",
    "copy_task",
    "needle",
    "RecallModel",
    "train_recall",
    "accuracy_by_distance",
]
