"""MQAR: does the Titans memory actually *remember*?

Next-step prediction loss (the world-model setup) is nearly blind to long-term
memory -- predicting the next latent is dominated by local dynamics, so a working
or a broken memory barely moves it. Titans/ATLAS therefore demonstrate memory on
tasks with an *explicit* long-range dependency, scored by retrieval **accuracy**
against a known chance level rather than by perplexity.

This package keeps exactly one such probe, **multi-query associative recall**
(MQAR): show ``num_pairs`` (key, value) pairs, then query one key. Sequences are
padded to be much longer than ``segment_size``, so attention only ever sees within
one segment and the *only* pathway from an early pair to the late query is the
memory threaded across segments. That is what makes the accuracy a memory
measurement rather than an attention measurement.

* :mod:`tasks` -- the ``associative_recall`` generator.
* :mod:`model` -- :class:`RecallModel`, a token model built on ``TitansMACBlock``.
* :mod:`run`   -- train + held-out accuracy, and accuracy bucketed by distance
  (``accuracy_by_distance``), which separates "solved inside the attention window"
  from "solved by the memory".

Earlier revisions also carried ``copy_task``, ``needle``, ``selective_copy`` and
``fuzzy_recall``, plus a three-config momentum sweep. They were removed: the video
[V;V]/[V;W] echo benchmark (:mod:`titans.worldmodel.echo`) is where memory variants
are compared now, and MQAR is kept as the one token-level cross-check.
"""

from .tasks import associative_recall
from .model import RecallModel
from .run import train_recall, train_recall_converge, accuracy_by_distance

__all__ = [
    "associative_recall",
    "RecallModel",
    "train_recall",
    "train_recall_converge",
    "accuracy_by_distance",
]
