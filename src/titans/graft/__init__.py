"""Grafting the neural long-term memory onto a frozen VLM.

Everything here only ever *adds* to the host model: a lateral branch on one
decoder layer's residual stream, no attention replaced, no context shortened.
"""

from .branch import (
    ETA_INIT,
    SANE_NS,
    MemoryBranch,
    fold_pca,
    sane_memory_kwargs,
    stride_for_reach,
)

__all__ = [
    "MemoryBranch",
    "SANE_NS",
    "ETA_INIT",
    "fold_pca",
    "sane_memory_kwargs",
    "stride_for_reach",
]
