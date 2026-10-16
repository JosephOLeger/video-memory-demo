"""Titans: neural long-term memory + MAC architecture (arXiv:2501.00663).

Phase 1: the neural long-term memory core (:mod:`titans.memory`).
Phase 2: Memory-As-Context (:class:`titans.mac.TitansMACBlock`), built from a
persistent-memory module, a Titans-flavoured causal self-attention, and the
memory's new retrieval-free ``update`` entry point.
"""

from .attention import CausalSelfAttention, DepthwiseSeparableCausalConv1d
from .mac import TitansMACBlock
from .memory import NeuralMemory, NeuralMemoryMLP, mlp_forward
from .persistent import PersistentMemory

__all__ = [
    "NeuralMemory",
    "NeuralMemoryMLP",
    "mlp_forward",
    "PersistentMemory",
    "CausalSelfAttention",
    "DepthwiseSeparableCausalConv1d",
    "TitansMACBlock",
]
