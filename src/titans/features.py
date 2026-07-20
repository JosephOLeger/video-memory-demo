"""Polynomial feature maps for memory keys/queries (ATLAS, arXiv:2505.23735 §3.1).

ATLAS's first named improvement over Titans is *memory capacity*. Its argument
(Proposition 2): a matrix-valued memory optimising ``||M phi_p(k_t) - v_t||^2``
can store ``O(d_k^p)`` linearly independent key-value pairs, against ``O(d_k)``
for the identity map. Raising the key dimension directly would cost ``O(d_in)``
parameters per extra dimension in the input projections; a feature map raises the
*effective* dimension for free.

The paper defines ``phi_p(x) = [x^beta]_{|beta| <= p}`` -- every monomial of
degree at most ``p`` -- and motivates it as a Taylor approximation of the
softmax/exponential kernel (Eq 4-5)::

    exp(q^T k) ~= phi_p(q) phi_p(k)^T = a_0 + a_1 q^T k + a_2 (q^T k)^2 + ... + a_p (q^T k)^p

with the ``a_i`` learnable and initialised at ``1/i!``. The ``a_i`` double as an
*input gate*: driving ``a_i -> 0`` drops the degree-``i`` features, and the model
can recover ``phi(x) = x`` exactly by learning ``a_1 -> 1``, ``a_{i != 1} -> 0``.

**Documented deviation: this is the diagonal subset, not the full monomial
basis.** The complete basis is combinatorially unusable here -- for ``d=64`` it is
2145 dimensions at ``p=2`` and 47905 at ``p=3``, and since our memory ``M`` is an
MLP (not a matrix) we cannot use the kernel trick to avoid materialising it: the
first memory layer would become ``2145 x hidden``, destroying the parameter
budget the feature map exists to protect. We therefore keep only the *pure
powers* ``x_j^i`` and drop the cross terms ``x_j x_l``::

    phi(x) = concat([c_1 * x, c_2 * x^2, ..., c_p * x^p]),   dim = p * d

so ``phi(q) . phi(k) = sum_i c_i^2 <q^i, k^i>`` rather than
``sum_i a_i (q^T k)^i``. The structure of Eq 5 (a learned, gated mixture of
increasing degrees) is preserved; the exact kernel identity is not. Coefficients
are stored as the *feature* scale ``c_i``, initialised at ``1/sqrt(i!)`` so that
the induced kernel weight ``c_i^2`` matches the paper's ``a_i = 1/i!``.

``a_0`` (the constant term) is dropped: the memory MLP already has bias terms, so
a constant feature is redundant.

Cross terms are the real loss here. A faithful alternative is the tensor-sketch
of Kacham et al. (2024a) (PolySketchFormer), which ATLAS cites -- it approximates
the full kernel in low dimension, at the cost of a stochastic approximation to
validate. Not implemented.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PolynomialFeatureMap(nn.Module):
    """Map ``[..., d] -> [..., d * degree]`` by concatenating scaled powers.

    Parameters
    ----------
    degree: highest power ``p``. ``degree=1`` is the identity map up to the
        learned scale ``c_1`` (initialised to 1), so it stays a no-op reparam of
        plain Titans.
    normalize: RMS-normalise the input before taking powers (default ``True``).
        Without it ``x^p`` is numerically explosive -- keys come straight from an
        unconstrained ``k_proj``, so a coordinate of magnitude 3 reaches 27 at
        ``p=3`` and the inner loss/gradient scale with it. Normalising first puts
        every degree on a comparable footing, which is what makes the learned
        ``c_i`` gate meaningful rather than a fight against scale. This is our
        choice, not the paper's -- ATLAS does not discuss the conditioning of
        ``phi``.
    """

    def __init__(self, degree: int = 2, normalize: bool = True, eps: float = 1e-6):
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.normalize = normalize
        self.eps = eps
        # c_i = 1/sqrt(i!) so the induced kernel weight c_i^2 = 1/i! (ATLAS Eq 5).
        coef = torch.tensor(
            [1.0 / math.sqrt(math.factorial(i)) for i in range(1, degree + 1)]
        )
        self.coef = nn.Parameter(coef)

    def out_dim(self, in_dim: int) -> int:
        """Dimension of the mapped features for an input of width ``in_dim``."""
        return in_dim * self.degree

    def extra_repr(self) -> str:
        return f"degree={self.degree}, normalize={self.normalize}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``[..., d]`` -> ``[..., d * degree]``."""
        if self.degree == 1:
            return x * self.coef[0]
        if self.normalize:
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            x = x / rms
        # Even powers discard sign -- that is what the monomial basis does, so it
        # is faithful rather than a bug; the odd powers carry the sign through.
        feats = [self.coef[i - 1] * x.pow(i) for i in range(1, self.degree + 1)]
        return torch.cat(feats, dim=-1)
