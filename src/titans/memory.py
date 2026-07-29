"""Neural long-term memory module (Titans, arXiv:2501.00663) -- Phase 1.

This implements the "neural memory as a deep MLP whose weights are updated at
test time" from Section 3.1 of the paper, plus the chunked / parallel
reformulation from Section 3.2.

The memory ``M`` is a small MLP whose *weights* are the online-learned state.
At every time step ``t`` we:

  1. project the token: ``k_t = x_t W_K``, ``v_t = x_t W_V``, ``q_t = x_t W_Q``;
  2. measure the associative-memory (inner) loss with the *current* weights
     ``M_{t-1}``:  ``l_t = || M_{t-1}(k_t) - v_t ||_2^2`` ;
  3. update the weights with a momentum ("surprise") + weight-decay rule::

         S_t = eta_t * S_{t-1} - theta_t * grad_W l_t
         M_t = (1 - alpha_t) * M_{t-1} + S_t

  4. retrieve ``y_t = M_t(q_t)`` (forward pass, no weight update).

``alpha_t`` (forget gate), ``eta_t`` (momentum decay) and ``theta_t`` (inner
learning rate) are produced by tiny linear heads on ``x_t``. They are scalars
per token (per-channel vectors are a straightforward extension -- see the note
in the README / report).

Two code paths are provided and are mathematically identical:

* :meth:`NeuralMemory.forward_naive_sequential` -- an explicit Python for-loop,
  the ground-truth reference. ``torch.func.grad`` gives the per-step inner
  gradient of ``M``'s own parameters (a functional/meta-learning inner loop;
  no autograd is tracked through the outer model for this inner update).
* :meth:`NeuralMemory.forward_chunked` -- the Section 3.2 parallelization:
  gradients for a whole chunk are computed in one batched (vmap'd) call at the
  chunk's start weights, then the momentum/weight-decay recurrence is scanned
  across the chunk.

Note on the chunk-start gradient. The parallel form (Eq. 16-17) evaluates the
inner gradient of every token in a chunk at the weights held at the *start* of
the chunk (mini-batch gradient descent). For a nonlinear deep MLP this is only
equal to the fully-online recurrence when ``chunk_size == 1``. To make the two
code paths comparable for any chunk size, ``forward_naive_sequential`` accepts
the same ``chunk_size`` and uses the same chunk-start semantics; with
``chunk_size=None`` (default) it runs the fully-online recurrence, which equals
``forward_chunked(..., chunk_size=1)``.

Cost of the objective (``window_size``, the ATLAS Omega rule -- see
:class:`NeuralMemory`). ``window_size=1`` is the **default** and is Titans' delta
rule: the per-token gradient is used directly, with no window machinery at all
(``_scan_chunked`` takes a dedicated branch), so choosing the cheap objective is
just leaving the parameter alone. ``window_size=c>1`` adds **no extra backward
pass** -- the gradient of a sum is the sum of gradients, and every per-token
gradient is already computed -- but it does add a ``c``-term weighted sum per
token per weight tensor in the scan, and creates the ``gamma_head`` gates. If a
run is too slow, dropping to ``window_size=1`` recovers the original cost exactly.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

from .features import PolynomialFeatureMap
from .metaplastic import MetaplasticCascade, cascade_step, independent_step

Params = Dict[str, torch.Tensor]


def newton_schulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Orthogonalise a batch of matrices toward the nearest semi-orthogonal one.

    The quintic Newton-Schulz iteration used by the Muon optimizer (Jordan et al.
    2024) and by ATLAS's own inner-loop optimiser (arXiv:2505.23735, Eq 57-58,
    ``M_t = alpha_t M_{t-1} + NewtonSchulz5(S_t)``). Iterating a degree-5
    polynomial map drives all singular values of ``G`` toward 1 while preserving
    its singular *vectors* -- i.e. it reshapes the update to have balanced
    "spread" across directions instead of being dominated by whichever direction
    the accumulated momentum happens to have reinforced most.

    Motivation here: diagnosed directly on this project's memory (see the needle/
    collapse probes) -- accumulating momentum from many sequential outer-product
    gradients (``-theta_t * grad``) over a long threaded sequence can drift the
    weight update toward an (approximately) rank-1 matrix, so the memory's output
    *direction* stops depending on the query even though its *magnitude* still
    does (a distinct failure mode from the bias-dominance collapse ``update_bias``
    guards against). Orthogonalising the momentum before it updates the weights
    is a direct, literature-grounded counter to this specific degeneracy.

    ``G`` is ``[B, out, in]`` (a batch of weight-shaped momentum tensors);
    returns a same-shape tensor with each matrix pushed toward semi-orthogonal.
    Anything with fewer than 3 dims (e.g. a batched bias ``[B, out]``, which has
    no per-example matrix structure to orthogonalise) passes through unchanged.
    """
    if G.dim() < 3:
        return G
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    # The NS iteration itself changes X's Frobenius norm (pushing singular values
    # toward 1 changes their sum of squares), so re-normalise to unit norm before
    # rescaling to G's original magnitude -- isolates "balance the singular
    # values" from a confounding change in update magnitude (theta_t's learned
    # scale assumes momentum sized like -theta_t*grad, not NS's native scale).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    return X * G.norm(dim=(-2, -1), keepdim=True)


def mlp_forward(
    params: Params,
    x: torch.Tensor,
    num_layers: int,
    activation: str = "silu",
    residual: bool = False,
    normalize_output: bool = True,
) -> torch.Tensor:
    """Pure functional forward of the memory MLP for a *single* example.

    ``params`` holds ``w{i}`` of shape ``[out, in]`` and ``b{i}`` of shape
    ``[out]``. ``x`` has shape ``[in]``. Designed to be composed with
    ``torch.func.grad`` / ``torch.func.vmap``.

    ``activation`` selects the between-layer nonlinearity: ``"silu"`` (this
    project's original) or ``"gelu"`` (what ATLAS specifies, Table 7 caption).
    ``residual`` adds ATLAS's skip connection ``M(x) = x + W_1 sigma(W_2 x)``
    (Table 7 caption). When a feature map has widened the input, that formula is
    ill-typed as literally written -- ``phi(k)`` has ``p*d`` dims while ``M`` must
    output ``d`` to be compared against ``v`` -- and the paper does not say how the
    two combine. We add the **leading ``d`` slice of the input**, which for our
    ``phi`` is exactly its degree-1 block ``c_1 * k``, i.e. the un-mapped key: the
    skip carries the original key, and the higher-degree terms reach the output
    only through the learned layers. That is the reading that degenerates to the
    paper's formula when ``phi`` is the identity. Documented as a resolution of an
    under-specified point, not as the paper's own choice.

    ``normalize_output`` RMS-normalises the result. It was introduced to stop the
    online update diverging: the momentum recurrence lets ``M``'s weights -- and
    hence ``M(q)`` and the associative loss -- grow geometrically per segment and
    overflow to ``NaN`` once a sequence spans enough segments.

    **It is a deviation from ATLAS, and an expensive one.** The paper normalises
    the memory's *inputs*: "We apply normalization on keys and queries to
    stabilize the training" (arXiv:2505.23735 p12), and writes the deep memory as
    ``M(x) = x + W_1 sigma(W_2 x)`` with no output normalisation at all. Pinning
    ``||M(k)|| = 1`` while the target ``v = W_V y`` is left unnormalised makes the
    two live on different scales -- measured here, ``||v|| ~ 0.01`` against
    ``||M(k)|| = 1``, a factor 100. The inner loss ``||M(k) - v||^2`` is then
    dominated by ``M``'s own fixed-norm output, the error ``e = M(k) - v`` is
    essentially ``M(k)``, and the gradient carries almost no information about
    ``v``: the associative rule stops associating. Prefer ``normalize_qk`` on
    :class:`NeuralMemory`, which is what the paper actually does.
    """
    h = x
    for i in range(num_layers):
        h = h @ params[f"w{i}"].transpose(-1, -2) + params[f"b{i}"]
        if i < num_layers - 1:
            h = F.gelu(h) if activation == "gelu" else F.silu(h)
    if residual and x.shape[-1] >= h.shape[-1]:
        # Equal widths -> the paper's M(x) = x + W_1 sigma(W_2 x) verbatim.
        # Wider input (a feature map is active) -> the leading slice, which is
        # phi's degree-1 block = the original key. See the docstring.
        h = h + (x if h.shape[-1] == x.shape[-1] else x[..., : h.shape[-1]])
    if not normalize_output:
        return h
    # Parameter-free RMSNorm: keeps M's output on a bounded scale regardless of
    # how large the online-updated weights become. See the docstring for why this
    # is a deviation worth avoiding.
    rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return h / rms


class NeuralMemoryMLP(nn.Module):
    """The memory network ``M``: a small MLP, ``in_dim -> ... -> in_dim``.

    Only stores the *initial* (outer-trained) weights ``M_0``. The test-time
    update is carried out functionally by :class:`NeuralMemory`, so nothing here
    is mutated in place.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        in_dim: Optional[int] = None,
        activation: str = "silu",
        residual: bool = False,
        normalize_output: bool = True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Titans memory expects a deep MLP (num_layers >= 2).")
        self.dim = dim
        # ``in_dim`` differs from ``dim`` when a polynomial feature map widens the
        # keys/queries (ATLAS phi): M maps ``phi(k) in R^{p*d}`` back to ``R^d``.
        self.in_dim = in_dim or dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Kept here so :meth:`forward` cannot drift from the functional path that
        # the inner loop actually runs -- it silently ignored both before.
        self.activation = activation
        self.residual = residual
        self.normalize_output = normalize_output

        self.params = nn.ParameterDict()
        for i in range(num_layers):
            in_d = self.in_dim if i == 0 else hidden_dim
            out_d = dim if i == num_layers - 1 else hidden_dim
            w = torch.empty(out_d, in_d)
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
            self.params[f"w{i}"] = nn.Parameter(w)
            self.params[f"b{i}"] = nn.Parameter(torch.zeros(out_d))

    def init_state(self, batch: int) -> Params:
        """Return a fresh per-sequence weight state with a leading batch dim."""
        return {
            k: v.unsqueeze(0).expand(batch, *v.shape).clone()
            for k, v in self.params.items()
        }

    def forward(self, params: Params, x: torch.Tensor) -> torch.Tensor:
        return mlp_forward(params, x, self.num_layers, self.activation, self.residual,
                           self.normalize_output)


class NeuralMemory(nn.Module):
    """Neural long-term memory: outer projections + the online-updated MLP.

    Parameters
    ----------
    dim: token / key / value / query dimension (``d_in``).
    hidden_dim: hidden width of the memory MLP.
    num_layers: depth ``L_M`` of the memory MLP (>= 2).
    theta_scale: multiplies the softplus output for ``theta_t`` (inner lr).
    window_size: ATLAS *Omega rule* window ``c`` (arXiv:2505.23735). With ``c=1``
        (default) the inner update optimises the loss of the current token only
        -- the original Titans/Delta rule. With ``c>1`` each update instead
        optimises a decayed sum over the last ``c`` tokens
        ``sum_{i=t-c+1}^{t} gamma_i * ||M(k_i) - v_i||^2`` (Eq 9), so the memory
        memorises a local *context* rather than individual tokens. Because the
        gradient is linear, this is just a gated sliding-window sum of the
        per-token gradients we already compute -- no extra backward passes.
    use_cascade_momentum: opt-in flag (default ``False`` -> bit-identical to the
        original scalar-momentum Titans). When ``True`` the single momentum scalar
        ``S_t`` per weight is replaced by a length-``K`` metaplastic cascade state
        ``U_t`` (:class:`~titans.metaplastic.MetaplasticCascade`); the visible
        momentum fed into ``M_t = (1-alpha_t) M_{t-1} + S_t`` becomes
        ``read(U_t) = U_t^(1)``. See :meth:`_step`.
    cascade_reservoirs, cascade_beta1, cascade_g, cascade_selective,
    cascade_topology: forwarded to
        :class:`~titans.metaplastic.MetaplasticCascade` (``K``, geometric coupling
        hyperparameters, the data-dependent per-edge gate, and the chain-vs-
        independent topology). Ignored unless ``use_cascade_momentum`` is set.
        ``cascade_topology="independent"`` gives the diagonal-``B`` null hypothesis:
        ``K`` uncoupled exponential filters (gradient into all reservoirs, readout
        = mean), used by the sweep's ``uncoupled`` config to test whether the
        cascade's *coupling* -- not merely having K filters -- adds anything.

    Cascade design choices (documented per the project convention):

    * **``eta_t``'s new role.** In the scalar rule ``eta_t`` decays the whole
      momentum before the fresh surprise is added (``S_t = eta_t S_{t-1} - ...``).
      Reservoir 1 of the cascade has *no* self-decay (see ``metaplastic.py``), so
      a run of same-sign gradients can grow it linearly. We keep ``eta_t`` as a
      per-token multiplicative leak applied to the *entire* cascade state before
      the diffusion step: ``U_t = B_t (eta_t U_{t-1}) + e_1 (-theta_t grad_t)``.
      This (a) restores the missing leak, (b) preserves ``eta_t``'s data-dependent
      plasticity-control semantics, and (c) makes ``K=1`` collapse *exactly* onto
      the scalar recurrence, so the cascade is a strict generalisation.
    * **Selective gate input.** When ``cascade_selective`` is set the per-edge
      couplings ``B_t`` are gated by the *token* ``x_t`` (the same ``[B, dim]``
      representation feeding ``alpha_head``/``eta_head``), not by the per-key
      gradient: the surprise-driven consolidation the selective mode models is a
      property of the event (the token), shared across all weight entries, so one
      gate vector per token broadcasts over every parameter tensor.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        theta_scale: float = 1.0,
        window_size: int = 1,
        window_normalize: bool = False,
        use_cascade_momentum: bool = False,
        cascade_reservoirs: int = 4,
        cascade_beta1: float = 0.5,
        cascade_g: float = 2.0,
        cascade_selective: bool = False,
        cascade_topology: str = "chain",
        cascade_readout: Optional[str] = None,
        cascade_gate_signal: str = "content",
        cascade_leak: str = "full",
        cascade_slow_decay: float = 0.99,
        update_bias: bool = True,
        orthogonalize_momentum: bool = False,
        feature_degree: int = 1,
        feature_normalize: bool = True,
        memory_expansion: Optional[int] = None,
        memory_activation: str = "silu",
        memory_residual: bool = False,
        normalize_qk: "bool | str" = False,
        normalize_output: bool = True,
        alpha_init: Optional[float] = None,
    ):
        super().__init__()
        # ATLAS Table 7 specifies the memory MLP as 2 layers, expansion factor 4,
        # GELU, and a residual M(x) = x + W_1 sigma(W_2 x). All three are opt-in
        # here so the default stays bit-identical to the previous behaviour
        # (expansion 1, SiLU, no residual).
        if memory_expansion is not None:
            if memory_expansion < 1:
                raise ValueError("memory_expansion must be >= 1")
            hidden_dim = dim * memory_expansion
        hidden_dim = hidden_dim or dim
        if memory_activation not in ("silu", "gelu"):
            raise ValueError("memory_activation must be 'silu' or 'gelu'")
        self.memory_activation = memory_activation
        self.memory_residual = memory_residual
        # ATLAS p12: "We apply normalization on keys and queries to stabilize the
        # training." That is where the paper puts its stabiliser -- on M's INPUTS.
        # `normalize_output` (in mlp_forward) instead pins M's output norm, which
        # is not in the paper and breaks the scale match with the target v.
        # Defaults keep the historical behaviour; set normalize_qk=True and
        # normalize_output=False for the paper's arrangement.
        # False | "rms" | "l2"  (True is accepted as "rms" for back-compat).
        # The choice is not cosmetic: it sets lambda_max(K K^T), and ATLAS p25
        # gives the inner loop's convergence condition as
        #     0 < eta < 2 / lambda_max(K K^T).
        # RMS-normalising leaves ||k||^2 = d, so a chunk of m collinear keys has
        # lambda_max ~ m*d (measured 247 at m=8, d=32) and the step must stay
        # under 0.008 -- which our theta ~ 0.69 violates by 85x, and which is
        # exactly the divergence observed when the output norm is removed.
        # L2-normalising leaves ||k|| = 1, so lambda_max ~ m (~8) and the bound
        # is ~0.25: thirty-two times more usable, for the same stabilising role.
        if normalize_qk is True:
            normalize_qk = "rms"
        if normalize_qk not in (False, None, "rms", "l2"):
            raise ValueError("normalize_qk must be False, 'rms' or 'l2'")
        self.normalize_qk = normalize_qk or False
        self.normalize_output = normalize_output
        self.dim = dim
        self.theta_scale = theta_scale
        # ATLAS phi: polynomial feature map on keys and queries (degree 1 = the
        # identity map, i.e. plain Titans up to a learned unit-init scale).
        self.feature_degree = feature_degree
        self.feature_map = (
            PolynomialFeatureMap(feature_degree, normalize=feature_normalize)
            if feature_degree > 1
            else None
        )
        mem_in_dim = dim * feature_degree if feature_degree > 1 else dim
        if window_size < 1:
            raise ValueError("window_size (Omega c) must be >= 1")
        self.window_size = window_size
        # Divide the Omega window sum by its gate mass (see _scan_chunked). Off by
        # default: Eq 58 writes a plain sum, and turning this on is a deliberate
        # deviation to separate "more context" from "bigger inner step".
        self.window_normalize = window_normalize
        # Default True preserves prior behaviour (biases updated online, same as
        # weights). False keeps M's bias terms fixed at their outer-trained value
        # -- see the note in _step: neither Titans nor ATLAS's published M_t
        # recurrence has a bias term at all, so this tests whether the online
        # bias update (an nn.Linear-convention addition, not a paper-specified
        # mechanism) is what drives the memory's collapse to a near-constant
        # function under long inner-loop threading.
        self.update_bias = update_bias
        # Opt-in Newton-Schulz orthogonalisation of the scalar-path momentum
        # before it updates 2D weight matrices (see newton_schulz5's docstring).
        # Scoped to the plain (non-cascade) path for now.
        self.orthogonalize_momentum = orthogonalize_momentum

        # Opt-in metaplastic cascade momentum (default off -> scalar Titans).
        self.use_cascade_momentum = use_cascade_momentum
        if cascade_leak not in ("full", "shallow", "none", "slow", "surprise"):
            raise ValueError(
                "cascade_leak must be 'full', 'shallow', 'none', 'slow' or 'surprise'"
            )
        self.cascade_leak = cascade_leak
        self.cascade_slow_decay = cascade_slow_decay
        if use_cascade_momentum:
            # `dim` is needed by the cascade whenever a head reads the token: the
            # content-gated coupling (cascade_selective + gate_signal="content") or
            # the context-dependent readout (cascade_readout="gated"). The
            # surprise-gated coupling needs no `dim` (its head takes the scalar
            # log-surprise, not the token).
            needs_dim = (cascade_selective and cascade_gate_signal == "content") or (
                cascade_readout == "gated"
            )
            self.cascade = MetaplasticCascade(
                cascade_reservoirs,
                beta1=cascade_beta1,
                g=cascade_g,
                dim=dim if needs_dim else None,
                selective=cascade_selective,
                topology=cascade_topology,
                readout=cascade_readout,
                gate_signal=cascade_gate_signal,
            )

        self.memory = NeuralMemoryMLP(dim, hidden_dim, num_layers, in_dim=mem_in_dim,
                                      activation=memory_activation,
                                      residual=memory_residual,
                                      normalize_output=normalize_output)
        self.num_layers = num_layers

        # Outer, ordinarily-trained projections (shared across time).
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)

        # Data-dependent gate heads: scalar per token.
        self.alpha_head = nn.Linear(dim, 1)  # forget gate  -> sigmoid
        self.eta_head = nn.Linear(dim, 1)    # momentum decay -> sigmoid
        self.theta_head = nn.Linear(dim, 1)  # inner lr -> softplus
        # Omega per-token pruning gate gamma_i (only needed for c>1). Created
        # conditionally so c=1 stays bit-identical to the original Titans memory.
        if window_size > 1:
            self.gamma_head = nn.Linear(dim, 1)  # context gate -> sigmoid

        # alpha_init sets the forget gate's resting value via alpha_head's bias.
        #
        # The default (no bias init) leaves alpha = sigmoid(0) = 0.5, so
        # M_t = (1 - alpha) M_{t-1} + S_t keeps HALF the memory per token: a
        # half-life of one token, which over a 24-token sequence retains 0.5^24
        # ~ 6e-8. That is a one-token buffer, not a long-term memory, and it is
        # the dominant cause of the rank-1 collapse this project chased for
        # weeks: nothing survives long enough to accumulate into more than one
        # direction. Measured over 16 threaded segments, PR(w1) goes from 1.0 at
        # alpha=0.5 to 16.1 -- full rank -- at alpha=0.002 (half-life ~350
        # tokens), with no Newton-Schulz and no bias freeze.
        #
        # Half-life in tokens is ln(2) / -ln(1 - alpha).
        if alpha_init is not None:
            if not 0.0 < alpha_init < 1.0:
                raise ValueError("alpha_init must be in (0, 1)")
            with torch.no_grad():
                self.alpha_head.bias.fill_(math.log(alpha_init / (1 - alpha_init)))

    # ------------------------------------------------------------------ utils
    def _grad_fn(self):
        num_layers = self.num_layers
        act, res = self.memory_activation, self.memory_residual
        nout = self.normalize_output

        def loss(params: Params, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            pred = mlp_forward(params, k, num_layers, act, res, nout)
            return ((pred - v) ** 2).sum()

        return grad(loss, argnums=0)

    def _project(self, x: torch.Tensor):
        """Return k, v, q and gate scalars alpha, eta, theta ([B,N]).

        ``k`` is the *feature-mapped* key ``phi(W_K x)`` (ATLAS Eq 56/58), so its
        width is ``p*d`` when a polynomial map is active; ``v`` and ``q`` keep
        width ``d`` (``q`` is mapped inside :meth:`retrieve`, since callers hand
        it a plain ``q_proj`` output).
        """
        k = self.k_proj(x)
        if self.normalize_qk:
            k = self._norm_qk(k)
        if self.feature_map is not None:
            k = self.feature_map(k)
        v = self.v_proj(x)
        q = self.q_proj(x)
        alpha = torch.sigmoid(self.alpha_head(x)).squeeze(-1)
        eta = torch.sigmoid(self.eta_head(x)).squeeze(-1)
        theta = F.softplus(self.theta_head(x)).squeeze(-1) * self.theta_scale
        return k, v, q, alpha, eta, theta

    def _norm_qk(self, z: torch.Tensor) -> torch.Tensor:
        """Normalise keys/queries; see the note in ``__init__`` on lambda_max."""
        if self.normalize_qk == "l2":
            return z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        return z / (z.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)

    @staticmethod
    def _bcast(coef: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-batch scalar ``coef`` [B] against leaf ``ref`` [B,...]."""
        return coef.view(coef.shape[0], *([1] * (ref.dim() - 1)))

    def _gamma(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Omega per-token pruning gates ``gamma_i`` ``[B, N]`` (``None`` if ``c=1``).

        Every path that runs the windowed objective must call this. ``forward_chunked``
        previously did not, so with ``c>1`` it summed the window *unweighted* and
        ``gamma_head`` received no gradient at all -- a parameter that existed, was
        counted, and did nothing.
        """
        if self.window_size <= 1:
            return None
        return torch.sigmoid(self.gamma_head(x)).squeeze(-1)

    def _step(
        self,
        state: Params,
        S: Params,
        g: Params,
        alpha_t: torch.Tensor,
        eta_t: torch.Tensor,
        theta_t: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
    ) -> Tuple[Params, Params]:
        """One momentum + weight-decay update (shared by both code paths).

        In cascade mode ``S`` carries the ``[..., K]`` reservoir state ``U`` and
        ``x_t`` (the token ``[B, dim]``) supplies the selective per-edge gate;
        otherwise the plain scalar recurrence runs unchanged.
        """
        new_S: Params = {}
        new_state: Params = {}
        if self.use_cascade_momentum and self.cascade.selective:
            if self.cascade.gate_signal == "surprise":
                # Raw surprise (Titans Sec 3.1: the memory's own loss gradient),
                # aggregated over every weight tensor and log-compressed so a
                # linear+sigmoid head can act as a stable learned threshold
                # regardless of how gradient scale drifts over training.
                sq = sum((gk.reshape(gk.shape[0], -1) ** 2).sum(dim=-1) for gk in g.values())
                surprise_t = torch.log1p(torch.sqrt(sq + 1e-12))  # [B]
                betas_t = self.cascade.selective_betas(surprise_t.unsqueeze(-1))
            else:
                betas_t = self.cascade.selective_betas(x_t)  # [B, K-1]
        else:
            betas_t = None
        for key, w in state.items():
            if not self.update_bias and key.startswith("b"):
                # Neither Titans nor ATLAS's formal M_t recurrence has a bias term
                # (every published update rule is a pure matrix/weight recurrence,
                # see e.g. Titans Eq 13-18/32-35, ATLAS Eq 45-58) -- biases are an
                # nn.Linear-convention addition of this implementation, subject to
                # the same online decay+momentum as weights. Keep them fixed at
                # their outer-trained value: no online update, no momentum state.
                new_state[key] = w
                new_S[key] = S[key]
                continue
            gk = g[key]
            if self.use_cascade_momentum:
                # Surprise term and the per-token eta_t leak; ``cascade_leak``
                # decides where the leak applies ("full" = whole state, the original;
                # "shallow" = only reservoir 0, so the slow reservoirs are NOT drained
                # and can provide long retention; "none" = no leak, the downstream
                # (1-alpha) forget gate supplies the only decay). topology decides how
                # the gradient is injected/mixed, then readout() feeds S_t.
                injected = -self._bcast(theta_t, gk) * gk
                eta_b = self._bcast(eta_t, S[key])
                if self.cascade_leak == "full":
                    u_prev = eta_b * S[key]
                elif self.cascade_leak == "none":
                    u_prev = S[key]
                elif self.cascade_leak == "slow":
                    # gentle fixed decay: retains long without ever saturating.
                    u_prev = self.cascade_slow_decay * S[key]
                elif self.cascade_leak == "surprise":
                    # event-driven: leak the old state in proportion to the CURRENT
                    # surprise (theta_t). Boring tokens (theta~0) barely leak; a
                    # surprising token displaces old memory -> turnover paced by
                    # events, not by every step (Benna-Fusi-like, avoids saturation).
                    u_prev = torch.exp(-self._bcast(theta_t, S[key])) * S[key]
                else:  # "shallow": leak reservoir 0 only (functional, no in-place)
                    r0 = eta_b * torch.ones_like(S[key][..., :1])
                    rest = torch.ones_like(S[key][..., 1:])
                    u_prev = torch.cat([r0, rest], dim=-1) * S[key]
                if self.cascade.topology == "independent":
                    # Diagonal B: same gradient into every reservoir, each with its
                    # own fixed decay, no cross-reservoir mixing.
                    new_S[key] = independent_step(
                        u_prev, injected, self.cascade.decays
                    )
                else:
                    # Chain: gradient into reservoir 0 only, tridiagonal diffusion.
                    if betas_t is not None:
                        b = betas_t.view(
                            betas_t.shape[0], *([1] * (w.dim() - 1)), betas_t.shape[-1]
                        )
                    else:
                        b = self.cascade.base_betas
                    new_S[key] = cascade_step(u_prev, injected, b)
                s_read = self.cascade.readout(new_S[key], x_t=x_t)
                if self.orthogonalize_momentum and w.dim() >= 3:
                    # ATLAS Eq 57 orthogonalises whatever plays the role of the
                    # visible momentum before it updates M. On the cascade path
                    # that is the readout, not S itself (S carries the [.., K]
                    # reservoir state, which has no weight-matrix structure to
                    # orthogonalise). Previously this branch ignored the flag
                    # entirely, so cascade+NS silently ran as plain cascade.
                    s_read = newton_schulz5(s_read)
                new_state[key] = self._bcast(1.0 - alpha_t, w) * w + s_read
            else:
                new_S[key] = (
                    self._bcast(eta_t, gk) * S[key] - self._bcast(theta_t, gk) * gk
                )
                update = new_S[key]
                if self.orthogonalize_momentum and w.dim() >= 3:
                    # Only 2D weight matrices (batched: [B, out, in]) have an
                    # orthogonality structure to speak of; bias vectors ([B, out])
                    # pass through untouched (and are usually frozen anyway when
                    # update_bias=False).
                    update = newton_schulz5(update)
                new_state[key] = self._bcast(1.0 - alpha_t, w) * w + update
        return new_state, new_S

    def retrieve(self, state: Params, q: torch.Tensor) -> torch.Tensor:
        """Forward pass of the memory at a given weight state (no update).

        ``q`` has shape ``[B, d]`` (a single query per batch element) or
        ``[B, T, d]`` (a whole segment of queries). ``mlp_forward`` handles the
        extra ``T`` dim inside each vmapped example, so both shapes work and the
        output mirrors ``q``'s shape. Weights in ``state`` carry a leading ``B``.

        This decouples the query stream from the update stream: ``state`` may have
        been produced by *other* tokens than the ones in ``q`` (used by MAC).

        The polynomial feature map is applied here rather than by the caller:
        every call site hands in a raw ``q_proj`` output, and ``M`` expects
        ``phi(q)`` whenever it was built for mapped keys.
        """
        if self.normalize_qk:
            q = self._norm_qk(q)
        if self.feature_map is not None:
            q = self.feature_map(q)
        return vmap(mlp_forward, in_dims=(0, 0, None, None, None, None))(
            state, q, self.num_layers, self.memory_activation, self.memory_residual,
            self.normalize_output
        )

    # ------------------------------------------------------------- carry state
    def init_momentum(self, state: Params) -> Params:
        """Zero momentum state ``S`` matching the shapes of ``state``.

        In cascade mode each leaf gains a trailing reservoir axis of size ``K``.
        """
        if self.use_cascade_momentum:
            return {key: self.cascade.init_state(w) for key, w in state.items()}
        return {key: torch.zeros_like(w) for key, w in state.items()}

    def init_memory(self, batch: int) -> Tuple[Params, Params]:
        """Fresh ``(state, S)`` carry for a new sequence (weights ``M_0``, ``S=0``).

        MAC threads this ``(state, S)`` pair across segments of a long sequence,
        so both the weights *and* the momentum must be carried, not just weights.
        """
        state = self.memory.init_state(batch)
        return state, self.init_momentum(state)

    def _scan_chunked(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        eta: torch.Tensor,
        theta: torch.Tensor,
        state: Params,
        S: Params,
        chunk_size: int,
        q: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> Tuple[Params, Params, Optional[list]]:
        """Shared chunked update recurrence (Section 3.2; ATLAS Omega for c>1).

        Computes, per chunk, all inner gradients at the chunk-start weights in one
        nested-``vmap`` call, then scans the momentum / weight-decay recurrence
        across the chunk. If ``q`` is given, retrieves ``M_t(q_t)`` at every step
        and returns the list of retrievals; otherwise skips retrieval entirely
        (the ``update``-only path, saving the wasted forward passes).

        For the Omega rule (``self.window_size = c > 1``) the gradient used at
        step ``t`` is the gated sliding-window sum
        ``sum_{i=t-c+1}^{t} gamma_i * grad_i`` -- valid because the gradient of a
        sum equals the sum of gradients, and every ``grad_i`` is already computed
        at the chunk-start weights. ``gamma`` holds the per-token gates ``[B, N]``.
        The window is clamped at chunk boundaries (a documented approximation);
        long-range flow across chunks still happens through the memory state.
        """
        grad_fn = self._grad_fn()
        # inner vmap: tokens in the chunk (params shared = chunk start);
        # outer vmap: the batch dimension B.
        chunk_grad = vmap(vmap(grad_fn, in_dims=(None, 0, 0)), in_dims=(0, 0, 0))

        c = self.window_size
        N = k.shape[1]
        ys: Optional[list] = [] if q is not None else None
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            G = chunk_grad(state, k[:, start:end], v[:, start:end])
            for j in range(end - start):
                t = start + j
                if c == 1:
                    g = {key: G[key][:, j] for key in state}
                else:
                    # Omega: decayed sum of per-token grads over the window,
                    # clamped to the current chunk (local indices lo..j).
                    lo = max(0, j - c + 1)
                    g = {}
                    # Optional: divide by the window's total gate mass, turning
                    # Eq 58's plain sum into a weighted average. The sum makes the
                    # effective inner step grow with c (measured: ||Delta M||_F
                    # 119 -> 190 -> 299 for c = 1, 4, 8), so a c>1 run changes the
                    # step size *and* the context at once -- the confound suspected
                    # of causing Omega's measured recall regression in July. Off by
                    # default: the sum is what the paper writes.
                    if self.window_normalize and gamma is not None:
                        mass = sum(gamma[:, start + i] for i in range(lo, j + 1))
                    else:
                        mass = None
                    for key in state:
                        Gk = G[key]
                        acc = None
                        for i in range(lo, j + 1):
                            gi = Gk[:, i]
                            if gamma is not None:
                                gi = self._bcast(gamma[:, start + i], gi) * gi
                            acc = gi if acc is None else acc + gi
                        if mass is not None:
                            acc = acc / self._bcast(mass + 1e-6, acc)
                        elif self.window_normalize:
                            acc = acc / (j - lo + 1)
                        g[key] = acc
                x_t = x[:, t] if x is not None else None
                state, S = self._step(
                    state, S, g, alpha[:, t], eta[:, t], theta[:, t], x_t=x_t
                )
                if ys is not None:
                    ys.append(self.retrieve(state, q[:, t]))
        return state, S, ys

    def update(
        self,
        x: torch.Tensor,
        state: Params,
        S: Params,
        chunk_size: int,
    ) -> Tuple[Params, Params]:
        """Evolve the memory on an input stream ``x`` *without* retrieving.

        Runs the same §3.2 chunked update recurrence as :meth:`forward_chunked`
        but computes no per-token retrievals -- the caller (MAC) only wants the
        evolved carry ``(state, S)``. ``x`` has shape ``[B, T, d]``; ``state`` and
        ``S`` are the incoming carry and are *not* re-initialised, so this can be
        called once per segment to thread memory across a long sequence.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if x.dim() != 3:
            raise ValueError("update expects x of shape [B, T, d]")
        k, v, _, alpha, eta, theta = self._project(x)
        gamma = self._gamma(x)
        state, S, _ = self._scan_chunked(
            k, v, alpha, eta, theta, state, S, chunk_size, q=None, gamma=gamma, x=x
        )
        return state, S

    # ------------------------------------------------------------- reference
    def forward_naive_sequential(
        self, x: torch.Tensor, chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Params]:
        """Ground-truth reference: explicit for-loop recurrence.

        With ``chunk_size=None`` (default) the inner gradient at step ``t`` is
        taken at the current weights ``M_{t-1}`` -- the fully-online recurrence.
        With an integer ``chunk_size`` the gradients within a chunk are all
        evaluated at the chunk's start weights (mini-batch semantics), which is
        what :meth:`forward_chunked` parallelizes.

        Returns ``(Y, final_state)`` with ``Y`` of shape ``x.shape[:-1] + (dim,)``.
        """
        orig_shape = x.shape
        d = orig_shape[-1]
        N = orig_shape[-2]
        B = x.numel() // (N * d)
        x = x.reshape(B, N, d)

        k, v, q, alpha, eta, theta = self._project(x)
        gamma = self._gamma(x)
        grad_fn = self._grad_fn()
        batched_grad = vmap(grad_fn, in_dims=(0, 0, 0))  # over batch B

        state = self.memory.init_state(B)
        S = self.init_momentum(state)
        anchor = state  # weights at which gradients are evaluated
        c = self.window_size

        ys = []
        for t in range(N):
            if chunk_size is not None and t % chunk_size == 0:
                anchor = state  # freeze at chunk start
            grad_state = state if chunk_size is None else anchor
            # Omega (ATLAS Eq 9): the gated sum over the last c tokens. Clamped to
            # the current chunk when chunking, exactly as ``_scan_chunked`` does, so
            # the two paths stay comparable; with ``chunk_size=None`` the window is
            # unclamped, which is the fully-online reading of the rule.
            lo = t - c + 1
            if chunk_size is not None:
                lo = max(lo, (t // chunk_size) * chunk_size)
            lo = max(lo, 0)
            g = None
            for i in range(lo, t + 1):
                gi = batched_grad(grad_state, k[:, i], v[:, i])
                if gamma is not None:
                    gi = {key: self._bcast(gamma[:, i], val) * val
                          for key, val in gi.items()}
                g = gi if g is None else {key: g[key] + gi[key] for key in gi}
            if self.window_normalize:
                mass = (sum(gamma[:, i] for i in range(lo, t + 1))
                        if gamma is not None else None)
                g = {key: (val / self._bcast(mass + 1e-6, val) if mass is not None
                           else val / (t - lo + 1)) for key, val in g.items()}
            state, S = self._step(
                state, S, g, alpha[:, t], eta[:, t], theta[:, t], x_t=x[:, t]
            )
            ys.append(self.retrieve(state, q[:, t]))

        Y = torch.stack(ys, dim=1)
        Y = Y.reshape(*orig_shape[:-1], self.dim)
        return Y, state

    # ------------------------------------------------------------- chunked
    def forward_chunked(
        self, x: torch.Tensor, chunk_size: int
    ) -> Tuple[torch.Tensor, Params]:
        """Parallel reformulation (Section 3.2).

        Within each chunk the inner gradients of all tokens are computed in a
        single batched (nested-vmap) call at the chunk-start weights, then the
        momentum / weight-decay recurrence (Eq. 18) is scanned across the chunk.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        orig_shape = x.shape
        d = orig_shape[-1]
        N = orig_shape[-2]
        B = x.numel() // (N * d)
        x = x.reshape(B, N, d)

        k, v, q, alpha, eta, theta = self._project(x)

        state, S = self.init_memory(B)
        state, _, ys = self._scan_chunked(
            k, v, alpha, eta, theta, state, S, chunk_size, q=q,
            gamma=self._gamma(x), x=x,
        )

        Y = torch.stack(ys, dim=1)
        Y = Y.reshape(*orig_shape[:-1], self.dim)
        return Y, state
