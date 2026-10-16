"""Neural long-term memory (Titans, arXiv:2501.00663), with the ATLAS additions.

The memory ``M`` is a small MLP whose *weights* are the online-learned state. At
every token ``t``:

  1. project: ``k_t = W_K x_t``, ``v_t = W_V x_t``, ``q_t = W_Q x_t``, and the gates
     ``alpha_t`` (forget), ``eta_t`` (momentum), ``theta_t`` (step) from linear heads;
  2. take the gradient ``g_t`` of the inner loss ``||M(k_t) - v_t||^2``;
  3. write, with one of three rules (``orthogonalize_momentum``)::

         False         S_t = eta_t S_{t-1} - theta_t g_t     M_t = (1-alpha_t) M_{t-1} + S_t
         "unit"        S_t = eta_t S_{t-1} - g_t             M_t = (1-alpha_t) M_{t-1} + theta_t NS5(S_t)
         "unit_squash" S_t = eta_t S_{t-1} - g_t             M_t = (1-alpha_t) M_{t-1} + theta_t psi_c(||S_t||) NS5(S_t)

     ``False`` is Titans literally (unbounded, diverges). ``"unit"`` is ATLAS Eq 32
     (``||dM||_2 <= theta_t``, collapses). ``"unit_squash"`` keeps that bound and
     makes the amplitude grow with ``||S_t||``: ``psi_c(n) = (2/pi) arctan(pi n / 2c)``.
  4. retrieve ``y_t = M_t(q_t)``.

Chunked parallel form (Titans §3.2): inside a chunk every gradient is taken at the
chunk-start weights, then the recurrence is scanned. :meth:`NeuralMemory.forward_naive_sequential`
is the explicit for-loop reference; with ``chunk_size=None`` it is fully online.

ATLAS additions: the Omega objective (``window_size = c > 1``, a gated sum of the
last ``c`` per-token gradients, Eq 9) and the polynomial feature map (``feature_degree``).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

from .features import PolynomialFeatureMap

Params = Dict[str, torch.Tensor]

WRITE_RULES = (False, "unit", "unit_squash")


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz (Muon, ATLAS Eq 57): singular values of ``G`` -> ~1.

    ``G`` is ``[B, out, in]``; the singular vectors are kept, the magnitude is not,
    so ``||NS5(G)||_2 ~ 1`` whatever ``G`` was. Anything with fewer than 3 dims (a
    batched bias) passes through unchanged.
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
    return X


def mlp_forward(
    params: Params,
    x: torch.Tensor,
    num_layers: int,
    activation: str = "silu",
    residual: bool = False,
    normalize_output: bool = True,
    normalize_hidden: "bool | float" = False,
) -> torch.Tensor:
    """Functional forward of the memory MLP for a *single* example (vmap/grad-able).

    ``params`` holds ``w{i}`` ``[out, in]`` and ``b{i}`` ``[out]``.

    * ``activation``: ``"silu"`` or ``"gelu"`` (ATLAS Table 7).
    * ``residual``: ATLAS's ``M(x) = x + W_1 sigma(W_2 x)``. When a feature map has
      widened the input, the skip adds its leading ``d`` slice, i.e. phi's degree-1
      block = the un-mapped key (the paper leaves this case unspecified).
    * ``normalize_output``: parameter-free RMSNorm of the output. Not in ATLAS, and
      it breaks the scale match with ``v`` (the inner loss then carries almost no
      information about ``v``); ``sane_ns`` turns it off.
    * ``normalize_hidden``: L2-normalise every hidden activation to the given norm
      (``True`` = 1). For the 2-layer memory this pins ``||h_0||`` and therefore the
      last layer's Gauss-Newton curvature ``2 ||h_0||^2``.
    """
    h = x
    for i in range(num_layers):
        h = h @ params[f"w{i}"].transpose(-1, -2) + params[f"b{i}"]
        if i < num_layers - 1:
            h = F.gelu(h) if activation == "gelu" else F.silu(h)
            if normalize_hidden:
                target = 1.0 if normalize_hidden is True else float(normalize_hidden)
                h = target * h / (h.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    if residual and x.shape[-1] >= h.shape[-1]:
        h = h + (x if h.shape[-1] == x.shape[-1] else x[..., : h.shape[-1]])
    if not normalize_output:
        return h
    rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return h / rms


class NeuralMemoryMLP(nn.Module):
    """The memory network ``M``. Stores only the outer-trained initial weights ``M_0``;
    the test-time state is carried functionally by :class:`NeuralMemory`."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        in_dim: Optional[int] = None,
        activation: str = "silu",
        residual: bool = False,
        normalize_output: bool = True,
        normalize_hidden: "bool | float" = False,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Titans memory expects a deep MLP (num_layers >= 2).")
        self.dim = dim
        self.in_dim = in_dim or dim          # p*d when a feature map widens the keys
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.residual = residual
        self.normalize_output = normalize_output
        self.normalize_hidden = normalize_hidden

        self.params = nn.ParameterDict()
        for i in range(num_layers):
            in_d = self.in_dim if i == 0 else hidden_dim
            out_d = dim if i == num_layers - 1 else hidden_dim
            w = torch.empty(out_d, in_d)
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
            self.params[f"w{i}"] = nn.Parameter(w)
            self.params[f"b{i}"] = nn.Parameter(torch.zeros(out_d))

    def init_state(self, batch: int) -> Params:
        """A fresh per-sequence weight state with a leading batch dim."""
        return {
            k: v.unsqueeze(0).expand(batch, *v.shape).clone()
            for k, v in self.params.items()
        }

    def forward(self, params: Params, x: torch.Tensor) -> torch.Tensor:
        return mlp_forward(params, x, self.num_layers, self.activation, self.residual,
                           self.normalize_output, self.normalize_hidden)


class NeuralMemory(nn.Module):
    """Neural long-term memory: outer projections, gate heads, and the online MLP.

    Parameters
    ----------
    dim: token / key / value / query width.
    hidden_dim: hidden width of the memory MLP (``memory_expansion * dim`` if given).
    num_layers: depth of the memory MLP (>= 2).
    theta_scale: multiplies the step gate ``theta_t``.
    theta_gate: ``"softplus"`` (Titans, unbounded) or ``"sigmoid"`` (capped at
        ``2 log 2 * theta_scale``, equal to softplus at init).
    window_size: ATLAS Omega window ``c``; ``1`` is the Titans delta rule.
    window_stride, window_cross_chunk: ``False`` clamps the window at the chunk start
        (widths ``1..m``); ``True`` is a real sliding window whose earlier tokens are
        re-differentiated at the current chunk-start weights. With a stride ``s`` the
        window is ``t`` plus the last ``c-1`` positions ``p < t`` with ``p % s == 0``.
    update_bias: ``False`` keeps M's biases at their outer-trained value (neither
        paper's recurrence has a bias term).
    orthogonalize_momentum: the write rule, one of :data:`WRITE_RULES` (module doc).
    orthogonalize_clip: the scale ``c`` of ``psi_c`` under ``"unit_squash"``.
    feature_degree, feature_normalize: ATLAS polynomial feature map on keys/queries.
    memory_expansion, memory_activation, memory_residual: ATLAS Table 7 MLP recipe.
    normalize_qk: ``False``, ``"rms"`` or ``"l2"`` on keys and queries (ATLAS p12).
        ``"l2"`` gives ``lambda_max(K K^T) ~ m`` per chunk, hence a usable step.
    normalize_output, normalize_hidden: see :func:`mlp_forward`.
    alpha_init, eta_init: resting values of the forget and momentum gates (head bias).
    rank_cap: probe only, caps the online write of ``w1`` to a fixed rank-r key subspace.

    Read-time instruments, set as attributes on a loaded model (never in a cfg):
    ``alpha_override``, ``eta_override``, ``theta_override`` pin the gate to a constant.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        theta_scale: float = 1.0,
        theta_gate: str = "softplus",
        window_size: int = 1,
        window_stride: int = 1,
        window_cross_chunk: bool = False,
        update_bias: bool = True,
        orthogonalize_momentum: "bool | str" = False,
        orthogonalize_clip: float = 1.0,
        feature_degree: int = 1,
        feature_normalize: bool = True,
        memory_expansion: Optional[int] = None,
        memory_activation: str = "silu",
        memory_residual: bool = False,
        normalize_qk: "bool | str" = False,
        normalize_output: bool = True,
        normalize_hidden: "bool | float" = False,
        alpha_init: Optional[float] = None,
        eta_init: Optional[float] = None,
        rank_cap: Optional[int] = None,
    ):
        super().__init__()
        if memory_expansion is not None:
            if memory_expansion < 1:
                raise ValueError("memory_expansion must be >= 1")
            hidden_dim = dim * memory_expansion
        hidden_dim = hidden_dim or dim
        if memory_activation not in ("silu", "gelu"):
            raise ValueError("memory_activation must be 'silu' or 'gelu'")
        self.memory_activation = memory_activation
        self.memory_residual = memory_residual
        if normalize_qk is True:
            normalize_qk = "rms"
        if normalize_qk not in (False, None, "rms", "l2"):
            raise ValueError("normalize_qk must be False, 'rms' or 'l2'")
        self.normalize_qk = normalize_qk or False
        self.normalize_output = normalize_output
        self.normalize_hidden = normalize_hidden
        self.dim = dim
        self.theta_scale = theta_scale
        if theta_gate not in ("softplus", "sigmoid"):
            raise ValueError("theta_gate must be 'softplus' or 'sigmoid'")
        self.theta_gate = theta_gate
        self.feature_degree = feature_degree
        self.feature_map = (PolynomialFeatureMap(feature_degree, normalize=feature_normalize)
                            if feature_degree > 1 else None)
        mem_in_dim = dim * feature_degree if feature_degree > 1 else dim
        if window_size < 1:
            raise ValueError("window_size (Omega c) must be >= 1")
        if window_stride < 1:
            raise ValueError("window_stride must be >= 1")
        if window_stride > 1 and not window_cross_chunk:
            raise ValueError("window_stride > 1 needs window_cross_chunk=True")
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_cross_chunk = window_cross_chunk
        self.update_bias = update_bias
        if orthogonalize_momentum not in WRITE_RULES:
            raise ValueError(f"orthogonalize_momentum must be one of {WRITE_RULES}, "
                             f"got {orthogonalize_momentum!r}")
        self.orthogonalize_momentum = orthogonalize_momentum
        self._orth_mode = orthogonalize_momentum
        self.orthogonalize_clip = orthogonalize_clip

        self.memory = NeuralMemoryMLP(dim, hidden_dim, num_layers, in_dim=mem_in_dim,
                                      activation=memory_activation,
                                      residual=memory_residual,
                                      normalize_output=normalize_output,
                                      normalize_hidden=normalize_hidden)
        self.num_layers = num_layers

        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_head = nn.Linear(dim, 1)   # forget gate   -> sigmoid
        self.eta_head = nn.Linear(dim, 1)     # momentum gate -> sigmoid
        self.theta_head = nn.Linear(dim, 1)   # step gate     -> softplus | sigmoid
        if window_size > 1:
            self.gamma_head = nn.Linear(dim, 1)   # Omega per-token gate -> sigmoid

        # Resting gate values via the head biases. sigmoid(0) = 0.5 would give the
        # forget gate a one-token half-life and the momentum a ~2-token horizon.
        for name, init, head in (("alpha_init", alpha_init, self.alpha_head),
                                 ("eta_init", eta_init, self.eta_head)):
            if init is not None:
                if not 0.0 < init < 1.0:
                    raise ValueError(f"{name} must be in (0, 1)")
                with torch.no_grad():
                    head.bias.fill_(math.log(init / (1 - init)))

        self.alpha_override: Optional[float] = None
        self.eta_override: Optional[float] = None
        self.theta_override: Optional[float] = None
        self.rank_cap = rank_cap
        self._rank_bases: dict = {}

    # ------------------------------------------------------------------ utils
    def _grad_fn(self):
        num_layers = self.num_layers
        act, res = self.memory_activation, self.memory_residual
        nout, nhid = self.normalize_output, self.normalize_hidden

        def loss(params: Params, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            pred = mlp_forward(params, k, num_layers, act, res, nout, nhid)
            return ((pred - v) ** 2).sum()

        return grad(loss, argnums=0)

    def _unit_update(self, raw: torch.Tensor) -> torch.Tensor:
        """``NS5(S)`` under ``"unit"``, ``psi_c(||S||) NS5(S)`` under ``"unit_squash"``.

        ``psi_c`` lies in ``[0, 1)``, so both rules share the bound ``||dM||_2 <= theta_t``;
        ``unit`` is the ``c -> 0`` limit.
        """
        u = newton_schulz5(raw)
        if self._orth_mode != "unit_squash":
            return u
        c = self.orthogonalize_clip
        n = raw.norm(dim=(-2, -1), keepdim=True) if raw.dim() >= 3 else raw.norm(
            dim=-1, keepdim=True)
        return u * (2.0 / math.pi) * torch.atan(n * (math.pi / (2.0 * c)))

    def _project(self, x: torch.Tensor):
        """``k, v, q`` and the gates ``alpha, eta, theta`` ``[B, N]``.

        ``k`` is the feature-mapped key ``phi(W_K x)``; ``q`` is mapped later, in
        :meth:`retrieve`.
        """
        k = self.k_proj(x)
        if self.normalize_qk:
            k = self._norm_qk(k)
        if self.feature_map is not None:
            k = self.feature_map(k)
        v = self.v_proj(x)
        q = self.q_proj(x)
        alpha = torch.sigmoid(self.alpha_head(x)).squeeze(-1)
        if self.alpha_override is not None:
            alpha = torch.full_like(alpha, self.alpha_override)
        eta = torch.sigmoid(self.eta_head(x)).squeeze(-1)
        if self.eta_override is not None:
            eta = torch.full_like(eta, self.eta_override)
        theta = self._theta(self.theta_head(x)).squeeze(-1) * self.theta_scale
        if self.theta_override is not None:
            theta = torch.full_like(theta, self.theta_override)
        return k, v, q, alpha, eta, theta

    #: ``2 * softplus(0)``: the sigmoid gate starts exactly where softplus does.
    _SIGMOID_GAIN = 2.0 * math.log(2.0)

    def _theta(self, h: torch.Tensor) -> torch.Tensor:
        if self.theta_gate == "sigmoid":
            return self._SIGMOID_GAIN * torch.sigmoid(h)
        return F.softplus(h)

    def _norm_qk(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize_qk == "l2":
            return z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        return z / (z.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)

    @staticmethod
    def _bcast(coef: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-batch scalar ``coef`` [B] against leaf ``ref`` [B,...]."""
        return coef.view(coef.shape[0], *([1] * (ref.dim() - 1)))

    def _gamma(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Omega per-token gates ``gamma_i`` ``[B, N]`` (``None`` when ``c = 1``)."""
        if self.window_size <= 1:
            return None
        return torch.sigmoid(self.gamma_head(x)).squeeze(-1)

    def _window_idx(self, t: int, chunk_start: int) -> list:
        """Positions of the Omega window of token ``t`` (ascending, ``t`` last)."""
        c, s = self.window_size, self.window_stride
        if not self.window_cross_chunk:
            return list(range(max(t - c + 1, chunk_start, 0), t + 1))
        if c == 1:
            return [t]
        last = ((t - 1) // s) * s                       # last grid point < t
        grid = [p for p in range(last - (c - 2) * s, last + 1, s) if p >= 0]
        return grid + [t]

    def _rank_project(self, A: torch.Tensor, r: int) -> torch.Tensor:
        """Project the in-side of each ``[B, out, in]`` matrix onto a fixed r-dim subspace."""
        in_d = A.shape[-1]
        if r >= in_d:
            return A
        key = (r, in_d, A.device, A.dtype)
        proj = self._rank_bases.get(key)
        if proj is None:
            g = torch.Generator(device=A.device).manual_seed(1234 + r)
            M = torch.randn(in_d, r, generator=g, device=A.device, dtype=A.dtype)
            B, _ = torch.linalg.qr(M)
            proj = B @ B.transpose(-2, -1)
            self._rank_bases[key] = proj
        return A @ proj

    def _rank_w0(self, key: str, like: torch.Tensor) -> torch.Tensor:
        """Cached ``W_0[key]``: the rank cap bounds the write ``W - W_0``."""
        ck = ("w0", key, like.shape[0], like.device, like.dtype)
        w0 = self._rank_bases.get(ck)
        if w0 is None:
            w0 = self.memory.init_state(like.shape[0])[key].to(
                device=like.device, dtype=like.dtype)
            self._rank_bases[ck] = w0
        return w0

    def _step(
        self,
        state: Params,
        S: Params,
        g: Params,
        alpha_t: torch.Tensor,
        eta_t: torch.Tensor,
        theta_t: torch.Tensor,
    ) -> Tuple[Params, Params]:
        """One write (shared by every code path); the rules are in the module doc."""
        new_S: Params = {}
        new_state: Params = {}
        for key, w in state.items():
            if not self.update_bias and key.startswith("b"):
                new_state[key] = w
                new_S[key] = S[key]
                continue
            gk = g[key]
            if self._orth_mode:
                new_S[key] = self._bcast(eta_t, gk) * S[key] - gk
                update = self._bcast(theta_t, gk) * self._unit_update(new_S[key])
            else:
                new_S[key] = (
                    self._bcast(eta_t, gk) * S[key] - self._bcast(theta_t, gk) * gk
                )
                update = new_S[key]
            new_state[key] = self._bcast(1.0 - alpha_t, w) * w + update
        if self.rank_cap is not None and "w1" in new_state:
            w0 = self._rank_w0("w1", new_state["w1"])
            new_state["w1"] = w0 + self._rank_project(new_state["w1"] - w0, self.rank_cap)
        return new_state, new_S

    def retrieve(self, state: Params, q: torch.Tensor) -> torch.Tensor:
        """``M_state(q)`` without updating. ``q`` is ``[B, d]`` or ``[B, T, d]`` (raw
        ``q_proj`` output; the qk norm and feature map are applied here)."""
        if self.normalize_qk:
            q = self._norm_qk(q)
        if self.feature_map is not None:
            q = self.feature_map(q)
        return vmap(mlp_forward, in_dims=(0, 0, None, None, None, None, None))(
            state, q, self.num_layers, self.memory_activation, self.memory_residual,
            self.normalize_output, self.normalize_hidden
        )

    # ------------------------------------------------------------- carry state
    def init_momentum(self, state: Params) -> Params:
        return {key: torch.zeros_like(w) for key, w in state.items()}

    def init_memory(self, batch: int) -> Tuple[Params, Params]:
        """Fresh ``(state, S)`` carry: weights ``M_0`` and zero momentum."""
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
    ) -> Tuple[Params, Params, Optional[list]]:
        """Chunked recurrence: per chunk, every gradient in one nested-vmap call at the
        chunk-start weights, then the write scanned token by token. With ``q``,
        retrieves ``M_t(q_t)`` after every write and returns the list."""
        chunk_grad = vmap(vmap(self._grad_fn(), in_dims=(None, 0, 0)), in_dims=(0, 0, 0))
        c = self.window_size
        N = k.shape[1]
        ys: Optional[list] = [] if q is not None else None
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            G = chunk_grad(state, k[:, start:end], v[:, start:end])
            old: dict = {}
            if c > 1 and self.window_cross_chunk:
                need = sorted({p for t in range(start, end)
                               for p in self._window_idx(t, start) if p < start})
                if need:
                    ix = torch.tensor(need, device=k.device)
                    Go = chunk_grad(state, k[:, ix], v[:, ix])
                    old = {p: n for n, p in enumerate(need)}
            for j in range(end - start):
                t = start + j
                if c == 1:
                    g = {key: G[key][:, j] for key in state}
                else:
                    win = self._window_idx(t, start)
                    g = {}
                    for key in state:
                        acc = None
                        for p in win:
                            gi = Go[key][:, old[p]] if p < start else G[key][:, p - start]
                            if gamma is not None:
                                gi = self._bcast(gamma[:, p], gi) * gi
                            acc = gi if acc is None else acc + gi
                        g[key] = acc
                state, S = self._step(state, S, g, alpha[:, t], eta[:, t], theta[:, t])
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
        """Evolve the carry ``(state, S)`` on ``x`` ``[B, T, d]`` without retrieving."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if x.dim() != 3:
            raise ValueError("update expects x of shape [B, T, d]")
        k, v, _, alpha, eta, theta = self._project(x)
        state, S, _ = self._scan_chunked(k, v, alpha, eta, theta, state, S, chunk_size,
                                         q=None, gamma=self._gamma(x))
        return state, S

    # ------------------------------------------------------------- reference
    def forward_naive_sequential(
        self, x: torch.Tensor, chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Params]:
        """Explicit for-loop reference. ``chunk_size=None``: gradients at the current
        weights (fully online); an integer: at the chunk-start weights, which is what
        :meth:`forward_chunked` parallelises. Returns ``(Y, final_state)``."""
        orig_shape = x.shape
        d = orig_shape[-1]
        N = orig_shape[-2]
        B = x.numel() // (N * d)
        x = x.reshape(B, N, d)

        k, v, q, alpha, eta, theta = self._project(x)
        gamma = self._gamma(x)
        batched_grad = vmap(self._grad_fn(), in_dims=(0, 0, 0))

        state = self.memory.init_state(B)
        S = self.init_momentum(state)
        anchor = state
        c = self.window_size
        ys = []
        for t in range(N):
            if chunk_size is not None and t % chunk_size == 0:
                anchor = state
            grad_state = state if chunk_size is None else anchor
            cs = (t // chunk_size) * chunk_size if chunk_size is not None else 0
            if chunk_size is None and not self.window_cross_chunk:
                win = list(range(max(t - c + 1, 0), t + 1))
            else:
                win = self._window_idx(t, cs)
            g = None
            for i in win:
                gi = batched_grad(grad_state, k[:, i], v[:, i])
                if gamma is not None:
                    gi = {key: self._bcast(gamma[:, i], val) * val
                          for key, val in gi.items()}
                g = gi if g is None else {key: g[key] + gi[key] for key in gi}
            state, S = self._step(state, S, g, alpha[:, t], eta[:, t], theta[:, t])
            ys.append(self.retrieve(state, q[:, t]))

        Y = torch.stack(ys, dim=1).reshape(*orig_shape[:-1], self.dim)
        return Y, state

    # ------------------------------------------------------------- chunked
    def forward_chunked(
        self, x: torch.Tensor, chunk_size: int
    ) -> Tuple[torch.Tensor, Params]:
        """Parallel form (Titans §3.2): retrieve after every write, fresh carry."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        orig_shape = x.shape
        d = orig_shape[-1]
        N = orig_shape[-2]
        B = x.numel() // (N * d)
        x = x.reshape(B, N, d)

        k, v, q, alpha, eta, theta = self._project(x)
        state, S = self.init_memory(B)
        state, _, ys = self._scan_chunked(k, v, alpha, eta, theta, state, S, chunk_size,
                                          q=q, gamma=self._gamma(x))
        Y = torch.stack(ys, dim=1).reshape(*orig_shape[:-1], self.dim)
        return Y, state
