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
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

Params = Dict[str, torch.Tensor]


def mlp_forward(params: Params, x: torch.Tensor, num_layers: int) -> torch.Tensor:
    """Pure functional forward of the memory MLP for a *single* example.

    ``params`` holds ``w{i}`` of shape ``[out, in]`` and ``b{i}`` of shape
    ``[out]``. ``x`` has shape ``[in]``. SiLU is applied between layers.
    Designed to be composed with ``torch.func.grad`` / ``torch.func.vmap``.
    """
    h = x
    for i in range(num_layers):
        h = h @ params[f"w{i}"].transpose(-1, -2) + params[f"b{i}"]
        if i < num_layers - 1:
            h = F.silu(h)
    return h


class NeuralMemoryMLP(nn.Module):
    """The memory network ``M``: a small MLP, ``in_dim -> ... -> in_dim``.

    Only stores the *initial* (outer-trained) weights ``M_0``. The test-time
    update is carried out functionally by :class:`NeuralMemory`, so nothing here
    is mutated in place.
    """

    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Titans memory expects a deep MLP (num_layers >= 2).")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.params = nn.ParameterDict()
        for i in range(num_layers):
            in_d = dim if i == 0 else hidden_dim
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
        return mlp_forward(params, x, self.num_layers)


class NeuralMemory(nn.Module):
    """Neural long-term memory: outer projections + the online-updated MLP.

    Parameters
    ----------
    dim: token / key / value / query dimension (``d_in``).
    hidden_dim: hidden width of the memory MLP.
    num_layers: depth ``L_M`` of the memory MLP (>= 2).
    theta_scale: multiplies the softplus output for ``theta_t`` (inner lr).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        theta_scale: float = 1.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.theta_scale = theta_scale

        self.memory = NeuralMemoryMLP(dim, hidden_dim, num_layers)
        self.num_layers = num_layers

        # Outer, ordinarily-trained projections (shared across time).
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)

        # Data-dependent gate heads: scalar per token.
        self.alpha_head = nn.Linear(dim, 1)  # forget gate  -> sigmoid
        self.eta_head = nn.Linear(dim, 1)    # momentum decay -> sigmoid
        self.theta_head = nn.Linear(dim, 1)  # inner lr -> softplus

    # ------------------------------------------------------------------ utils
    def _loss(self, params: Params, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pred = mlp_forward(params, k, self.num_layers)
        return ((pred - v) ** 2).sum()

    def _grad_fn(self):
        num_layers = self.num_layers

        def loss(params: Params, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            pred = mlp_forward(params, k, num_layers)
            return ((pred - v) ** 2).sum()

        return grad(loss, argnums=0)

    def _project(self, x: torch.Tensor):
        """Return k, v, q ([B,N,d]) and gate scalars alpha, eta, theta ([B,N])."""
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_proj(x)
        alpha = torch.sigmoid(self.alpha_head(x)).squeeze(-1)
        eta = torch.sigmoid(self.eta_head(x)).squeeze(-1)
        theta = F.softplus(self.theta_head(x)).squeeze(-1) * self.theta_scale
        return k, v, q, alpha, eta, theta

    @staticmethod
    def _bcast(coef: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-batch scalar ``coef`` [B] against leaf ``ref`` [B,...]."""
        return coef.view(coef.shape[0], *([1] * (ref.dim() - 1)))

    def _step(
        self,
        state: Params,
        S: Params,
        g: Params,
        alpha_t: torch.Tensor,
        eta_t: torch.Tensor,
        theta_t: torch.Tensor,
    ) -> Tuple[Params, Params]:
        """One momentum + weight-decay update (shared by both code paths)."""
        new_S: Params = {}
        new_state: Params = {}
        for key, w in state.items():
            gk = g[key]
            new_S[key] = self._bcast(eta_t, gk) * S[key] - self._bcast(theta_t, gk) * gk
            new_state[key] = self._bcast(1.0 - alpha_t, w) * w + new_S[key]
        return new_state, new_S

    def retrieve(self, state: Params, q: torch.Tensor) -> torch.Tensor:
        """Forward pass of the memory at a given weight state (no update).

        ``q`` has shape ``[B, d]`` (a single query per batch element) or
        ``[B, T, d]`` (a whole segment of queries). ``mlp_forward`` handles the
        extra ``T`` dim inside each vmapped example, so both shapes work and the
        output mirrors ``q``'s shape. Weights in ``state`` carry a leading ``B``.

        This decouples the query stream from the update stream: ``state`` may have
        been produced by *other* tokens than the ones in ``q`` (used by MAC).
        """
        return vmap(mlp_forward, in_dims=(0, 0, None))(state, q, self.num_layers)

    # ------------------------------------------------------------- carry state
    def init_momentum(self, state: Params) -> Params:
        """Zero momentum state ``S`` matching the shapes of ``state``."""
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
    ) -> Tuple[Params, Params, Optional[list]]:
        """Shared chunked update recurrence (Section 3.2).

        Computes, per chunk, all inner gradients at the chunk-start weights in one
        nested-``vmap`` call, then scans the momentum / weight-decay recurrence
        across the chunk. If ``q`` is given, retrieves ``M_t(q_t)`` at every step
        and returns the list of retrievals; otherwise skips retrieval entirely
        (the ``update``-only path, saving the wasted forward passes).
        """
        grad_fn = self._grad_fn()
        # inner vmap: tokens in the chunk (params shared = chunk start);
        # outer vmap: the batch dimension B.
        chunk_grad = vmap(vmap(grad_fn, in_dims=(None, 0, 0)), in_dims=(0, 0, 0))

        N = k.shape[1]
        ys: Optional[list] = [] if q is not None else None
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            G = chunk_grad(state, k[:, start:end], v[:, start:end])
            for j in range(end - start):
                t = start + j
                g = {key: G[key][:, j] for key in state}
                state, S = self._step(
                    state, S, g, alpha[:, t], eta[:, t], theta[:, t]
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
        state, S, _ = self._scan_chunked(
            k, v, alpha, eta, theta, state, S, chunk_size, q=None
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
        grad_fn = self._grad_fn()
        batched_grad = vmap(grad_fn, in_dims=(0, 0, 0))  # over batch B

        state = self.memory.init_state(B)
        S = {key: torch.zeros_like(w) for key, w in state.items()}
        anchor = state  # weights at which gradients are evaluated

        ys = []
        for t in range(N):
            if chunk_size is not None and t % chunk_size == 0:
                anchor = state  # freeze at chunk start
            grad_state = state if chunk_size is None else anchor
            g = batched_grad(grad_state, k[:, t], v[:, t])
            state, S = self._step(state, S, g, alpha[:, t], eta[:, t], theta[:, t])
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
            k, v, alpha, eta, theta, state, S, chunk_size, q=q
        )

        Y = torch.stack(ys, dim=1)
        Y = Y.reshape(*orig_shape[:-1], self.dim)
        return Y, state
