"""Metaplastic K-reservoir momentum cascade (opt-in momentum for NeuralMemory).

Motivation
----------
Titans' neural memory carries a single momentum scalar per weight,
``S_t = eta_t S_{t-1} - theta_t grad_t`` (then ``M_t = (1-alpha_t) M_{t-1} + S_t``).
That is *one* degree of freedom -- the decay ``eta_t`` sets both how fast the
memory reacts to new gradients and how long it retains them, the classic
plasticity-stability tradeoff. Biological synapses instead use a **cascade** of
internal states of decreasing plasticity coupled by bidirectional diffusion
(Fusi/Abbott/Drew 2005; Benna & Fusi 2016, Nat. Neurosci.): only the shallowest
reservoir sees new events; deeper, slower reservoirs fill by diffusion and can
flow back to stabilise the shallow one against noise. This decouples reaction
speed from retention and yields (over an intermediate horizon set by the number
of reservoirs) power-law-like forgetting instead of a single exponential.

This module replaces the scalar ``S`` by a length-``K`` vector
``u = (u^(1), ..., u^(K))`` **per weight entry**. Only ``u^(1)`` receives the
gradient; consecutive reservoirs exchange mass by diffusion::

    u_t^(1) = u_{t-1}^(1) + grad_t + b_2 (u_{t-1}^(2) - u_{t-1}^(1))
    u_t^(k) = u_{t-1}^(k) + b_k (u_{t-1}^(k-1) - u_{t-1}^(k))
                          + b_{k+1} (u_{t-1}^(k+1) - u_{t-1}^(k))      (1<k<K)
    u_t^(K) = u_{t-1}^(K) + b_K (u_{t-1}^(K-1) - u_{t-1}^(K))

i.e. ``u_t = B u_{t-1} + e_1 grad_t`` with ``B`` tridiagonal (``K x K``). ``B``
mixes only the ``K`` depths of one scalar's own cascade -- never different weight
entries of ``M``. The "visible" momentum that plugs into the (unchanged)
``M_t = (1-alpha_t) M_{t-1} + read(u_t)`` rule is :func:`read`, i.e. ``u^(1)``.

Shape convention
----------------
``K`` is the **last** axis. If the ordinary momentum ``S`` matches a memory
weight of shape ``[B, d_out, d_in]``, the cascade state ``U`` has shape
``[B, d_out, d_in, K]``. The ``K-1`` edge couplings ``(b_2, ..., b_K)`` are held
as a vector of length ``K-1`` (edge ``e`` connects reservoirs ``e`` and ``e+1``,
0-indexed) broadcastable over the leading (weight/batch) dims.

Topology
--------
The default cascade is a **chain**: gradient injected into reservoir 0 only,
tridiagonal (nearest-neighbour) diffusion between depths. An alternative
``topology="independent"`` realises the degenerate **diagonal-``B``** case
directly (see :func:`independent_step` / :func:`geometric_decays`): ``K``
uncoupled exponential filters, the gradient injected into *all* of them, readout
= mean over reservoirs. This is the honest "would any K-way mixture of
exponential filters do as well?" null hypothesis for the coupled cascade -- the
provably-less-expressive filter bank named under mode 1 below, but built
explicitly rather than emerging from a fixed chain.

Coupling modes
--------------
1. **Fixed geometric** ``B`` (:func:`geometric_betas`): ``b_k = beta1 * g^-(k-1)``
   for ``g > 1`` (coupling weakens with depth), plain hyperparameters. This is
   what the analytical diagnostic uses. NOTE: with a *fixed* ``B`` the cascade is
   diagonalisable ``B = P Lambda P^-1`` and is therefore exactly a fixed linear
   combination of ``K`` independent exponential-momentum filters -- no more
   expressive than several ordinary Titans momentums mixed by a learned readout.
   ``topology="independent"`` is exactly this filter bank in its already-diagonal
   form (``B = diag(decays)``, ``decays`` from :func:`geometric_decays`).
2. **Data-dependent / selective** ``B_t`` (:class:`MetaplasticCascade` with
   ``selective=True``): a learned ``Linear(dim, K-1)`` + sigmoid head on the
   token ``x_t`` (same linear+sigmoid style as ``alpha_head``/``eta_head`` in
   ``NeuralMemory``, but one gate *per edge*) emits per-token gates in ``(0, 1)``
   that scale the base geometric couplings edge-by-edge. This matters: only when
   ``B_t`` actually varies with the data so successive ``B_t`` do *not* commute
   can the cascade do asymmetric, event-triggered consolidation -- behaviour a
   fixed linear filter bank cannot reproduce.

   Design note (deviation from a naive reading of "a per-token *scalar*"): a
   single global scalar ``s_t`` scaling *all* couplings uniformly gives
   ``B_t = I + s_t D`` for a fixed diffusion Laplacian ``D``. All such ``B_t``
   share ``D``'s eigenvectors, so they commute and are simultaneously
   diagonalisable in one fixed basis -- i.e. still exactly a bank of ``K``
   exponential filters with only time-varying rates, *no more expressive* than
   the fixed cascade. A **per-edge** gate changes the *shape* of ``D_t`` (not
   just a global scale), making the ``B_t`` genuinely non-commuting, which is the
   whole point of the data-dependent mode.

Stability of ``B`` (checked, not assumed)
-----------------------------------------
``B`` is symmetric and row-stochastic (each row sums to 1, entries >= 0 as long
as ``b_k + b_{k+1} <= 1``), hence doubly stochastic: **all eigenvalues lie in
(-1, 1] with the top one exactly 1** (the conserved total mass of a closed
diffusion). So, unlike the geometric per-segment blow-up we hit in ``memory.py``,
there is *no* eigenvalue of modulus > 1 -- the cascade cannot grow geometrically.
The caveat is the eigenvalue exactly 1: reservoir 1 has no decay of its own
(only outflow), so under a sustained run of same-sign gradients the conserved
total -- and thus ``u^(1)`` -- grows at most **linearly** in sequence length,
not geometrically. Downstream, the memory's own forget gate ``(1 - alpha_t)``
provides the missing leak. Verified numerically for the defaults ``K=6,
beta1=0.5, g=2``: eigenvalues ``[0.40, 0.77, 0.90, 0.96, 0.99, 1.00]`` (spectral
radius 1, all positive -> comfortably away from the -1 boundary). Keep
``beta1 * (1 + 1/g) <= 1`` so the shallowest diagonal ``1 - b_2 - b_3`` stays
non-negative.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Coupling construction / spectral diagnostics (fixed geometric B).
# --------------------------------------------------------------------------- #


def geometric_betas(
    num_reservoirs: int,
    beta1: float = 0.5,
    g: float = 2.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Edge couplings ``(b_2, ..., b_K)`` for a geometric cascade.

    ``b_k = beta1 * g^-(k-1)`` for ``k = 2..K`` (task formula), so the returned
    vector has length ``K-1``: ``[beta1/g, beta1/g^2, ..., beta1/g^(K-1)]``.
    Coupling weakens with depth (``g > 1``), giving reservoirs of geometrically
    increasing time constants. Defaults are chosen to keep ``B`` comfortably
    stable (see the module docstring's stability note).
    """
    if num_reservoirs < 1:
        raise ValueError("num_reservoirs (K) must be >= 1")
    if g <= 1.0:
        raise ValueError("geometric spacing factor g must be > 1")
    ks = torch.arange(1, num_reservoirs, dtype=dtype, device=device)  # 1..K-1
    return beta1 * g ** (-ks)


def geometric_decays(
    num_reservoirs: int,
    beta1: float = 0.5,
    g: float = 2.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-reservoir self-decays for the **independent** (diagonal-``B``) topology.

    Returns a length-``K`` vector ``lambda_k = 1 - beta1 * g^-k`` for
    ``k = 0..K-1`` (so ``lambda_0 = 1 - beta1``, then geometrically approaching 1
    as depth grows): ``[1-beta1, 1-beta1/g, 1-beta1/g^2, ...]``. These are the
    fixed decay rates of ``K`` *independent* exponential-momentum filters -- the
    diagonal-``B`` degenerate case documented under "Coupling modes / fixed
    coupling" in the module docstring, used as the sweep's null hypothesis
    ("would any K-way mixture of exponential filters do as well as the coupled
    cascade?"). Reuses the same ``(beta1, g)`` geometric spacing as the chain's
    edge couplings so the two configs share one hyperparameter family; concretely
    these mirror the *eigenvalues* of the fixed tridiagonal ``B`` (a fixed cascade
    is exactly such a filter bank -- see the note on mode 1). With ``beta1=0.5,
    g=2`` and ``K=4`` this is ``[0.5, 0.75, 0.875, 0.9375]``, all in ``(0, 1)``.
    """
    if num_reservoirs < 1:
        raise ValueError("num_reservoirs (K) must be >= 1")
    if g <= 1.0:
        raise ValueError("geometric spacing factor g must be > 1")
    ks = torch.arange(0, num_reservoirs, dtype=dtype, device=device)  # 0..K-1
    return 1.0 - beta1 * g ** (-ks)


def build_tridiagonal_B(betas: torch.Tensor) -> torch.Tensor:
    """Assemble the ``K x K`` diffusion matrix ``B`` from edge couplings.

    ``betas`` is the length ``K-1`` vector of edge couplings ``(b_2, ..., b_K)``
    (edge ``e`` couples reservoirs ``e`` and ``e+1``, 0-indexed). Returns the
    symmetric row-stochastic ``B`` such that ``u_t = B u_{t-1} + e_1 grad_t``.
    Provided mainly for the eigenvalue/stability diagnostic; the scan does not
    materialise ``B`` (see :func:`cascade_step`).
    """
    betas = torch.as_tensor(betas)
    Km1 = betas.shape[-1]
    K = Km1 + 1
    B = torch.eye(K, dtype=betas.dtype, device=betas.device)
    for e in range(Km1):
        b = betas[e]
        B[e, e] = B[e, e] - b
        B[e, e + 1] = B[e, e + 1] + b
        B[e + 1, e + 1] = B[e + 1, e + 1] - b
        B[e + 1, e] = B[e + 1, e] + b
    return B


def spectral_radius(betas: torch.Tensor) -> torch.Tensor:
    """Largest-modulus eigenvalue of ``B``; should be exactly 1 (mass conserving)."""
    B = build_tridiagonal_B(betas)
    return torch.linalg.eigvals(B).abs().max().real


# --------------------------------------------------------------------------- #
# Pure recurrence: step + scan (structured so a future parallel associative
# scan -- cumulative tridiagonal products, Mamba-style -- can drop in without
# touching the public API).
# --------------------------------------------------------------------------- #


def cascade_step(
    u_prev: torch.Tensor,
    grad_t: torch.Tensor,
    betas_t: torch.Tensor,
) -> torch.Tensor:
    """One cascade step ``u_t = B_t u_{t-1} + e_1 grad_t`` (pure, no in-place).

    Parameters
    ----------
    u_prev : ``[..., K]`` previous cascade state.
    grad_t : ``[...]`` gradient injected into reservoir 1 (matches ``u_prev``
        without the trailing ``K`` axis).
    betas_t : ``[..., K-1]`` edge couplings, broadcastable over the leading dims
        of ``u_prev``. For a fixed cascade this is the same every step; for the
        selective cascade it varies with the token.

    Implemented via the tridiagonal diffusion flux directly (no ``B`` matmul),
    so it is ``O(K)`` and fully functional -- important because the outer model's
    autograd runs through this recurrence (in-place edits would break it).
    """
    # Net diffusion flux across each edge e: f_e = b_e * (u^(e+1) - u^(e)).
    flux = betas_t * (u_prev[..., 1:] - u_prev[..., :-1])  # [..., K-1]
    # Reservoir e gains +f_e (from its right neighbour); reservoir e+1 gains -f_e.
    inflow_from_right = F.pad(flux, (0, 1))  # [..., K], last reservoir gets 0
    inflow_from_left = F.pad(-flux, (1, 0))  # [..., K], first reservoir gets 0
    # Gradient injected only into reservoir 1 (e_1).
    inj = F.pad(grad_t.unsqueeze(-1), (0, u_prev.shape[-1] - 1))
    return u_prev + inflow_from_right + inflow_from_left + inj


def independent_step(
    u_prev: torch.Tensor,
    grad_t: torch.Tensor,
    decays: torch.Tensor,
) -> torch.Tensor:
    """One step of ``K`` *independent* exponential filters (diagonal ``B``).

    ``u_t^(k) = decays_k * u_{t-1}^(k) + grad_t`` for every ``k`` -- i.e. the
    same injected ``grad_t`` reaches **all** reservoirs (not just reservoir 0),
    each leaks at its own fixed rate ``decays_k``, and *no* mass is exchanged
    between reservoirs (``B = diag(decays)`` is diagonal, not tridiagonal). This
    is the genuine "K independent momentums" null hypothesis: a fixed bank of
    exponential filters with no coupling. Contrast :func:`cascade_step`, where the
    gradient hits only reservoir 0 and off-diagonal diffusion is what spreads it.

    Parameters
    ----------
    u_prev : ``[..., K]`` previous filter-bank state.
    grad_t : ``[...]`` gradient injected identically into every reservoir.
    decays : ``[..., K]`` (or broadcastable ``[K]``) per-reservoir self-decays.
    """
    return decays * u_prev + grad_t.unsqueeze(-1)


def cascade_scan(
    u_init: torch.Tensor,
    grads: torch.Tensor,
    betas: torch.Tensor,
    *,
    time_varying: bool = False,
) -> torch.Tensor:
    """Sequential scan of :func:`cascade_step` over a time axis.

    Parameters
    ----------
    u_init : ``[..., K]`` initial cascade state.
    grads : ``[T, ...]`` gradients, time-major (leading axis is time ``T``).
    betas : edge couplings. If ``time_varying`` is False, a single ``[..., K-1]``
        tensor reused every step (fixed ``B``). If True, ``[T, ..., K-1]`` with a
        leading time axis (selective / data-dependent ``B_t``).
    Returns
    -------
    ``[T, ..., K]`` stacked cascade states ``u_1, ..., u_T``.

    This is intentionally a Python loop -- consistent with the current project
    state (the single-momentum recurrence in ``memory.py`` is also a Python scan
    across chunks). The step/scan split is the seam where a parallel associative
    scan could later replace the loop without changing callers.
    """
    T = grads.shape[0]
    u = u_init
    out = []
    for t in range(T):
        b_t = betas[t] if time_varying else betas
        u = cascade_step(u, grads[t], b_t)
        out.append(u)
    return torch.stack(out, dim=0)


def read(u: torch.Tensor) -> torch.Tensor:
    """The visible momentum: reservoir 1, i.e. what ``S_t`` used to be.

    This is what plugs into ``M_t = (1 - alpha_t) M_{t-1} + read(u_t)`` in the chain
    topology, opt-in via ``NeuralMemory(use_cascade_momentum=True)`` (see
    :meth:`NeuralMemory._step`). The ``independent`` topology reads the mean instead
    (:meth:`MetaplasticCascade.readout`).
    """
    return u[..., 0]


# --------------------------------------------------------------------------- #
# Module wrapper: holds base couplings, optional selective head.
# --------------------------------------------------------------------------- #


class MetaplasticCascade(nn.Module):
    """K-reservoir cascade with fixed or data-dependent (selective) couplings.

    Parameters
    ----------
    num_reservoirs : ``K``, number of internal states per weight entry.
    beta1, g : geometric coupling hyperparameters (see :func:`geometric_betas`).
    dim : token dimension; required only when ``selective=True`` to build the
        surprise head.
    selective : if True, a learned head produces per-token, per-edge gates in
        ``(0, 1)`` that scale the base couplings edge-by-edge, making ``B_t``
        data-dependent *and non-commuting* (mode 2 in the module docstring). If
        False, ``B`` is the fixed geometric matrix (mode 1). Only valid for
        ``topology="chain"``. What drives the gate is ``gate_signal``.
    gate_signal : ``"content"`` (default) gates on the token representation
        ``x_t`` via ``Linear(dim, K-1)`` -- a smooth, content-addressed edge gate
        with no explicit relationship to how surprising the token actually is.
        ``"surprise"`` instead gates on the memory's own *raw* gradient magnitude
        ``||grad_t||`` (Titans' own definition of surprise, Sec 3.1) via
        ``Linear(1, K-1)`` on ``log1p(||grad_t||)``: a learned per-edge
        threshold+slope on the surprise itself, so consolidation is literally
        gated by "is this surprising enough to pass deeper" rather than by content
        -- the mechanism this module's docstring motivates but the ``"content"``
        mode does not actually implement.
    topology : ``"chain"`` (default) is the tridiagonal diffusion cascade
        (gradient into reservoir 0, mass exchanged between neighbours).
        ``"independent"`` is the diagonal-``B`` null hypothesis: ``K`` uncoupled
        exponential filters (:func:`geometric_decays`), the gradient injected into
        *all* of them (see :func:`independent_step`). Independent mode has no
        coupling to gate, so it forbids ``selective``.
    readout : which reservoir(s) feed the visible momentum -- see :meth:`readout`.
        ``"gated"`` makes the *readout itself* context-dependent (a learned
        ``Linear(dim, K)`` + softmax on ``x_t``), instead of a fixed combination:
        without this, even a perfectly content/surprise-gated diffusion has its
        distinctions discarded by a readout that always blends the same way.

    The base couplings / decays are stored as non-trained buffers so a fixed
    cascade has no learnable parameters, mirroring how the geometric spacing is a
    plain hyperparameter of the biological model.
    """

    def __init__(
        self,
        num_reservoirs: int,
        beta1: float = 0.5,
        g: float = 2.0,
        dim: Optional[int] = None,
        selective: bool = False,
        topology: str = "chain",
        readout: Optional[str] = None,
        gate_signal: str = "content",
    ):
        super().__init__()
        if num_reservoirs < 1:
            raise ValueError("num_reservoirs (K) must be >= 1")
        if topology not in ("chain", "independent"):
            raise ValueError("topology must be 'chain' or 'independent'")
        if topology == "independent" and selective:
            raise ValueError(
                "topology='independent' has no coupling to gate -> selective=False"
            )
        if gate_signal not in ("content", "surprise"):
            raise ValueError("gate_signal must be 'content' or 'surprise'")
        self.num_reservoirs = num_reservoirs
        self.selective = selective
        self.topology = topology
        self.gate_signal = gate_signal
        # Readout: which reservoir(s) feed the visible momentum. Default preserves
        # prior behaviour (chain -> shallowest u^1; independent -> mean). ``deep`` /
        # ``mean`` / ``sum`` / ``learned`` / ``gated`` expose the *slow* reservoirs,
        # whose long retention is otherwise invisible when only u^1 is read.
        if readout is None:
            readout = "mean" if topology == "independent" else "shallow"
        if readout not in ("shallow", "mean", "sum", "deep", "learned", "gated"):
            raise ValueError(f"unknown readout mode {readout!r}")
        self.readout_mode = readout
        if readout == "learned":
            self.readout_w = nn.Parameter(torch.zeros(num_reservoirs))
        if readout == "gated":
            if dim is None:
                raise ValueError("readout='gated' needs `dim` for the readout head")
            self.readout_head = nn.Linear(dim, num_reservoirs)
        base = geometric_betas(num_reservoirs, beta1, g, dtype=torch.get_default_dtype())
        self.register_buffer("base_betas", base)  # [K-1]
        # Diagonal-B decays for the independent topology (unused by the chain).
        decays = geometric_decays(num_reservoirs, beta1, g, dtype=torch.get_default_dtype())
        self.register_buffer("decays", decays)  # [K]

        if selective:
            if num_reservoirs < 2:
                raise ValueError("selective cascade needs K >= 2 (>=1 edge to gate)")
            # One gate per edge (K-1 outputs), so the *shape* of the diffusion --
            # not just its global scale -- is data-dependent (non-commuting B_t).
            if gate_signal == "content":
                if dim is None:
                    raise ValueError(
                        "selective cascade (gate_signal='content') needs `dim`"
                    )
                self.beta_head = nn.Linear(dim, num_reservoirs - 1)
            else:  # "surprise": scalar log-magnitude in, per-edge gate out
                self.beta_head = nn.Linear(1, num_reservoirs - 1)

    # ------------------------------------------------------------------ state
    def init_state(self, template: torch.Tensor) -> torch.Tensor:
        """Zero cascade state shaped like ``template`` (a weight/momentum tensor)
        with an appended reservoir axis of size ``K``."""
        return template.new_zeros(*template.shape, self.num_reservoirs)

    def betas(self) -> torch.Tensor:
        """The fixed base edge couplings ``[K-1]`` (mode-1 use)."""
        return self.base_betas

    def selective_betas(self, signal: torch.Tensor) -> torch.Tensor:
        """Per-token, per-edge couplings from the head: ``base_betas * sigmoid(W signal)``.

        ``signal`` is ``[..., dim]`` (``gate_signal="content"``: the token ``x_t``)
        or ``[..., 1]`` (``gate_signal="surprise"``: ``log1p(||grad_t||)``, see
        :meth:`~titans.memory.NeuralMemory._step`); returns ``[..., K-1]``. Each
        edge gate in ``(0, 1)`` never *increases* a coupling beyond its geometric
        base (so stability is preserved: shrinking a stable doubly-stochastic
        ``B`` keeps it stable), but per-edge gating reshapes the diffusion profile
        per token, yielding an event-triggered, non-commuting ``B_t``. Under
        ``gate_signal="surprise"`` this is a literal learned threshold+slope on
        the surprise magnitude: large ``|W|`` sharpens it toward a hard "pass only
        if surprising enough" gate, exactly what a fixed geometric cascade cannot
        express.
        """
        if not self.selective:
            raise RuntimeError("selective_betas requires selective=True")
        gate = torch.sigmoid(self.beta_head(signal))  # [..., K-1]
        return gate * self.base_betas  # per-edge scaling -> [..., K-1]

    # --------------------------------------------------------------- recurrence
    def step(
        self,
        u_prev: torch.Tensor,
        grad_t: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One step. Uses selective couplings from ``x_t`` if ``selective``,
        else the fixed base couplings."""
        if self.selective:
            if x_t is None:
                raise ValueError("selective cascade step needs the token x_t")
            betas_t = self.selective_betas(x_t)
        else:
            betas_t = self.base_betas
        return cascade_step(u_prev, grad_t, betas_t)

    def scan(
        self,
        u_init: torch.Tensor,
        grads: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scan over time. ``grads`` is ``[T, ...]``; ``x`` (``[T, ..., dim]``) is
        required for the selective mode to derive per-step ``B_t``."""
        if self.selective:
            if x is None:
                raise ValueError("selective cascade scan needs the token stream x")
            betas = self.selective_betas(x)  # [T, ..., K-1]
            return cascade_scan(u_init, grads, betas, time_varying=True)
        return cascade_scan(u_init, grads, self.base_betas)

    @staticmethod
    def read(u: torch.Tensor) -> torch.Tensor:
        """Visible momentum (reservoir 1); see module-level :func:`read`."""
        return read(u)

    def readout(self, u: torch.Tensor, x_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Visible momentum fed into the memory's weight-decay rule.

        ``readout_mode`` selects which reservoir(s) are read: ``shallow`` = u^1
        (fastest; the original Titans-like momentum), ``deep`` = u^K (slowest),
        ``mean``/``sum`` = pool all K, ``learned`` = a *fixed* (outer-trained,
        context-independent) softmax-weighted combination, ``gated`` = a
        *per-token* softmax-weighted combination from a ``Linear(dim, K)`` head on
        ``x_t`` -- unlike every other mode, the mixture itself can change with the
        current input, so a threshold-gated diffusion's distinction between
        "this got consolidated deep" and "this stayed shallow" can actually be
        read out differently depending on context, instead of always being
        blended the same fixed way. ``mean``/``learned``/``gated`` keep the
        readout on a single-filter scale; ``sum`` inflates it ~K-fold (rescale the
        inner lr accordingly).
        """
        mode = self.readout_mode
        if mode == "shallow":
            return u[..., 0]
        if mode == "deep":
            return u[..., -1]
        if mode == "mean":
            return u.mean(dim=-1)
        if mode == "sum":
            return u.sum(dim=-1)
        if mode == "gated":
            if x_t is None:
                raise ValueError("readout='gated' needs the token x_t")
            w = torch.softmax(self.readout_head(x_t), dim=-1)  # [..., K]
            w = w.view(w.shape[0], *([1] * (u.dim() - 2)), w.shape[-1])
            return (u * w).sum(dim=-1)
        w = torch.softmax(self.readout_w, dim=0)  # learned
        return (u * w).sum(dim=-1)
