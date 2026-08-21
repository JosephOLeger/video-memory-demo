"""A neural long-term memory grafted onto a frozen VLM as a lateral branch.

What this is, and what it deliberately is not
---------------------------------------------
The block only ever **adds**. It reads the decoder's hidden states at one layer,
writes them into a Titans memory, retrieves from that memory, and adds the result
back into the residual stream. It does not replace an attention block, it does not
shorten the context, and it does not touch a single VLM parameter.

**There is no attention inside the branch**, and that is the point. In
:class:`~titans.mac.TitansMACBlock` the memory sits *behind* an attention block
whose output ``y_i`` both writes to and reads from the memory -- the feedback loop
that ``decouple_memory_update`` exists to sever. Here the decoder's layer-``L``
output is already an attended representation, so the branch consumes it directly.
A second attention would duplicate what the VLM already does, cost context, and
reintroduce the loop. (This is also why ``learned_qk_scale`` -- part of ``sane_ns``
-- has no analogue here: it is a parameter of the *attention* block, not of the
memory.)

Why the up-projection starts at zero
------------------------------------
``up`` is zero-initialised, so at step 0 the branch contributes exactly ``0`` and
the grafted model is **bit-identical** to the frozen VLM. That is not cosmetic: it
makes "does the memory help?" measurable against the untouched model rather than
against a differently-initialised one, and any early instability is attributable
to the branch alone. :meth:`MemoryBranch.is_identity` asserts it.

MAC cannot do this. Its ``_gate`` mixes ``y`` and ``r`` multiplicatively
(``y*sigmoid(W r) + r*sigmoid(W y)``), which has no parameter setting that returns
``y`` unchanged. A plain additive residual does, so the graft gets a free exact
baseline that the MAC block never had.

Read cadence vs write cadence
-----------------------------
The two are separated on purpose. A memory holds a trace for a finite number of
*writes*, so at one write per token a memory sized for a GPU we own reaches minutes,
not the 5.9 h median of EgoLifeQA. Writing once every ``write_stride`` tokens
multiplies the reach by that stride at **no VRAM cost**, while the VLM still sees
every frame and the branch still reads at every position. The alternative to a
fixed stride is the surprise gate the memory already implements
(``cascade_selective``); this class supports either (``write_stride``, or an
explicit ``write_mask``).

Streaming callers must pass ``token_offset`` -- the absolute index of ``h[:, 0]``
in the stream -- or the stride's phase silently resets at every call and the write
cadence stops being uniform.

The input basis is a PCA, not a learned projection
--------------------------------------------------
``down`` is meant to carry a **fixed** basis fitted on the frozen host's layer-L
activations (:func:`titans.worldmodel.bottleneck.fit_pca`), the same move that made
the reduction a property of the *data* in the world model instead of a bundle of
parameters free to co-adapt with the memory.

There is a second reason, specific to the zero-init residual. At step 0 the
gradient reaching ``down`` must pass through ``up``, which is zero, so
``dL/d(down) = 0`` exactly -- and the same holds for ``q_proj`` and the memory's
gate heads. A learned ``down`` therefore starts from a random basis that receives
no gradient on the first step and only bootstraps through whatever ``up`` becomes.
That is survivable (it is LoRA's ``A`` random / ``B`` zero structure) but it is a
bootstrap cost a good fixed basis removes outright.

The learned-``down`` path is kept for the ablation, and because the PCA cannot be
fitted before the host is wired up and its activations sampled.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..memory import NeuralMemory

Params = dict


def fold_pca(pca) -> Tuple[torch.Tensor, torch.Tensor]:
    """The affine map that ``bottleneck.apply_pca`` computes, as ``(W, b)``.

    ``apply_pca`` is ``z = ((h - mean) / std) @ components.T / sv``, which is affine
    in ``h``; folding it into one ``[k, D]`` matrix and a ``[k]`` bias makes the
    branch's input a single matmul and lets ``down`` stay an ordinary ``nn.Linear``
    whose ``state_dict`` shape does not depend on how it was built.
    """
    comps = pca["components"].double()                       # [k, D]
    mean = pca["mean"].double()                              # [D]
    std = pca["std"].double()                                # [D]
    sv = pca["sv"].double() if pca["whiten"] else torch.ones_like(comps[:, 0])
    W = comps / std.unsqueeze(0) / sv.unsqueeze(1)           # [k, D]
    b = -((mean / std) @ comps.T) / sv                       # [k]
    return W.float(), b.float()

#: ``sane_ns`` expressed in :class:`~titans.memory.NeuralMemory`'s own argument
#: names. The standing rule for this project is that every arm is a delta on
#: ``sane_ns``, so the graft starts there rather than on library defaults --
#: those would silently reintroduce ``normalize_output=True``, which empties the
#: inner loss. ``normalize_retrieval`` is not a memory argument (the MAC block
#: owns that RMSNorm); it is realised here by ``MemoryBranch.norm_r``.
SANE_NS = dict(
    alpha_init=0.002,
    theta_scale=0.03,
    normalize_qk="l2",
    normalize_output=False,
    orthogonalize_momentum=True,
    update_bias=False,
)

#: Kept out of :data:`SANE_NS` because it is a separate standing decision, not
#: part of the config: every run we launch sets ``eta_init``. It is the momentum
#: gate's init, not ``eta`` itself -- the outer loop moves it bimodally.
ETA_INIT = 0.995


def sane_memory_kwargs(**overrides) -> dict:
    """``sane_ns`` + ``eta_init``, with explicit per-run deltas on top."""
    return {**SANE_NS, "eta_init": ETA_INIT, **overrides}


def memory_kwargs_from_args(args) -> dict:
    """The memory's construction kwargs, from a run's saved ``args``.

    Training and evaluation must build the **same** memory, and most of the ways
    they can disagree do not change any parameter shape: ``orthogonalize_momentum``,
    ``theta_gate`` and the cascade flags all leave ``state_dict`` identical. So
    ``load_state_dict`` succeeds, nothing warns, and the checkpoint is evaluated
    under a different inner loop than the one that trained it. Having one function
    called by both scripts is what keeps that from happening quietly.

    Accepts a dict (``blob["args"]``) or an argparse namespace.
    """
    g = ((lambda k, d=None: args.get(k, d)) if isinstance(args, dict)
         else (lambda k, d=None: getattr(args, k, d)))
    kw = dict(hidden_dim=g("memory_hidden", 64),
              alpha_init=g("alpha_init", 1.27e-4),
              orthogonalize_momentum=g("muon", "unit"))
    if g("muon_scale") is not None:            # 0.0 is a real clip, not "absent"
        kw["orthogonalize_clip"] = g("muon_scale")
    if g("cascade"):
        kw.update(use_cascade_momentum=True, cascade_reservoirs=g("reservoirs", 6))
    return kw


class MemoryBranch(nn.Module):
    """One lateral Titans memory, sized for a decoder of width ``d_model``.

    Parameters
    ----------
    d_model: the VLM's hidden width (2048 for Qwen2.5-VL-3B).
    d_mem: the memory's own width. This is the capacity knob: the memory MLP holds
        ``~2 * d_mem * hidden_dim`` parameters, and measured reach grows as
        ``N^0.93`` in that count. Kept separate from ``d_model`` precisely so it
        can be swept without touching the host model.
    chunk_size: the inner loop's parallel chunk, as in ``NeuralMemory.update``.
    write_stride: write once every ``write_stride`` tokens (1 = every token).
    normalize_retrieval: RMSNorm on the retrieval before ``up``. SANE's fifth
        ingredient: it bounds what *leaves* the memory without touching the inner
        loss the way ``normalize_output`` does. On by default, as in ``sane_ns``.
    down_pca: a basis fitted by :func:`titans.worldmodel.bottleneck.fit_pca` on the
        host's layer-L activations. Its ``k`` must equal ``d_mem``. Frozen unless
        ``train_down`` says otherwise; passing ``None`` falls back to a learned
        projection (the ablation, and the only option before a basis exists).
    train_down: override the freeze. Defaults to ``down_pca is None``.
    **memory_kwargs: forwarded to :class:`~titans.memory.NeuralMemory`, on top of
        :func:`sane_memory_kwargs`. This is where the cascade options land
        (``use_cascade_momentum``, ``cascade_areas``, ``cascade_terminal_leak``,
        ``cascade_leak_rate``, ...).
    """

    def __init__(self, d_model: int, d_mem: int = 128, *, chunk_size: int = 8,
                 write_stride: int = 1, normalize_retrieval: bool = True,
                 down_pca=None, train_down: Optional[bool] = None,
                 **memory_kwargs):
        super().__init__()
        if d_model < 1 or d_mem < 1:
            raise ValueError("d_model and d_mem must be positive")
        if write_stride < 1:
            raise ValueError("write_stride must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.d_model = d_model
        self.d_mem = d_mem
        self.chunk_size = chunk_size
        self.write_stride = write_stride

        self.down = nn.Linear(d_model, d_mem, bias=down_pca is not None)
        self.pca_explained = None
        if down_pca is not None:
            if int(down_pca["k"]) != d_mem:
                raise ValueError(f"basis has k={down_pca['k']}, d_mem is {d_mem}")
            if down_pca["components"].shape[1] != d_model:
                raise ValueError(
                    f"basis was fitted on D={down_pca['components'].shape[1]}, "
                    f"d_model is {d_model}")
            W, b = fold_pca(down_pca)
            with torch.no_grad():
                self.down.weight.copy_(W)
                self.down.bias.copy_(b)
            self.pca_explained = float(down_pca["explained"])
        if train_down is None:
            train_down = down_pca is None
        self.down.requires_grad_(bool(train_down))

        self.up = nn.Linear(d_mem, d_model, bias=False)
        # The identity-at-init property the whole measurement rests on.
        nn.init.zeros_(self.up.weight)
        self.norm_r = nn.RMSNorm(d_mem) if normalize_retrieval else None

        self.memory = NeuralMemory(dim=d_mem, **sane_memory_kwargs(**memory_kwargs))

    def trainable_parameters(self):
        """The parameters an optimiser should see -- a frozen ``down`` is excluded."""
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ carry
    def init_carry(self, batch: int) -> Tuple[Params, Params]:
        """A fresh ``(state, S)`` -- the memory weights and their momentum."""
        return self.memory.init_memory(batch)

    def is_identity(self) -> bool:
        """True while the branch is still an exact no-op on the residual stream."""
        return bool(torch.all(self.up.weight == 0))

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor, state: Optional[Params] = None,
                S: Optional[Params] = None, *, token_offset: int = 0,
                write_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Params, Params]:
        """``h + delta(h)``, carrying the memory forward. See :meth:`delta`."""
        d, state, S = self.delta(h, state, S, token_offset=token_offset,
                                 write_mask=write_mask)
        return h + d, state, S

    def delta(self, h: torch.Tensor, state: Optional[Params] = None,
              S: Optional[Params] = None, *, token_offset: int = 0,
              write_mask: Optional[torch.Tensor] = None,
              ) -> Tuple[torch.Tensor, Params, Params]:
        """What the branch *contributes*, ``up(norm(retrieve(...)))``, before the add.

        Returned separately from the sum because the host runs in bf16 while this
        branch runs in fp32: the caller casts the delta back and adds it in the
        host's dtype, so a zero delta leaves the residual stream **bit-identical**
        rather than merely close. Doing the add here and casting the sum would
        round every activation through fp32 and lose that guarantee.

        ``h`` is ``[B, N, d_model]``; the carry is returned so a long video can be
        streamed segment by segment without materialising the whole sequence.
        ``write_mask`` (``[N]`` or ``[B, N]``, bool) overrides the stride when the
        caller wants a content-dependent cadence.

        Note the order: the segment is written *before* it is read, so a token can
        retrieve what it just wrote. That is the MAC block's ``r_i`` path, not its
        ``h_i`` path, and it is the one that carries the long-range signal.
        """
        if h.dim() != 3:
            raise ValueError(f"expected [B, N, d_model], got {tuple(h.shape)}")
        B, N, d = h.shape
        if d != self.d_model:
            raise ValueError(f"input width {d} != d_model {self.d_model}")
        if state is None or S is None:
            state, S = self.init_carry(B)

        z = self.down(h)                                        # [B, N, d_mem]

        # --- write ---------------------------------------------------------
        # Subsampling happens on the *update* only; the retrieval below still runs
        # at every position, so the host model loses no resolution.
        idx = self._write_positions(N, token_offset, write_mask, h.device)
        if idx.numel():
            z_w = z.index_select(1, idx)
            # `NeuralMemory.update` clamps its last chunk (`end = min(...)`), so a
            # ragged tail is written, not dropped.
            state, S = self.memory.update(z_w, state, S,
                                          min(self.chunk_size, z_w.shape[1]))

        # --- read ----------------------------------------------------------
        r = self.memory.retrieve(state, self.memory.q_proj(z))  # [B, N, d_mem]
        if self.norm_r is not None:
            r = self.norm_r(r)
        return self.up(r), state, S

    def _write_positions(self, n: int, offset: int, mask: Optional[torch.Tensor],
                         device) -> torch.Tensor:
        """Indices into the segment at which the memory is written."""
        if mask is not None:
            m = mask if mask.dim() == 1 else mask[0]
            if m.shape[0] != n:
                raise ValueError(f"write_mask length {m.shape[0]} != sequence {n}")
            return torch.nonzero(m.to(device=device, dtype=torch.bool),
                                 as_tuple=False).flatten()
        if self.write_stride == 1:
            return torch.arange(n, device=device)
        # Phase carried from the absolute position, so a stride survives being cut
        # into segments of any length.
        first = (-offset) % self.write_stride
        return torch.arange(first, n, self.write_stride, device=device)


def stride_for_reach(target_seconds: float, fps: float, tokens_per_frame: int,
                     writes_available: int) -> int:
    """Tokens between writes so ``target_seconds`` fits in ``writes_available``.

    The memory's reach is counted in *writes*, not in seconds or tokens, which is
    why the graft's temporal span is a scheduling choice rather than a property of
    the host model. ``writes_available`` is the memory's measured lifetime at its
    size -- measure it with ``titans.worldmodel.bf_score``, do not assume it.
    """
    if min(target_seconds, fps, tokens_per_frame, writes_available) <= 0:
        raise ValueError("all arguments must be positive")
    frames_per_write = max(1.0, (target_seconds * fps) / writes_available)
    return max(1, int(round(frames_per_write * tokens_per_frame)))
