"""Wiring the lateral memory branch into a frozen Qwen2.5-VL decoder.

The graft replaces one decoder layer with a wrapper that calls the original layer
and then adds the branch's contribution. The original module is kept, not copied,
so :func:`ungraft` restores the model exactly and nothing about the host is
rebuilt, re-initialised, or retrained.

Three properties this module exists to guarantee
------------------------------------------------
1. **Bit-identical at init.** ``up`` is zero, the delta is exactly ``0``, and the
   add happens in the host's own dtype -- so the grafted model's logits equal the
   base model's *exactly*, not approximately. :func:`identity_check` runs the base
   model first and the grafted model second on the same input, in one process, so
   the comparison costs no second copy of the weights.
2. **The host never trains.** Everything outside the branch is
   ``requires_grad_(False)``, and layers below the graft site can additionally run
   under ``no_grad`` -- which is the whole reason the site is high. It is the
   *lowest* graft site that sets the activation-memory bill.
3. **The memory does not run in bf16.** The host is bf16; the branch's inner loop
   (chunked gradients, Newton-Schulz, an online-updated weight matrix) is not
   something to hand to 8 mantissa bits, and this project has already been bitten
   by seed-dependent NaN divergence *in fp32*. The wrapper follows the branch's own
   precision -- build the branch in fp32 (the default) and the host's bf16 tensor is
   cast on the way in, the delta cast back on the way out.

Layer indexing is read off the model, never assumed: pass ``layer=None`` to land
at two thirds of depth, which is 24 of 36 on Qwen2.5-VL-3B-Instruct.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .branch import MemoryBranch


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The text decoder's ``ModuleList``, wherever this ``transformers`` puts it.

    The path moved between major versions (``model.layers`` ->
    ``model.language_model.layers``), so it is searched rather than hard-coded; a
    wrong guess here would silently graft onto the vision tower.
    """
    candidates = (
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
        "model.model.layers",
        "layers",
    )
    for path in candidates:
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node):
            return node
    found = [n for n, m in model.named_modules() if isinstance(m, nn.ModuleList)]
    raise AttributeError(
        "could not find the decoder layers; ModuleLists present: " + ", ".join(found[:10]))


class GraftedLayer(nn.Module):
    """``layer`` followed by the memory branch, with the carry held here.

    The carry lives on the module rather than being threaded through
    ``forward``'s arguments because the host's decoder loop calls the layer and
    has no idea the branch exists. :meth:`reset` starts a new video;
    :attr:`token_offset` keeps the write cadence's phase across chunks of one.
    """

    def __init__(self, layer: nn.Module, branch: MemoryBranch, *,
                 branch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.layer = layer
        self.branch = branch
        # Follow the branch's own precision rather than pinning one here: which
        # dtype the inner loop deserves is a property of how the branch was built.
        # Build it in fp32 (the default) and the bf16 host is cast on the way in.
        self.branch_dtype = branch_dtype or next(branch.parameters()).dtype
        self.enabled = True
        # Reading without writing. The answer segment of a distillation step must
        # retrieve from the state the context built and leave it alone: writing the
        # question into memory gives the loss a second gradient path and changes the
        # state the phase before it produced. Set False around that segment.
        self.writes_enabled = True
        self.reset()

    def reset(self) -> None:
        """Forget the video: fresh memory, phase back to zero."""
        self.carry: Tuple[Optional[dict], Optional[dict]] = (None, None)
        self.token_offset = 0

    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        if not self.enabled:
            return out
        h = out[0] if isinstance(out, tuple) else out
        state, S = self.carry
        mask = None
        if not self.writes_enabled:
            mask = torch.zeros(h.shape[1], dtype=torch.bool, device=h.device)
        d, state, S = self.branch.delta(h.to(self.branch_dtype), state, S,
                                        token_offset=self.token_offset,
                                        write_mask=mask)
        if getattr(self, "sub_base", False):
            # 16/11 (eval only): baseline-subtracted read. Remove what the branch adds with an EMPTY memory, i.e. the
            # constant bias of the read path, and keep only the part of the read that depends on the carried state.
            # With state == init this makes the layer an exact no-op (d - d0 == 0), so `init` must equal `off`.
            st0, S0 = self.branch.init_carry(h.shape[0])
            d0, _, _ = self.branch.delta(h.to(self.branch_dtype), st0, S0, token_offset=self.token_offset,
                                         write_mask=torch.zeros(h.shape[1], dtype=torch.bool, device=h.device))
            d = d - d0
        self.carry = (state, S)
        self.token_offset += h.shape[1]
        # Added in the host's dtype: a zero delta must leave `h` untouched bit for
        # bit, which casting the *sum* through fp32 would not preserve.
        h = h + d.to(h.dtype)
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h


def default_layer(n_layers: int) -> int:
    """Two thirds of depth -- high enough that the memory's addresses recur."""
    return max(0, min(n_layers - 1, round(2 * n_layers / 3)))


def freeze_host(model: nn.Module) -> int:
    """Freeze every host parameter. Returns how many tensors were frozen."""
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += 1
    return n


def graft(model: nn.Module, branch: MemoryBranch, *, layer: Optional[int] = None,
          branch_dtype: Optional[torch.dtype] = None,
          freeze: bool = True) -> GraftedLayer:
    """Install ``branch`` after decoder layer ``layer``. Returns the wrapper.

    ``freeze`` freezes the host *before* the branch is installed, so the branch's
    own parameters keep their ``requires_grad``.
    """
    layers = find_decoder_layers(model)
    if freeze:
        freeze_host(model)
    idx = default_layer(len(layers)) if layer is None else layer
    if not 0 <= idx < len(layers):
        raise IndexError(f"layer {idx} out of range for {len(layers)} layers")
    if isinstance(layers[idx], GraftedLayer):
        raise RuntimeError(f"layer {idx} is already grafted")
    wrapper = GraftedLayer(layers[idx], branch, branch_dtype=branch_dtype)
    layers[idx] = wrapper
    wrapper.graft_index = idx
    return wrapper


def ungraft(model: nn.Module, wrapper: GraftedLayer) -> None:
    """Put the original layer back, leaving the host exactly as it was."""
    layers = find_decoder_layers(model)
    idx = getattr(wrapper, "graft_index", None)
    if idx is None or layers[idx] is not wrapper:
        idx = next((i for i, m in enumerate(layers) if m is wrapper), None)
    if idx is None:
        raise ValueError("that wrapper is not installed on this model")
    layers[idx] = wrapper.layer


@contextlib.contextmanager
def disabled(wrapper: GraftedLayer):
    """Temporarily pass the layer through untouched."""
    was, wrapper.enabled = wrapper.enabled, False
    try:
        yield
    finally:
        wrapper.enabled = was


def branch_report(branch: MemoryBranch, host_params: int) -> dict:
    """Parameter accounting, for the sentence "we added 0.0x % and nothing else"."""
    trainable = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in branch.parameters() if not p.requires_grad)
    return {
        "branch_trainable": trainable,
        "branch_frozen_basis": frozen,
        "host_params": host_params,
        "pct_of_host": 100.0 * trainable / host_params if host_params else float("nan"),
        "pca_explained": branch.pca_explained,
    }


@torch.no_grad()
def identity_check(model: nn.Module, branch: MemoryBranch, inputs: dict, *,
                   layer: Optional[int] = None,
                   branch_dtype: Optional[torch.dtype] = None) -> dict:
    """Base logits vs grafted logits on the same input, in one process.

    Runs the ungrafted model first, keeps its logits, then installs the graft and
    runs again -- so the comparison needs no second copy of the weights and, unlike
    comparing the graft against itself with ``enabled=False``, it does catch a
    wrapper that mangles the layer's arguments.
    """
    was_training = model.training
    model.eval()
    try:
        base = model(**inputs).logits.float().clone()
        wrapper = graft(model, branch, layer=layer, branch_dtype=branch_dtype)
        try:
            wrapper.reset()
            after = model(**inputs).logits.float()
            same = bool(torch.equal(base, after))
            return {
                "identical": same,
                "max_abs_diff": float((after - base).abs().max()),
                "layer": wrapper.graft_index,
                "wrote_tokens": int(wrapper.token_offset),
                "memory_moved": bool(wrapper.carry[0] is not None),
            }
        finally:
            ungraft(model, wrapper)
    finally:
        model.train(was_training)
