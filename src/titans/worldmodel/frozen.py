"""A trained world model, packaged so inference needs no configuration.

The problem this solves. A checkpoint stores only ``state_dict``, so reloading one
means re-specifying the architecture by hand, and getting a flag wrong is *not*
reliably an error: the write rule and the gates change no parameter shape, so ``load_state_dict`` succeeds and the model quietly computes something else.
Every inference experiment then silently measures a different architecture.

A frozen artifact carries its own config. :func:`load_frozen` returns a model in
``eval()`` with ``requires_grad_(False)`` everywhere and nothing left to decide::

    model, info = load_frozen("frozen_muon/mu_unit10k_s0.pt")

"Frozen" here means the *outer* weights: attention, gates, projections and the
memory's own parameters (its heads and its initial weights). The memory's **state**
is exactly what stays live -- it is updated by the inner loop at inference time, which
is the whole point of the architecture. Freezing the outer weights is what isolates
that state as the only moving part.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch

from .atlas import CONFIGS
from .model import WorldModel

FORMAT = 1


def build_cfg(config: str = "sane_ns", *, eta_init: Optional[float] = None,
              theta_scale: Optional[float] = None,
              theta_gate: Optional[str] = None,
              normalize_hidden: Optional[float] = None,
              muon: Optional[str] = None,
              muon_scale: Optional[float] = None,
              alpha_init: Optional[float] = None,
              use_memory: bool = True,
              memory_hidden_dim: Optional[int] = None,
              memory_chunk_size: Optional[int] = None,
              num_heads: Optional[int] = None) -> dict:
    """The kwargs a :class:`WorldModel` needs, from the sweep's own flag names.

    Kept in one place so ``echo.py``, ``long_run.py`` and this module cannot drift
    into building three slightly different models from the same flags.
    """
    cfg = dict(CONFIGS[config])
    if not use_memory:
        # The ablation the whole claim is measured against: same block, memory branch
        # removed. Without it "the memory helps" has no denominator.
        cfg["use_memory"] = False
    if memory_hidden_dim is not None:
        cfg["memory_hidden_dim"] = memory_hidden_dim
    if num_heads is not None:
        # Every run of this project so far has used TitansMACBlock's default of a
        # SINGLE attention head (WorldModel never passed one), i.e. head_dim = dim.
        # That is not only a baseline question: the attention output y_i is what
        # *writes* into the memory, and README 8.1 already traced a rank-1 momentum
        # to attention outputs being near-collinear. One head is the same risk by a
        # different route, so the head count is exposed as a swept axis rather than
        # silently inherited.
        cfg["num_heads"] = num_heads
    if memory_chunk_size is not None:
        # Pinning m while segment_size varies is what isolates "wider attention
        # window" from "longer inner-loop chunk"; left to its default (= seg) the
        # two move together and the arm cannot be interpreted.
        cfg["memory_chunk_size"] = memory_chunk_size
    if eta_init is not None:
        cfg["eta_init"] = eta_init
    if theta_scale is not None:
        cfg["theta_scale"] = theta_scale
    if theta_gate is not None:
        cfg["theta_gate"] = theta_gate
    if normalize_hidden is not None:
        cfg["normalize_hidden"] = normalize_hidden
    if alpha_init is not None:
        # The forget gate's resting value, and under a bounded step (`muon="unit"`)
        # the ONLY thing that sets how far back the memory reaches: every deposit is
        # then the same size, so theta cancels out of the trace-to-interference ratio
        # and retention is governed by alpha alone. Closed form for a target recall
        # distance d writes: alpha* = 1 - sqrt(d/(1+d)) ~ 1/(2d), giving a half-life
        # of 1.39*d. Derived and checked numerically 2026-08-18.
        cfg["alpha_init"] = alpha_init
    if muon is not None:
        # The write rule (titans.memory.WRITE_RULES); sane_ns defaults to "unit".
        cfg["memory_orthogonalize_momentum"] = muon
    if muon_scale is not None:
        # The scale c of psi_c under "unit_squash".
        cfg["memory_orthogonalize_clip"] = muon_scale
    return cfg


def save_frozen(model: WorldModel, cfg: dict, path: str, meta: Optional[dict] = None):
    """Write weights **and** the config that reproduces the architecture."""
    blob = {
        "format": FORMAT,
        "model": model.state_dict(),
        "cfg": cfg,
        "arch": {"dim": model.dim, "depth": len(model.blocks),
                 "segment_size": model.blocks[0].segment_size},
        "meta": meta or {},
    }
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load_frozen(path: str, device: str = "cpu"):
    """``(model, info)`` -- eval mode, no grad anywhere, architecture from the file.

    ``strict=True`` on purpose: a config that does not reproduce the saved parameter
    set must fail here rather than load a partially-initialised model.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "cfg" not in blob:
        raise ValueError(
            f"{path} is a plain checkpoint, not a frozen artifact (no embedded cfg). "
            "Convert it with `python -m titans.worldmodel.frozen --ckpt ... --out ...` "
            "and the flags it was trained with."
        )
    arch = blob["arch"]
    model = WorldModel(latent_dim=arch["dim"], dim=arch["dim"], depth=arch["depth"],
                       segment_size=arch["segment_size"], **blob["cfg"])
    model.load_state_dict(blob["model"], strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = {"cfg": blob["cfg"], "arch": arch, "meta": blob.get("meta", {}),
            "format": blob.get("format")}
    return model, info


def add_config_flags(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The architecture flags, shared with the sweeps so the names never diverge."""
    ap.add_argument("--config", default="sane_ns", choices=sorted(CONFIGS))
    ap.add_argument("--eta-init", type=float, default=None, dest="eta_init")
    ap.add_argument("--theta-scale", type=float, default=None, dest="theta_scale")
    ap.add_argument("--theta-gate", default=None, dest="theta_gate",
                    choices=["softplus", "sigmoid"])
    ap.add_argument("--normalize-hidden", type=float, default=None, dest="normalize_hidden")
    ap.add_argument("--no-memory", action="store_true", dest="no_memory",
                    help="ablate the memory branch entirely (attention-only control)")
    ap.add_argument("--memory-hidden", type=int, default=None, dest="memory_hidden_dim",
                    help="width of the memory MLP's hidden layer (capacity axis)")
    ap.add_argument("--memory-chunk", type=int, default=None, dest="memory_chunk_size",
                    help="inner-loop chunk m; defaults to segment_size")
    ap.add_argument("--heads", type=int, default=None, dest="num_heads",
                    help="attention heads; the block's default is 1, i.e. head_dim = dim")
    return ap


def cfg_from_args(a) -> dict:
    return build_cfg(a.config, eta_init=a.eta_init, theta_scale=a.theta_scale,
                     theta_gate=getattr(a, "theta_gate", None),
                     normalize_hidden=a.normalize_hidden,
                     use_memory=not getattr(a, "no_memory", False),
                     memory_hidden_dim=getattr(a, "memory_hidden_dim", None),
                     memory_chunk_size=getattr(a, "memory_chunk_size", None),
                     num_heads=getattr(a, "num_heads", None))


def main() -> None:
    """Convert a training checkpoint into a frozen artifact, verifying it loads."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint from training.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, required=True, help="latent width of the cache")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--label", default=None)
    add_config_flags(ap)
    a = ap.parse_args()

    cfg = cfg_from_args(a)
    model = WorldModel(latent_dim=a.dim, dim=a.dim, depth=a.depth, segment_size=a.seg,
                       **cfg)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # strict: the flags must reproduce exactly the saved parameter set, which is the
    # check the plain-checkpoint path never performed.
    model.load_state_dict(blob["model"], strict=True)
    meta = dict(blob.get("meta") or {})
    meta.update(source_ckpt=os.path.basename(a.ckpt), train_step=blob.get("step"),
                label=a.label or (blob.get("meta") or {}).get("label"))
    save_frozen(model, cfg, a.out, meta)

    check, info = load_frozen(a.out)
    assert not any(p.requires_grad for p in check.parameters())
    print(json.dumps({"out": a.out, "params": sum(p.numel() for p in check.parameters()),
                      "arch": info["arch"], "meta": info["meta"]}))


if __name__ == "__main__":
    main()
