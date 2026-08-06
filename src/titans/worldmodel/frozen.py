"""A trained world model, packaged so inference needs no configuration.

The problem this solves. A checkpoint stores only ``state_dict``, so reloading one
means re-specifying the architecture by hand -- for the cascade that is six flags
(``--cascade --reservoirs 6 --leak none --readout gated --selective --gate surprise``)
and getting any of them wrong is *not* reliably an error: several change no parameter
shape, so ``load_state_dict`` succeeds and the model quietly computes something else.
Every inference experiment then silently measures a different architecture.

A frozen artifact carries its own config. :func:`load_frozen` returns a model in
``eval()`` with ``requires_grad_(False)`` everywhere and nothing left to decide::

    model, info = load_frozen("frozen/nocas.pt")

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
              normalize_hidden: Optional[float] = None,
              cascade: bool = False, reservoirs: int = 6, leak: str = "none",
              leak_rate: float = 1.0, readout: str = "gated",
              selective: bool = False, gate: str = "surprise") -> dict:
    """The kwargs a :class:`WorldModel` needs, from the sweep's own flag names.

    Kept in one place so ``echo.py``, ``long_run.py`` and this module cannot drift
    into building three slightly different models from the same flags.
    """
    cfg = dict(CONFIGS[config])
    if eta_init is not None:
        cfg["eta_init"] = eta_init
    if theta_scale is not None:
        cfg["theta_scale"] = theta_scale
    if normalize_hidden is not None:
        cfg["normalize_hidden"] = normalize_hidden
    if cascade:
        cfg.update(memory_use_cascade=True, cascade_reservoirs=reservoirs,
                   cascade_leak=leak, cascade_leak_rate=leak_rate,
                   cascade_gate_signal=gate)
        if readout != "none":
            cfg["cascade_readout"] = readout
        if selective:
            cfg["cascade_selective"] = True
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
    ap.add_argument("--normalize-hidden", type=float, default=None, dest="normalize_hidden")
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    ap.add_argument("--leak", default="none")
    ap.add_argument("--leak-rate", type=float, default=1.0, dest="leak_rate")
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--gate", default="surprise")
    return ap


def cfg_from_args(a) -> dict:
    return build_cfg(a.config, eta_init=a.eta_init, theta_scale=a.theta_scale,
                     normalize_hidden=a.normalize_hidden, cascade=a.cascade,
                     reservoirs=a.reservoirs, leak=a.leak, leak_rate=a.leak_rate,
                     readout=a.readout, selective=a.selective, gate=a.gate)


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
                      "arch": info["arch"], "meta": info["meta"],
                      "cascade": bool(cfg.get("memory_use_cascade"))}))


if __name__ == "__main__":
    main()
