"""How far inside its stability threshold does a trained model actually sit?

The inner loop's chunk map is expansive once ``theta_t * lambda_max`` crosses ~2.37
(README; ``stability.chunk_transfer``). Both factors are properties of the *trained*
model, not of the config: ``theta_t = theta_scale * softplus(w_theta.x + b_theta)``
is a learned head, and ``lambda_max`` is the weighted Gram of the hidden activations
``h_0``, which are computed from online-updated weights. So the margin cannot be
predicted from flags -- it has to be read off a checkpoint on real data.

This is the stability axis that needs no run to be pushed to failure. Driving
``theta_scale`` up until training breaks measures the same thing, but destroys the
run to do it; the margin is free and is available for every arm.

Reported per chunk of a real forward, then reduced::

    theta_lambda_hidden_max   worst chunk of the pass
    margin = 2.37 / that      >1 means inside the threshold, <1 means past it

Alongside it, the health quantities that distinguish "stable" from "quietly dead"
(``pr``, ``cos_ret``), because a memory can sit deep inside the threshold precisely
by having stopped carrying anything.

    python -m titans.worldmodel.margin --frozen frozen/cas_tag.pt \\
        --val cache/egolife_pca128_val.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from ..stability import trace_forward
from .frozen import load_frozen
from .harness import memory_health
from .train import load_latent_cache

THRESHOLD = 2.37


@torch.no_grad()
def memory_input_stream(model, z: torch.Tensor) -> torch.Tensor:
    """The tensor actually fed to ``memory.update``, captured from a real forward.

    Not the latents: the memory sees the *attention output* ``y_i``, which depends on
    what the memory itself already retrieved. Probing on the raw latents would measure
    a curvature the memory never experiences.
    """
    mem = model.blocks[0].memory
    seen: list[torch.Tensor] = []
    real = mem.update

    def spy(y, state, S, chunk):
        seen.append(y.detach())
        return real(y, state, S, chunk)

    mem.update = spy
    try:
        model(z)
    finally:
        mem.update = real
    return torch.cat(seen, dim=1)


@torch.no_grad()
def measure(model, z: torch.Tensor, info: dict) -> dict:
    blk, mem = model.blocks[0], model.blocks[0].memory
    x = memory_input_stream(model, z)
    tr = trace_forward(mem, x, blk.memory_chunk_size, b=0)
    fin = [r for r in tr if r.get("finite") and "theta" in r]
    if not fin:
        return {"all_finite": False}

    def col(k):
        return [r[k] for r in fin if r.get(k) is not None]

    tl = col("theta_lambda_hidden")
    h = memory_health(model, z[:4])
    cfg = info["cfg"]
    return {
        "all_finite": all(r.get("finite") for r in tr),
        "chunks": len(tr), "tokens": int(x.shape[1]),
        "theta_lambda_hidden_max": max(tl),
        "theta_lambda_hidden_mean": sum(tl) / len(tl),
        "margin": THRESHOLD / max(tl),
        "kappa": fin[0]["kappa"],
        "theta_t_min": min(col("theta")), "theta_t_max": max(col("theta")),
        "alpha_t_mean": sum(col("alpha")) / len(col("alpha")),
        "eta_t_mean": sum(col("eta")) / len(col("eta")),
        "lambda_hidden_max": max(col("lambda_hidden_w")),
        "lambda_keys_max": max(col("lambda_keys_w")),
        "h0_norm_max": max(col("h0_norm_max")),
        # health: a huge margin reached by having no memory left is not stability
        "pr": h["pr"], "w1_norm": h["norm"], "cos_ret": h["cos_ret"],
        "cascade": bool(cfg.get("memory_use_cascade")),
        "leak": cfg.get("cascade_leak"), "leak_rate": cfg.get("cascade_leak_rate"),
        "eta_init": cfg.get("eta_init"), "theta_scale": cfg.get("theta_scale"),
        # The two axes of the dilution study: which reservoirs are read, and how fast
        # reservoir 0 drains into the deeper ones. Without them the audit rows are only
        # separable by label, which is one rename away from being unreadable.
        "readout": cfg.get("cascade_readout"), "beta1": cfg.get("cascade_beta1"),
        "reservoirs": cfg.get("cascade_reservoirs"),
        "label": info["meta"].get("label"), "seed": info["meta"].get("seed"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", required=True, action="append",
                    help="a frozen artifact, or a glob matching several; repeatable, "
                         "so one audit can span several sweep directories")
    ap.add_argument("--val", required=True)
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--out", default=None, help="write one JSON list for all inputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    va, meta = load_latent_cache(a.val)
    if "pca" not in meta:
        raise SystemExit(f"{a.val} is not a PCA cache.")
    z = va[:a.nseq].to(a.device)

    # A glob that matches nothing is a directory the sweep has not filled yet, not an
    # error: the audit runs on a timer while cells are still landing. A literal path
    # that does not exist IS an error, and falls through to the per-artifact handler.
    paths = []
    for pat in a.frozen:
        hits = sorted(glob.glob(pat))
        if hits:
            paths.extend(hits)
        elif not any(c in pat for c in "*?["):
            paths.append(pat)
    paths = list(dict.fromkeys(paths))
    out = []
    for p in paths:
        try:
            model, info = load_frozen(p, a.device)
            rec = measure(model, z, info)
        except Exception as e:                       # one bad artifact must not
            rec = {"error": repr(e)[:200]}           # abort a 46-cell sweep
        rec["frozen"] = os.path.basename(p)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
