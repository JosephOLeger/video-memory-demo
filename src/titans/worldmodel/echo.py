"""The [V;V] / [V;W] echo test: does the memory hold *this* video's content?

Next-step latent L1 is nearly blind to long-term memory -- it is dominated by local
dynamics, so a working or a broken memory barely moves it (see the README). This is
the video-side benchmark that is *not* blind to it.

**Protocol.** Train on ``[V;V]`` -- a video concatenated with **itself**, so the
sequence is twice the length the attention window can ever see -- predicting ``K``
steps ahead. Then score the *second* copy: if the memory stored the first pass, the
second should be easier. Two controls make the reading unambiguous:

``second_memoff``  the same weights with ``reset_memory_each_segment=True``, so
                   nothing crosses a segment boundary. ``mem_effect = second_memoff
                   - second > 0`` means the threaded memory helps at all.

**The prefix-swap probe (headline).** Score the *same* content under two different
prefixes, so nothing but the memory's contents differs:

    [V1 ; V1]   second half = V1, and the memory has just seen V1
    [V2 ; V1]   second half = V1, but the memory has seen an unrelated V2

Both are pure inference on one trained model, and both are scored on positions
``T..2T`` — the *identical* frames of ``V1``. Hence

    memory_gain = second_cross - second_repeat       (larger = better)

If the memory retained V1, predicting V1 must be easier when V1 is what it holds.

An earlier metric, ``content_specificity = second_VW - second``, compared predicting
``W`` after ``V`` against predicting ``V`` after ``V``. Those two scored halves held
*different content*, so the gap confounded "the memory helps" with "W happens to be
harder to predict than V" -- the probe above exists precisely to remove that
confound, and the old one has been deleted rather than kept as a second ranking.
Result files produced before this change still carry the old field; they are
historical data and are not comparable to ``memory_gain``.

**PCA mode.** Pointed at a cache written by :mod:`titans.worldmodel.make_pca_cache`,
the model runs directly on the fixed PCA basis with **no learned input/output
projection** (``WorldModel(io_projection=False)``), and the standardisation is
skipped -- the PCA already centres and (whitened) scales the data. With a learned
``in_proj``/``out_proj`` at ``dim=64`` over 1024-D latents, 51% of the model's
parameters sit in those two projections and only 16% in the memory; on a k=128 PCA
cache it is 0% and 33%. Detected from the cache metadata rather than a flag, so a
run cannot silently mix the two.

**L1 values in PCA space are not comparable to non-PCA runs.** The reported
``space`` field records which one produced a result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .atlas import CONFIGS
from .harness import _standardize, memory_health
from .model import WorldModel
from .train import load_latent_cache, train_stats
from .training import train_loop


def double(v: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """``[V;V]`` when ``w`` is None, else ``[V;W]`` -- concatenated along time."""
    return torch.cat([v, v if w is None else w], dim=1)


@torch.no_grad()
def per_copy_l1_K(model, seq, T, K, reset, batch=8, margin=4):
    """K-step-ahead L1 on the first and second copy of a doubled sequence.

    ``margin`` skips the first few positions of each copy, where the prediction is
    dominated by the segment boundary rather than by memory.
    """
    model.eval()
    for b in model.blocks:
        b.reset_memory_each_segment = reset
    t1 = t2 = n1 = n2 = 0.0
    for i in range(0, seq.shape[0], batch):
        z = seq[i:i + batch]
        pred = model(z)
        err = (pred[:, :-K] - z[:, K:]).abs().mean(-1)
        a = err[:, margin:T - K]
        c = err[:, T + margin:2 * T - K]
        t1 += a.sum().item(); n1 += a.numel()
        t2 += c.sum().item(); n2 += c.numel()
    for b in model.blocks:
        b.reset_memory_each_segment = False
    return t1 / n1, t2 / n2


def build_cfg(a) -> dict:
    """SANE-only base (`sane_ns`) plus the cascade overrides the sweep varies."""
    cfg = dict(CONFIGS["sane_ns"])            # healthy no-cascade SANE base
    if getattr(a, "eta_init", None) is not None:
        cfg["eta_init"] = a.eta_init
    if getattr(a, "theta_scale", None) is not None:
        cfg["theta_scale"] = a.theta_scale
    if getattr(a, "normalize_hidden", None) is not None:
        # L2-bound the last layer's Gauss-Newton curvature, 2||h_0||^2. Unlike
        # normalize_output (which SANE removes) this leaves M(k)'s scale free.
        cfg["normalize_hidden"] = a.normalize_hidden
    if a.cascade:
        cfg["memory_use_cascade"] = True
        cfg["cascade_reservoirs"] = a.reservoirs
        cfg["cascade_leak"] = a.leak
        cfg["cascade_topology"] = a.topology
        if a.readout != "none":
            cfg["cascade_readout"] = a.readout
        if a.selective:
            cfg["cascade_selective"] = True
        cfg["cascade_gate_signal"] = a.gate
    return cfg


def prepare_data(train_path, val_path):
    """Load both splits and normalise them for training.

    Returns ``(train, val, space)``. A PCA-reduced cache is used as-is (it is
    already centred/whitened); a raw cache gets the per-channel standardisation
    fitted on train. ``space`` describes which, and carries ``dim``/``io_projection``
    for the model.
    """
    tr, meta = load_latent_cache(train_path)
    va, _ = load_latent_cache(val_path)
    if "pca" in meta:
        space = {"kind": "pca", "k": int(meta["k"]), "whiten": bool(meta["whiten"]),
                 "explained": float(meta["explained"]),
                 "latent_dim_original": int(meta.get("latent_dim_original", 0))}
        return tr, va, space
    m_, s_ = train_stats(tr)
    return _standardize(tr, m_, s_), _standardize(va, m_, s_), {"kind": "standardized"}


def run(a, device) -> dict:
    tr, va, space = prepare_data(a.train, a.val)
    T = tr.shape[1]
    K = a.horizon
    tr, va = tr.to(device), va.to(device)
    torch.manual_seed(a.seed)
    cfg = build_cfg(a)
    if space["kind"] == "pca":
        # The reduction IS the input representation: no learned projection sits
        # between the data and the memory, so dim is dictated by the cache.
        dim, io_projection = space["k"], False
    else:
        dim, io_projection = a.dim, True
    model = WorldModel(latent_dim=tr.shape[-1], dim=dim, depth=2,
                       segment_size=a.seg, io_projection=io_projection,
                       **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Train on [V;V] via train_loop, which adds resumable checkpointing and
    # optional truncated BPTT. With bptt_window=None and no --ckpt this is the
    # same computation as the original inline loop.
    train_loop(model, tr, steps=a.steps, opt=opt, batch_size=a.batch,
               transform=double,
               horizon=K, window=getattr(a, "bptt_window", None),
               ckpt_path=getattr(a, "ckpt", None),
               ckpt_every=getattr(a, "ckpt_every", 0),
               label=a.label, seed=a.seed, horizon_K=K)
    va = va[:a.n_eval]
    other = torch.roll(va, 1, 0)          # a different held-out video per row
    vv = double(va)                       # [V1 ; V1]
    cross = double(other, va)             # [V2 ; V1]  -- same second half as vv
    f, s = per_copy_l1_K(model, vv, T, K, reset=False)
    _, s_off = per_copy_l1_K(model, vv, T, K, reset=True)
    _, s_cross = per_copy_l1_K(model, cross, T, K, reset=False)
    h = memory_health(model, va[:4])
    return dict(label=a.label, seed=a.seed, horizon=K, T=T, seg=a.seg,
                cascade=a.cascade, reservoirs=a.reservoirs if a.cascade else 0,
                leak=a.leak if a.cascade else None,
                readout=(a.readout if a.cascade else None),
                topology=(a.topology if a.cascade else None),
                dim=dim, io_projection=io_projection, space=space,
                eta_init=getattr(a, "eta_init", None),
                bptt_window=getattr(a, "bptt_window", None),
                normalize_hidden=getattr(a, "normalize_hidden", None),
                theta_scale=cfg.get("theta_scale"),
                params=sum(p.numel() for p in model.parameters()),
                first=f, second=s, echo_gain=f - s,
                second_memoff=s_off, echo_gain_memoff=f - s_off,
                # prefix-swap probe: identical scored content, only the prefix differs
                second_repeat=s, second_cross=s_cross,
                memory_gain=s_cross - s,           # >0 => the memory retained V1
                mem_effect=s_off - s,              # >0 => threaded memory helps
                pr=h["pr"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-eval", type=int, default=96, dest="n_eval")
    ap.add_argument("--dim", type=int, default=64,
                    help="model width; ignored for a PCA cache, where dim = k")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=4)
    ap.add_argument("--leak", default="none")
    ap.add_argument("--readout", default="mean")     # "none" -> leave default
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--gate", default="content")
    ap.add_argument("--topology", default="chain")
    ap.add_argument("--theta-scale", type=float, default=None, dest="theta_scale",
                    help="inner-step scale; 0.01 fixes the NaN divergence, 0.03 is SANE's default")
    ap.add_argument("--eta-init", type=float, default=None, dest="eta_init",
                    help="resting momentum decay; horizon in tokens is 1/(1-eta)")
    ap.add_argument("--normalize-hidden", type=float, default=None,
                    dest="normalize_hidden",
                    help="target L2 norm for the memory's hidden activation h_0 "
                         "(bounds lambda_max(H_GN^(W1)) = 2||h_0||^2); unset = off")
    ap.add_argument("--bptt-window", type=int, default=None, dest="bptt_window",
                    help="truncate BPTT to this many steps; VRAM becomes independent\n"
                         "of sequence length (measured: 0.82 GiB at T=128..2048)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path; an existing one is resumed from")
    ap.add_argument("--ckpt-every", type=int, default=0, dest="ckpt_every")
    ap.add_argument("--out", default=None)
    return ap


def main() -> None:
    a = build_parser().parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    r = run(a, dev)
    r["elapsed_s"] = time.time() - t0
    sp = r["space"]
    tag = (f"PCA k={sp['k']}" if sp["kind"] == "pca" else "standardized")
    print(f"[{r['elapsed_s']:.0f}s] {a.label} K={r['horizon']} s{r['seed']} "
          f"seg={r['seg']} dim={r['dim']} [{tag}] PR={r['pr']:.1f}", flush=True)
    print(f"  memory_gain {r['memory_gain']:+.4f}  "
          f"(repeat={r['second_repeat']:.4f} cross={r['second_cross']:.4f})  "
          f"mem_effect {r['mem_effect']:+.4f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
    print(f"DONE {a.label} K{r['horizon']} s{r['seed']}", flush=True)


if __name__ == "__main__":
    main()
