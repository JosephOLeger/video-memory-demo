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

**PCA only.** The cache must come from :mod:`titans.worldmodel.make_pca_cache`: the
model runs directly on the fixed PCA basis and has no learned input/output projection
at all, so a raw 1024-D cache has nowhere to go. Standardisation is likewise gone --
the PCA already centres and (whitened) scales the data.

The learned-projection mode was removed on 2026-08-06. At ``dim=64`` over 1024-D
latents 51% of the model's parameters sat in those two projections and only 16% in
the memory, so a memory comparison was largely an I/O-capacity comparison; on a k=128
PCA cache it is 0% and 33%. Results produced in that space are historical and are not
comparable to anything here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .atlas import CONFIGS
from .frozen import build_cfg as _build_cfg, save_frozen
from .harness import memory_health
from .model import WorldModel
from .train import load_latent_cache
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
    """SANE-only base (`sane_ns`) plus the overrides the sweep varies.

    Delegates to :func:`titans.worldmodel.frozen.build_cfg` so that this module,
    ``long_run.py`` and the frozen artifacts cannot drift into building three
    slightly different models from the same flag names.
    """
    return _build_cfg(
        "sane_ns",
        eta_init=getattr(a, "eta_init", None),
        theta_scale=getattr(a, "theta_scale", None),
        normalize_hidden=getattr(a, "normalize_hidden", None),
        cascade=a.cascade, reservoirs=a.reservoirs, leak=a.leak,
        leak_rate=getattr(a, "leak_rate", 1.0), readout=a.readout,
        selective=a.selective, gate=a.gate, topology=a.topology,
        use_memory=not getattr(a, "no_memory", False),
        memory_hidden_dim=getattr(a, "memory_hidden_dim", None),
        memory_chunk_size=getattr(a, "memory_chunk_size", None),
    )


def prepare_data(train_path, val_path):
    """Load both splits and normalise them for training.

    Returns ``(train, val, space)``. The cache must already be PCA-reduced -- it is
    used as-is, being centred and (whitened) scaled by the reduction itself. A raw
    cache is rejected rather than standardised: with no learned input projection there
    is no way to map 1024-D latents onto the model width.
    """
    tr, meta = load_latent_cache(train_path)
    va, _ = load_latent_cache(val_path)
    if "pca" not in meta:
        raise SystemExit(
            f"{train_path} is not a PCA cache. The learned input/output projection was "
            "removed, so the model runs directly in the latent space and the data must "
            "arrive already reduced. Build one with "
            "`python -m titans.worldmodel.make_pca_cache`."
        )
    space = {"kind": "pca", "k": int(meta["k"]), "whiten": bool(meta["whiten"]),
             "explained": float(meta["explained"]),
             "latent_dim_original": int(meta.get("latent_dim_original", 0))}
    return tr, va, space


@torch.no_grad()
def evaluate(model, va: torch.Tensor, T: int, K: int, n_eval: int = 96) -> dict:
    """The [V;V] echo scores for a model, with no training involved.

    Split out of :func:`run` so a *saved* model can be scored -- which is what a
    convergence check needs: the same metric at several training steps, without
    re-training once per point.
    """
    va = va[:n_eval]
    other = torch.roll(va, 1, 0)          # a different held-out video per row
    vv = double(va)                       # [V1 ; V1]
    cross = double(other, va)             # [V2 ; V1]  -- same second half as vv
    f, s = per_copy_l1_K(model, vv, T, K, reset=False)
    _, s_off = per_copy_l1_K(model, vv, T, K, reset=True)
    _, s_cross = per_copy_l1_K(model, cross, T, K, reset=False)
    h = memory_health(model, va[:4])
    return dict(first=f, second=s, echo_gain=f - s,
                second_memoff=s_off, echo_gain_memoff=f - s_off,
                # prefix-swap probe: identical scored content, only the prefix differs
                second_repeat=s, second_cross=s_cross,
                memory_gain=s_cross - s,           # >0 => the memory retained V1
                mem_effect=s_off - s,              # >0 => threaded memory helps
                pr=h["pr"])


def cost_profile(model, T: int, device) -> dict:
    """What this architecture costs, so a memory arm can be compared to a wider window.

    The motivating claim is that keeping long-term memory in *weights* is far cheaper
    than keeping it in *context*. Testing it needs the cost on the record, not
    assumed: an attention window that reaches the same score at 8x the state is not a
    counterexample.

    ``state_bytes`` is the memory carry -- weights plus momentum, **constant in
    sequence length**. ``kv_bytes`` is what the attention window holds for the same
    forward, which grows with ``segment_size``; ``kv_bytes_full`` is what a
    full-context transformer would hold over ``T``, the baseline the claim is really
    against.
    """
    blk = model.blocks[0]
    C, dim, depth = blk.segment_size, model.dim, len(model.blocks)
    el = 4                                            # float32
    if blk.use_memory:
        state, S = blk.memory.init_memory(1)
        sb = sum(v.numel() for v in state.values()) + sum(v.numel() for v in S.values())
        state_bytes = int(sb * el * depth)
    else:
        state_bytes = 0
    # K and V over the block's own window: [persistent ; retrieved ; segment]
    win = blk.persistent(1).shape[1] + (C if blk.use_memory else 0) + C
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "segment_size": C, "attn_window": int(win),
        "state_bytes": state_bytes,
        "kv_bytes": int(2 * win * dim * el * depth),
        "kv_bytes_full": int(2 * T * dim * el * depth),
    }


def run(a, device) -> dict:
    tr, va, space = prepare_data(a.train, a.val)
    T = tr.shape[1]
    K = a.horizon
    tr, va = tr.to(device), va.to(device)
    torch.manual_seed(a.seed)
    cfg = build_cfg(a)
    # The reduction IS the input representation: no learned projection sits between
    # the data and the memory, so dim is dictated by the cache, not by a flag.
    dim = space["k"]
    model = WorldModel(latent_dim=tr.shape[-1], dim=dim, depth=2,
                       segment_size=a.seg, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cost = cost_profile(model, T, device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    # Train on [V;V] via train_loop, which adds resumable checkpointing and
    # optional truncated BPTT. With bptt_window=None and no --ckpt this is the
    # same computation as the original inline loop.
    every = getattr(a, "frozen_every", 0)
    if every and getattr(a, "frozen_out", None):
        # Numbered snapshots along one training run, rather than N separate runs to
        # N step counts: the convergence question is "does this trajectory settle",
        # and re-training per point would cost the sum of the points instead of the
        # largest, and answer a slightly different question (a fresh trajectory each).
        base = a.frozen_out[:-3] if a.frozen_out.endswith(".pt") else a.frozen_out
        done = 0
        while done < a.steps:
            chunk = min(every, a.steps - done)
            train_loop(model, tr, steps=chunk, opt=opt, batch_size=a.batch,
                       transform=double, horizon=K,
                       window=getattr(a, "bptt_window", None),
                       label=a.label, seed=a.seed, horizon_K=K)
            done += chunk
            save_frozen(model, cfg, f"{base}_at{done}.pt",
                        meta={"label": a.label, "seed": a.seed, "steps": done,
                              "horizon_K": K, "train": str(a.train)})
    else:
        train_loop(model, tr, steps=a.steps, opt=opt, batch_size=a.batch,
                   transform=double,
                   horizon=K, window=getattr(a, "bptt_window", None),
                   ckpt_path=getattr(a, "ckpt", None),
                   ckpt_every=getattr(a, "ckpt_every", 0),
                   label=a.label, seed=a.seed, horizon_K=K)
    if getattr(a, "frozen_out", None):
        # Written before evaluation so a cell that dies in eval still leaves a usable
        # artifact -- training is the expensive part and must not have to be redone.
        save_frozen(model, cfg, a.frozen_out,
                    meta={"label": a.label, "seed": a.seed, "steps": a.steps,
                          "horizon_K": K, "train": str(a.train)})

    cost["train_s"] = time.time() - t_train
    cost["s_per_step"] = cost["train_s"] / max(a.steps, 1)
    if str(device).startswith("cuda"):
        cost["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30

    return dict(**evaluate(model, va, T, K, a.n_eval), cost=cost,
                label=a.label, seed=a.seed, horizon=K, T=T, seg=a.seg,
                cascade=a.cascade, reservoirs=a.reservoirs if a.cascade else 0,
                leak=a.leak if a.cascade else None,
                leak_rate=(getattr(a, "leak_rate", 1.0) if a.cascade else None),
                readout=(a.readout if a.cascade else None),
                topology=(a.topology if a.cascade else None),
                use_memory=not getattr(a, "no_memory", False),
                memory_hidden_dim=getattr(a, "memory_hidden_dim", None),
                memory_chunk=getattr(a, "memory_chunk_size", None),
                dim=dim, space=space,
                eta_init=getattr(a, "eta_init", None),
                bptt_window=getattr(a, "bptt_window", None),
                normalize_hidden=getattr(a, "normalize_hidden", None),
                theta_scale=cfg.get("theta_scale"),
                params=cost["params"])


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
    ap.add_argument("--leak-rate", type=float, default=1.0, dest="leak_rate",
                    help="rate of the surprise-driven leak: u <- exp(-r*theta_t) u. "
                         "Below ~0.1 it is indistinguishable from leak=none.")
    ap.add_argument("--readout", default="mean")     # "none" -> leave default
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--gate", default="content")
    ap.add_argument("--topology", default="chain")
    ap.add_argument("--no-memory", action="store_true", dest="no_memory",
                    help="ablate the memory branch: the attention-only control that "
                         "gives 'the memory helps' a denominator")
    ap.add_argument("--memory-hidden", type=int, default=None, dest="memory_hidden_dim")
    ap.add_argument("--memory-chunk", type=int, default=None, dest="memory_chunk_size")
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
    ap.add_argument("--frozen-every", type=int, default=0, dest="frozen_every",
                    help="also save a numbered frozen snapshot every N steps, for a "
                         "convergence check along one trajectory")
    ap.add_argument("--frozen-out", default=None, dest="frozen_out",
                    help="also write a self-describing frozen artifact (frozen.py), "
                         "so downstream inference needs no config flags")
    ap.add_argument("--out", default=None)
    return ap


def main() -> None:
    a = build_parser().parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    r = run(a, dev)
    r["elapsed_s"] = time.time() - t0
    sp = r["space"]
    tag = f"PCA k={sp['k']}"
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
