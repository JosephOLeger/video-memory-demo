"""Does the momentum state stay bounded as the sequence gets long?

The scalar recurrence ``S_t = eta_t S_{t-1} - theta_t G_t`` decays as ``eta^m`` per
chunk, so ``||S||`` reaches a fixed point set by ``1/(1 - eta)``. The cascade at
``cascade_leak="none"`` has no such fixed point: ``u_t = B_t u_{t-1} + e_1 g_t`` with
``B`` doubly stochastic, so ``rho(B) = 1`` exactly (the constant vector is the
conserved total mass) and a sustained run of same-sign gradients grows the total
**linearly in t**. Nothing inside the cascade removes mass -- only the memory's own
forget gate ``(1 - alpha_t)`` downstream does, and at ``alpha_init=0.002`` that is a
346-token half-life against a state that never stops filling.

Every sequence this project has run is 64-512 tokens. That is short enough that a
linear ramp is indistinguishable from a fixed point, which is why the cascade has
looked unconditionally stable. This module pushes the length instead of the step
size, which is the axis the theory says the cascade's own condition lives on.

Why forward-only, and why that is not a cop-out. The question is a property of the
recurrence, not of the outer optimisation: given a realistic token stream, does the
state stay bounded? Dropping backprop removes the length ceiling entirely (no
activation graph), so ``T = 16384`` costs the same memory as ``T = 64`` and the two
are directly comparable. It also removes the confound that killed the ``theta``
sweep's interpretation -- with training in the loop, an arm can look stable simply
because the outer loop drove ``theta_t`` to zero and the memory stopped working.

The token stream is **real**, not synthetic: MQAR batches are pushed through the
model and the actual ``y_i`` entering ``memory.update`` is captured, then several
independent batches are concatenated to reach ``T``. Feeding Gaussian noise instead
would change the gradient statistics, and it is precisely a sustained same-sign run
of gradients that the linear-growth argument needs.

    python -m titans.membench.leak_length --cascade --leak none --T 4096
    python -m titans.membench.leak_length --cascade --leak surprise --leak-rate 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from ..worldmodel.atlas import CONFIGS
from .model import RecallModel
from .tasks import associative_recall

SEG, DIM = 8, 32


@torch.no_grad()
def capture_stream(model, make_task, tokens: int, device) -> torch.Tensor:
    """The real ``[B, T, dim]`` stream entering ``memory.update``, length >= tokens.

    Each MQAR batch yields ``N`` tokens of memory input; independent batches are
    concatenated until ``tokens`` is reached. Concatenating independent batches
    rather than tiling one keeps the stream non-periodic -- a tiled stream would
    let the state settle into a cycle and hide exactly the accumulation we are
    looking for.
    """
    block = model.blocks[0]
    mem = block.memory
    seen: list[torch.Tensor] = []
    real_update = mem.update

    def spy(y, state, S, chunk):
        seen.append(y.detach())
        return real_update(y, state, S, chunk)

    mem.update = spy
    try:
        step, total = 0, 0
        while total < tokens:
            b = make_task(1_000_000 + step)
            model(b.x.to(device))
            total = sum(t.shape[1] for t in seen)
            step += 1
    finally:
        mem.update = real_update
    return torch.cat(seen, dim=1)[:, :tokens]


@torch.no_grad()
def run_length(mem, x: torch.Tensor, chunk: int, every: int = 1) -> list[dict]:
    """Thread the memory recurrence over ``x``, recording state norms per chunk.

    Returns one record per sampled chunk: the token index, ``||S||`` (the whole
    cascade state, all reservoirs), ``||read(S)||`` (the part that actually reaches
    the weights) and ``||W1||``. Sampling every ``every`` chunks keeps a 16k-token
    trace from being 2048 records long while preserving the shape of the curve.
    """
    k, v, q, alpha, eta, theta = mem._project(x)[:6]
    gamma = mem._gamma(x)
    B, N = x.shape[0], x.shape[1]
    # init_memory, not zeros_like(state): under the cascade S carries a trailing
    # reservoir axis [..., K] that the weight tensors do not have.
    state, S = mem.init_memory(B)
    cascade = bool(getattr(mem, "use_cascade_momentum", False))
    out = []
    for c, start in enumerate(range(0, N, chunk)):
        end = min(start + chunk, N)
        state, S, _ = mem._scan_chunked(
            k[:, start:end], v[:, start:end], alpha[:, start:end],
            eta[:, start:end], theta[:, start:end], state, S, end - start,
            q=None, gamma=None if gamma is None else gamma[:, start:end],
            x=x[:, start:end],
        )
        finite = bool(torch.isfinite(state["w1"]).all())
        if c % every == 0 or not finite or end == N:
            s_all = math.sqrt(sum(float(t.double().pow(2).sum()) for t in S.values()))
            # read(u) = u[..., 0] under the chain topology; the scalar path has no
            # reservoir axis, in which case the visible momentum *is* the state.
            vis = [t[..., 0] if cascade and t.dim() > 2 else t for t in S.values()]
            s_vis = math.sqrt(sum(float(t.double().pow(2).sum()) for t in vis))
            out.append({
                "chunk": c, "token": end,
                "S_norm": s_all, "S_visible": s_vis,
                "w1_norm": float(state["w1"].double().norm()),
                "theta_mean": float(theta[:, start:end].mean()),
                "alpha_mean": float(alpha[:, start:end].mean()),
                "finite": finite,
            })
        if not finite:
            break
    return out


def growth_exponent(trace: list[dict], key: str = "S_norm") -> float:
    """Slope of ``log ||S||`` against ``log t`` over the second half of the trace.

    The second half only: the first chunks are a transient from the zero initial
    state, where every recurrence looks linear regardless of its fixed point. A
    slope near 1 is the unbounded accumulator, near 0 a bounded fixed point.
    """
    pts = [(r["token"], r[key]) for r in trace if r["finite"] and r[key] > 0]
    pts = pts[len(pts) // 2:]
    if len(pts) < 3:
        return float("nan")
    xs = [math.log(t) for t, _ in pts]
    ys = [math.log(s) for _, s in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return float("nan") if den == 0 else sum(
        (x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="sane_ns", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=4096, dest="T",
                    help="stream length in memory-update tokens")
    ap.add_argument("--seg", type=int, default=SEG)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--num-pairs", type=int, default=8, dest="num_pairs")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--every", type=int, default=8, help="record every Nth chunk")
    ap.add_argument("--theta-scale", type=float, default=None, dest="theta_scale")
    ap.add_argument("--eta-init", type=float, default=None, dest="eta_init")
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--reservoirs", type=int, default=6)
    ap.add_argument("--leak", default="none")
    ap.add_argument("--leak-rate", type=float, default=1.0, dest="leak_rate")
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--gate", default="surprise")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    def make_task(seed):
        return associative_recall(batch=a.batch, num_pairs=a.num_pairs,
                                  segment_size=a.seg, key_vocab=a.vocab,
                                  val_vocab=a.vocab, seed=seed)

    cfg = dict(CONFIGS[a.config])
    if a.theta_scale is not None:
        cfg["theta_scale"] = a.theta_scale
    if a.eta_init is not None:
        cfg["eta_init"] = a.eta_init
    if a.chunk is not None:
        cfg["memory_chunk_size"] = a.chunk
    if a.cascade:
        cfg.update(memory_use_cascade=True, cascade_reservoirs=a.reservoirs,
                   cascade_leak=a.leak, cascade_gate_signal=a.gate,
                   cascade_leak_rate=a.leak_rate)
        if a.readout != "none":
            cfg["cascade_readout"] = a.readout
        if a.selective:
            cfg["cascade_selective"] = True

    torch.manual_seed(a.seed)
    t0 = time.time()
    sample = make_task(0)
    model = RecallModel(sample.vocab_size, dim=DIM, depth=2, segment_size=a.seg,
                        num_heads=1, memory_num_layers=2, **cfg).to(a.device)
    model.eval()

    x = capture_stream(model, make_task, a.T, a.device)
    mem = model.blocks[0].memory
    chunk = a.chunk if a.chunk is not None else a.seg
    trace = run_length(mem, x, chunk, every=a.every)

    last = trace[-1]
    out = {
        "label": a.label or ("cas_" + a.leak if a.cascade else "nocas"),
        "config": a.config, "seed": a.seed, "T": a.T, "seg": a.seg, "chunk": chunk,
        "cascade": a.cascade, "leak": a.leak if a.cascade else None,
        "leak_rate": a.leak_rate if a.cascade else None,
        "theta_scale": cfg.get("theta_scale"), "eta_init": cfg.get("eta_init"),
        "reached": last["token"], "finite": last["finite"],
        "S_final": last["S_norm"], "S_visible_final": last["S_visible"],
        "w1_final": last["w1_norm"],
        "slope_S": growth_exponent(trace, "S_norm"),
        "slope_visible": growth_exponent(trace, "S_visible"),
        "trace": trace,
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "trace"}), flush=True)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
