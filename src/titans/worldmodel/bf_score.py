"""Score cascade artefacts on Benna & Fusi's criterion, not on ``memory_gain``.

The cascade's target is not "beat the scalar arm": it is a memory trace that
decays as ``t^(-1/2)``, the slowest power law that keeps the weight variance
finite (Benna & Fusi 2016, Fig. 2c). This module reads that off a trained
artefact.

Three numbers per artefact:

``slope_design``
    The log-log decay exponent of the readout's impulse response, from the
    architecture alone (K, beta1, g, areas, terminal leak). This is what the
    configuration *asks for*, and it is what the areas fix: -0.066 without them,
    -0.500 with.

``slope_realized``
    The same, with the uniform leak the model actually applies folded in. The
    leak is ``exp(-cascade_leak_rate * theta_t)`` on the whole state, so the
    effective per-step rate is ``leak_rate * <theta_t>`` with ``theta_t`` measured
    on real validation batches -- not the nominal ``leak_rate``. This is the
    number that says whether a *trained* model has a power-law regime at all.

``lifetime``
    Benna-Fusi's memory lifetime: with ``SNR(t) = sqrt(N) r(t) / sqrt(sum_s
    r(s)^2)`` and ``N`` the memory matrix size, the last ``t`` where ``SNR > 1``.
    A pure accumulator scores 0, and so does a chain with no areas.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from typing import Optional

import torch

from titans.metaplastic import cascade_step, geometric_betas
from titans.worldmodel.frozen import load_frozen


def transition(K: int, beta1: float, g: float, *, areas: Optional[float],
               terminal: bool) -> torch.Tensor:
    """The chain's one-step matrix ``B``, read off :func:`cascade_step` itself.

    The dynamics are linear in ``u``, so probing the real step function on the
    canonical basis recovers ``B`` exactly. Doing it this way rather than writing
    ``B`` out by hand is deliberate: the scorer then cannot drift away from the
    implementation it is supposed to be scoring.
    """
    b = geometric_betas(K, beta1, g, dtype=torch.float64)
    a = (torch.tensor([areas ** k for k in range(K)], dtype=torch.float64)
         if areas is not None else None)
    tb = (torch.tensor(beta1 * g ** (-K), dtype=torch.float64)
          if terminal else None)
    zero = torch.zeros((), dtype=torch.float64)
    cols = []
    for j in range(K):
        e = torch.zeros(K, dtype=torch.float64)
        e[j] = 1.0
        cols.append(cascade_step(e, zero, b, a, tb))
    return torch.stack(cols, dim=1)


def impulse_response(K: int, beta1: float, g: float, *, areas: Optional[float],
                     terminal: bool, uniform_leak: float = 0.0,
                     readout: str = "shallow", T: int = 200_000) -> torch.Tensor:
    """``r(t)``: the readout's response to one unit modification at ``t = 0``."""
    B = transition(K, beta1, g, areas=areas, terminal=terminal) * (1.0 - uniform_leak)
    w = _readout_vector(K, readout)
    u = torch.zeros(K, dtype=torch.float64)
    u[0] = 1.0                                  # the injection lands in reservoir 0
    out = torch.empty(T, dtype=torch.float64)
    for t in range(T):
        out[t] = w @ u
        u = B @ u
    return out


def _readout_vector(K: int, mode: str) -> torch.Tensor:
    w = torch.zeros(K, dtype=torch.float64)
    if mode == "deep":
        w[-1] = 1.0
    elif mode in ("mean", "gated", "learned"):
        # `gated`/`learned` are softmax mixtures; their untrained expectation is
        # the mean, which is the only basis-free stand-in available here.
        w[:] = 1.0 / K
    elif mode == "sum":
        w[:] = 1.0
    else:
        w[0] = 1.0                              # "shallow" = Benna-Fusi's u_1
    return w


def loglog_slope(r: torch.Tensor, lo: float, hi: float) -> float:
    t = torch.arange(len(r), dtype=torch.float64)
    m = (t >= lo) & (t <= hi) & (r > 1e-14)
    if int(m.sum()) < 10:
        return float("nan")
    x, y = torch.log(t[m]), torch.log(r[m])
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def lifetime(r: torch.Tensor, n_synapses: int) -> int:
    """Last ``t`` with ``SNR > 1``, the paper's memory lifetime."""
    pos = r.clamp(min=0.0)
    var = float((r ** 2).sum())
    if var <= 0:
        return 0
    snr = math.sqrt(n_synapses) * pos / math.sqrt(var)
    alive = torch.nonzero(snr > 1.0).flatten()
    return int(alive[-1]) if len(alive) else 0


def measure_theta(model, val: torch.Tensor, nseq: int, device) -> dict:
    """Mean/max of the inner-loop step size the trained model actually emits.

    The uniform leak is ``exp(-leak_rate * theta_t)``, so ``<theta_t>`` -- not the
    nominal ``leak_rate`` -- is what sets the exponential envelope on the trace.

    Measured with a hook on ``theta_head`` rather than by re-deriving its input:
    inside a MAC block the memory is fed the *attention output*, not the encoded
    latent, so recomputing it from ``z`` would silently measure a different tensor.
    """
    caught = []

    def hook(mod, _inp, out):
        caught.append(
            (torch.nn.functional.softplus(out) * theta_scale).detach().flatten().cpu()
        )

    handles, theta_scale = [], 1.0
    for blk in model.blocks:
        mem = getattr(blk, "memory", None)
        if mem is None or not hasattr(mem, "theta_head"):
            continue
        theta_scale = mem.theta_scale
        handles.append(mem.theta_head.register_forward_hook(hook))
    try:
        with torch.no_grad():
            for i in range(min(nseq, val.shape[0])):
                model(val[i: i + 1].to(device))
    finally:
        for h in handles:
            h.remove()

    if not caught:
        return {"theta_mean": float("nan"), "theta_max": float("nan")}
    t = torch.cat(caught)
    return {"theta_mean": float(t.mean()), "theta_max": float(t.max())}


def score_one(path: str, val: Optional[torch.Tensor], device, nseq: int) -> dict:
    model, info = load_frozen(path, device=device)
    cfg = info["cfg"] if isinstance(info, dict) and "cfg" in info else info
    row = {"artefact": path.split("/")[-1], "cascade": bool(cfg.get("memory_use_cascade"))}
    if not row["cascade"]:
        return row

    K = int(cfg.get("cascade_reservoirs", 4))
    beta1 = float(cfg.get("cascade_beta1", 0.5))
    g = float(cfg.get("cascade_g", 2.0))
    areas = cfg.get("cascade_areas")
    terminal = bool(cfg.get("cascade_terminal_leak", False))
    readout = cfg.get("cascade_readout") or "shallow"
    leak_rate = float(cfg.get("cascade_leak_rate", 1.0))
    leak_mode = cfg.get("cascade_leak", "full")
    row.update(K=K, beta1=beta1, g=g, areas=areas, terminal_leak=terminal,
               readout=readout, leak=leak_mode, leak_rate=leak_rate)

    n_syn = int(cfg.get("memory_hidden_dim") or cfg.get("dim") or 128) ** 2
    r_design = impulse_response(K, beta1, g, areas=areas, terminal=terminal,
                                readout=readout)
    row["slope_design"] = loglog_slope(r_design, 10, 1e3)
    row["slope_design_far"] = loglog_slope(r_design, 1e3, 1e5)
    row["lifetime_design"] = lifetime(r_design, n_syn)

    # Realized: fold in the uniform leak the trained model actually applies.
    # theta is measured whenever a cache is available, not only when the leak is
    # surprise-gated: <theta_t> is the conversion factor between the `--leak-rate`
    # flag and the effective per-step rate, so it is worth reading off every arm.
    eff = 0.0
    if val is not None:
        st = measure_theta(model, val, nseq, device)
        row.update(st)
        if leak_mode == "surprise" and not math.isnan(st["theta_mean"]):
            eff = 1.0 - math.exp(-leak_rate * st["theta_mean"])
    if leak_mode == "slow":
        eff = 1.0 - float(cfg.get("cascade_slow_decay", 0.99))
    row["effective_leak"] = eff
    if eff > 0:
        r_real = impulse_response(K, beta1, g, areas=areas, terminal=terminal,
                                  uniform_leak=eff, readout=readout)
    else:
        r_real = r_design
    row["slope_realized"] = loglog_slope(r_real, 10, 1e3)
    row["slope_realized_far"] = loglog_slope(r_real, 1e3, 1e5)
    row["lifetime_realized"] = lifetime(r_real, n_syn)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", action="append", required=True,
                    help="glob over artefacts; repeatable")
    ap.add_argument("--val", default=None, help="cache used to measure theta_t")
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = None
    if a.val:
        from titans.worldmodel.train import load_latent_cache
        val, _ = load_latent_cache(a.val)

    paths = sorted(p for pat in a.frozen for p in glob.glob(pat))
    rows = []
    for p in paths:
        try:
            rows.append(score_one(p, val, device, a.nseq))
        except Exception as exc:                      # one bad artefact must not
            rows.append({"artefact": p, "error": repr(exc)})   # stop the sweep

    hdr = f"{'artefact':30s} {'K':>3s} {'areas':>6s} {'term':>5s} {'r_eff':>9s} " \
          f"{'pente concue':>13s} {'pente reelle':>13s} {'duree vie':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r or not r.get("cascade"):
            print(f"{r['artefact'][:30]:30s} {'(pas de cascade)' if 'error' not in r else r['error'][:60]}")
            continue
        print(f"{r['artefact'][:30]:30s} {r['K']:3d} {str(r['areas']):>6s} "
              f"{str(r['terminal_leak'])[:5]:>5s} {r['effective_leak']:9.2e} "
              f"{r['slope_design']:13.3f} {r['slope_realized']:13.3f} "
              f"{r['lifetime_realized']:10d}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\n-> {a.out}")
    return rows


if __name__ == "__main__":
    main()
