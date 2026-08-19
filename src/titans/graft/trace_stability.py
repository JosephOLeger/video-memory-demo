"""Where, inside the graft's inner loop, does the memory actually blow up?

``check_qwen`` establishes *that* the branch goes NaN past a few hundred writes.
It cannot say *which* quantity diverges, because it only ever sees the logits.
This module runs the same stream through :mod:`titans.stability`, which probes the
inner loop **per chunk** -- the granularity the recurrence advances at, and the only
one that can see a blow-up that starts and finishes inside a single forward.

The stream is the real one: a real clip through the frozen host, layer-``L`` hidden
states, pushed through the *fitted* PCA basis. Anything measured on random token ids
would be measured on a distribution the model never produces (the mistake this
project already made once), and anything measured on a random ``down`` would be
measuring the projection rather than the memory.

What to read in the output, in order:

* ``theta`` -- the inner learning rate, ``softplus(head(x)) * theta_scale``, which
  has **no upper bound**. If this is what climbs, the fix is a bound on the step.
* ``lambda_hidden_w`` -- the weighted Gram curvature the theory note says governs
  the last layer's block. ``theta * lambda_hidden`` crossing ~2 is the classic
  gradient-descent instability threshold.
* ``rho_hidden`` -- the spectral radius of the 2x2 chunk map. ``> 1`` predicts the
  chunk is expansive; it is the *prediction*, so its first crossing should precede
  the blow-up rather than coincide with it.
* ``w1_fro`` / ``S_norm`` -- the state itself. If these grow smoothly for a long
  while and only then explode, the loop is a slow positive feedback; if they jump
  in one chunk, something upstream saturated.

Use ``--videos`` for anything that claims a stability wall: EgoLife's clips are
consecutive 30 s slices, so a sorted glob is one continuous recording. ``--repeat``
tiles a single clip instead, which makes the probe *easier* -- the memory learns the
loop and the driving gradient dies -- so a plateau measured that way proves nothing.

    python -m titans.graft.trace_stability --videos '<egolife>/**/*.mp4' \\
        --basis cache/qwen_l24_pca128.pt --frames 64 --chunk 8 --out trace.json
"""

from __future__ import annotations

import argparse
import json

import torch

from ..stability import trace_forward
from .branch import MemoryBranch
from .qwen import default_layer, find_decoder_layers


@torch.no_grad()
def host_activations(model, processor, paths, layer: int, *, num_frames: int,
                     max_side: int, prompt: str, device: str) -> torch.Tensor:
    """Layer-``layer`` hidden states ``[1, N, d_model]`` over ``paths``, concatenated.

    EgoLife's clips are consecutive 30 s slices of one recording, so handing this a
    sorted glob gives a **genuinely continuous, non-repeating** stream -- which is the
    only kind that can find a stability wall. Tiling one clip (``--repeat``) does the
    opposite of what it looks like: at period P the memory can *learn the loop*, the
    associative loss collapses, and nothing is left to drive the state anywhere. A
    plateau measured that way proves nothing.

    Kept on the CPU while collecting so the host and the whole stream never have to
    fit on the card at the same time.
    """
    from ..worldmodel.data import load_video_frames
    from .fit_basis import resize_frames, video_inputs

    layers = find_decoder_layers(model)
    grabbed = []

    def hook(_mod, _inp, out):
        grabbed.append((out[0] if isinstance(out, tuple) else out).detach())

    handle = layers[layer].register_forward_hook(hook)
    chunks = []
    try:
        for i, path in enumerate(paths):
            grabbed.clear()
            try:
                frames = resize_frames(load_video_frames(path, num_frames=num_frames),
                                       max_side)
                inputs = video_inputs(processor, frames, prompt, 1.0)
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                model(**inputs)
            except Exception as e:            # one bad file must not kill the sweep
                print(f"  skip {path.split('/')[-1]}: {type(e).__name__}", flush=True)
                continue
            if grabbed:
                chunks.append(grabbed[0].float().cpu())
                print(f"  [{i + 1}/{len(paths)}] {path.split('/')[-1]}: "
                      f"{chunks[-1].shape[1]} tokens", flush=True)
    finally:
        handle.remove()
    if not chunks:
        raise SystemExit("the forward hook never fired -- wrong layer?")
    return torch.cat(chunks, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--video", default=None, help="one clip")
    ap.add_argument("--videos", default=None,
                    help="glob over .mp4; the clips are concatenated in name order. "
                         "EgoLife's are consecutive 30 s slices, so this is a real "
                         "continuous stream -- the only input that can find a wall.")
    ap.add_argument("--num-videos", type=int, default=40, dest="num_videos")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--repeat", type=int, default=1,
                    help="tile the activations this many times. WARNING: this makes "
                         "the probe EASIER, not harder -- a periodic stream lets the "
                         "memory learn the loop and the driving gradient dies. Use "
                         "--videos for a real answer; keep this for reaching write "
                         "counts past the available content, and say so.")
    ap.add_argument("--prompt", default="Describe what happens in this video.")
    ap.add_argument("--override", action="append", default=[], metavar="K=V",
                    help="override a NeuralMemory kwarg on top of SANE, e.g. "
                         "--override normalize_hidden=True --override eta_init=0.9")
    ap.add_argument("--print-every", type=int, default=1, dest="print_every",
                    help="thin the streamed table; the breaking row always prints")
    ap.add_argument("--guard-gb", type=float, default=11.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    layers = find_decoder_layers(model)
    site = default_layer(len(layers)) if a.layer is None else a.layer
    d_model = int(model.config.text_config.hidden_size
                  if hasattr(model.config, "text_config") else model.config.hidden_size)

    if bool(a.videos) == bool(a.video):
        raise SystemExit("pass exactly one of --video / --videos")
    if a.videos:
        import glob as globmod
        paths = sorted(globmod.glob(a.videos, recursive=True))[:a.num_videos]
        if not paths:
            raise SystemExit(f"no video matched {a.videos}")
        print(f"{len(paths)} clips, flux continu", flush=True)
    else:
        paths = [a.video]
    h = host_activations(model, processor, paths, site, num_frames=a.frames,
                         max_side=a.max_side, prompt=a.prompt, device=a.device)
    del model
    torch.cuda.empty_cache()
    if a.repeat > 1:
        h = h.repeat(1, a.repeat, 1)
    print(f"layer {site}: activations {tuple(h.shape)}", flush=True)

    pca = None
    if a.basis:
        blob = torch.load(a.basis, map_location="cpu", weights_only=False)
        pca = blob["pca"]
    over = {}
    for kv in a.override:
        k, _, v = kv.partition("=")
        # A literal, not a string: `normalize_hidden="True"` is truthy either way,
        # but `eta_init="0.9"` would silently break the sigmoid-bias arithmetic.
        try:
            over[k] = json.loads(v.lower() if v in ("True", "False") else v)
        except json.JSONDecodeError:
            over[k] = v
    if over:
        print(f"overrides: {over}", flush=True)
    branch = MemoryBranch(d_model, a.d_mem, chunk_size=a.chunk, down_pca=pca,
                          **over).to(a.device)
    z = branch.down(h.to(a.device))
    del h                          # 2048-wide and the largest thing here; z is 128-wide
    torch.cuda.empty_cache()
    print(f"memory input z: std {float(z.std()):.3g}, "
          f"max |z| {float(z.abs().max()):.3g}", flush=True)

    keys = ("chunk", "token", "theta", "eta", "alpha", "kappa", "lambda_hidden_w",
            "theta_lambda_hidden", "rho_hidden", "h0_norm_max", "w1_fro",
            "w1_spec", "S_norm", "grad_norm_mean", "grad_cv_in_chunk", "finite")
    print("\n" + "  ".join(f"{k:>10.10}" for k in keys), flush=True)

    def show(r, every=a.print_every):
        # Printed as it is produced, not at the end: a long trace is hours of GPU
        # time and a crash used to take every row with it. Thinned so a 13000-chunk
        # run stays readable, but never thin away the row that broke.
        if r["chunk"] % every and r.get("finite", True):
            return
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append(f"{'':>10}" if v is None else
                         (f"{v:>10}" if isinstance(v, (int, bool)) else f"{v:>10.3g}"))
        print("  ".join(cells) + ("  " + r["note"] if r.get("note") else ""), flush=True)

    rows = trace_forward(branch.memory, z, a.chunk, on_row=show)

    finite = [r for r in rows if r.get("finite")]
    print(f"\n{len(finite)}/{len(rows)} chunks finis "
          f"({len(finite) * a.chunk} ecritures avant la casse)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"model": a.model, "layer": site, "chunk": a.chunk,
                       "basis": a.basis, "tokens": int(z.shape[1]),
                       "rows": rows}, f, indent=1)
        print("json:", a.out)


if __name__ == "__main__":
    main()
