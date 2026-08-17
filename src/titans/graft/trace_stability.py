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

    python -m titans.graft.trace_stability --video clip.mp4 \\
        --basis cache/qwen_l24_pca128.pt --frames 32 --out trace.json
"""

from __future__ import annotations

import argparse
import json

import torch

from ..stability import trace_forward
from .branch import MemoryBranch
from .qwen import default_layer, find_decoder_layers


@torch.no_grad()
def host_activations(model, processor, path: str, layer: int, *, num_frames: int,
                     max_side: int, prompt: str, device: str) -> torch.Tensor:
    """Layer-``layer`` hidden states ``[1, N, d_model]`` for one real clip."""
    from ..worldmodel.data import load_video_frames
    from .fit_basis import resize_frames, video_inputs

    layers = find_decoder_layers(model)
    grabbed = []

    def hook(_mod, _inp, out):
        grabbed.append((out[0] if isinstance(out, tuple) else out).detach())

    handle = layers[layer].register_forward_hook(hook)
    try:
        frames = resize_frames(load_video_frames(path, num_frames=num_frames), max_side)
        inputs = video_inputs(processor, frames, prompt, 1.0)
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        model(**inputs)
    finally:
        handle.remove()
    if not grabbed:
        raise SystemExit("the forward hook never fired -- wrong layer?")
    return grabbed[0].float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--video", required=True)
    ap.add_argument("--basis", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--repeat", type=int, default=1,
                    help="tile the activations this many times, to reach a write "
                         "count past the wall with one clip's worth of VRAM")
    ap.add_argument("--prompt", default="Describe what happens in this video.")
    ap.add_argument("--override", action="append", default=[], metavar="K=V",
                    help="override a NeuralMemory kwarg on top of SANE, e.g. "
                         "--override normalize_hidden=True --override eta_init=0.9")
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

    h = host_activations(model, processor, a.video, site, num_frames=a.frames,
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
    z = branch.down(h)
    print(f"memory input z: std {float(z.std()):.3g}, "
          f"max |z| {float(z.abs().max()):.3g}", flush=True)

    rows = trace_forward(branch.memory, z, a.chunk)
    keys = ("chunk", "token", "theta", "eta", "alpha", "kappa", "lambda_hidden_w",
            "theta_lambda_hidden", "rho_hidden", "h0_norm_max", "w1_fro",
            "w1_spec", "S_norm", "finite")
    print("\n" + "  ".join(f"{k:>10.10}" for k in keys), flush=True)
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append(f"{'':>10}" if v is None else
                         (f"{v:>10}" if isinstance(v, (int, bool)) else f"{v:>10.3g}"))
        print("  ".join(cells), flush=True)

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
