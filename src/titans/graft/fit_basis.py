"""Fit the memory's input basis on the frozen host's own layer-L activations.

Why a fitted basis and not a learned projection
-----------------------------------------------
Two reasons, and the second one is now measured rather than argued.

*By design*: fixing a PCA makes the reduction a property of the **data** instead of
a bundle of parameters free to co-adapt with the memory -- the move that removed
51 % of the world model's parameters and made "what did the memory learn?" answerable.

*By necessity*: a decoder's residual stream carries a few channels with enormous
magnitudes. Fed through a random projection they saturate the memory's inner loop,
and it **diverges** -- measured on Qwen2.5-VL-3B layer 24, identity holds at 64
writes and is NaN by 256. ``fit_pca`` standardises per channel before the SVD,
which is exactly the pathology that has to be removed before the memory sees
anything.

The activations are collected on **real video**, not on random tokens: the basis
has to describe the distribution the graft will actually meet, and egocentric video
is not text. Frames are decoded with the repo's own ``load_video_frames``.

    python -m titans.graft.fit_basis --videos '<egolife>/**/*.mp4' \\
        --k 128 --num-videos 24 --out cache/qwen_layer24_pca128.pt
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import time

import torch

from ..worldmodel.bottleneck import fit_pca
from ..worldmodel.data import load_video_frames
from .qwen import default_layer, find_decoder_layers


def resize_frames(frames: torch.Tensor, max_side: int) -> torch.Tensor:
    """Cap the longest side of ``[F, H, W, C]`` uint8 frames.

    Qwen2.5-VL uses *dynamic* resolution: it does not downscale for you, so native
    EgoLife frames turn 32 frames into tens of thousands of visual tokens and the
    vision tower asks for 11 GB in a single allocation. The basis does not need
    resolution -- it needs the *distribution* of layer-L activations -- so the cap
    is cheap here in a way it would not be at evaluation time.
    """
    f, h, w, c = frames.shape
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frames
    new_h, new_w = max(2, int(round(h * scale))), max(2, int(round(w * scale)))
    x = frames.permute(0, 3, 1, 2).float()                  # [F, C, H, W]
    x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear",
                                        align_corners=False, antialias=True)
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def video_inputs(processor, frames: torch.Tensor, prompt: str, fps: float):
    """One video turned into the exact dict the model's forward expects."""
    messages = [{"role": "user", "content": [{"type": "video"},
                                             {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    arr = frames.round().clamp(0, 255).to(torch.uint8).numpy()
    try:
        return processor(text=[text], videos=[arr], fps=fps, return_tensors="pt")
    except TypeError:                 # older processors do not take fps
        return processor(text=[text], videos=[arr], return_tensors="pt")


@torch.no_grad()
def collect(model, processor, paths, layer: int, *, num_frames: int, fps: float,
            per_video: int, prompt: str, device: str,
            max_side: int = 256) -> torch.Tensor:
    """Layer-``layer`` hidden states over ``paths``, subsampled to ``per_video`` each.

    Subsampling is uniform over the token axis rather than taking a prefix: the
    video tokens sit in one contiguous span and a prefix would fit the basis to the
    system prompt.
    """
    layers = find_decoder_layers(model)
    grabbed = []

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grabbed.append(h.detach()[0].float().cpu())     # [N, d_model]

    handle = layers[layer].register_forward_hook(hook)
    chunks, used = [], []
    try:
        for p in paths:
            grabbed.clear()
            try:
                frames = resize_frames(load_video_frames(p, num_frames=num_frames),
                                       max_side)
                inputs = video_inputs(processor, frames, prompt, fps)
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                model(**inputs)
            except Exception as e:                       # a bad file must not stop the fit
                print(f"  skip {p.split('/')[-1]}: {type(e).__name__}: {e}", flush=True)
                continue
            if not grabbed:
                continue
            h = grabbed[0]
            n = h.shape[0]
            idx = torch.linspace(0, n - 1, min(per_video, n)).round().long()
            chunks.append(h.index_select(0, idx))
            used.append(p)
            print(f"  {p.split('/')[-1]}: {n} tokens -> kept {len(idx)}", flush=True)
    finally:
        handle.remove()
    if not chunks:
        raise SystemExit("no activations collected -- check the video paths")
    print(f"collected from {len(used)}/{len(paths)} videos", flush=True)
    return torch.cat(chunks, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--videos", required=True, help="glob over .mp4 files")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--num-videos", type=int, default=24, dest="num_videos")
    ap.add_argument("--num-frames", type=int, default=32, dest="num_frames")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--per-video", type=int, default=512, dest="per_video")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side",
                    help="cap the frame's longest side; native EgoLife frames OOM "
                         "the vision tower at 32 frames")
    ap.add_argument("--prompt", default="Describe what happens in this video.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host

    paths = sorted(globmod.glob(a.videos, recursive=True))[:a.num_videos]
    if not paths:
        raise SystemExit(f"no video matched {a.videos}")
    print(f"{len(paths)} videos", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    layers = find_decoder_layers(model)
    site = default_layer(len(layers)) if a.layer is None else a.layer
    d_model = int(model.config.text_config.hidden_size
                  if hasattr(model.config, "text_config") else model.config.hidden_size)
    print(f"layer {site} of {len(layers)}, d_model={d_model}", flush=True)

    t0 = time.time()
    X = collect(model, processor, paths, site, num_frames=a.num_frames, fps=a.fps,
                per_video=a.per_video, prompt=a.prompt, device=a.device,
                max_side=a.max_side)
    print(f"activations {tuple(X.shape)} in {time.time() - t0:.0f}s", flush=True)
    # The pathology this basis exists to remove, before and after, in one number.
    ch_std = X.std(0)
    print(f"per-channel std: min {ch_std.min():.3g} median {ch_std.median():.3g} "
          f"max {ch_std.max():.3g}  (ratio {ch_std.max() / ch_std.median():.0f}x)",
          flush=True)

    pca = fit_pca(X, a.k)
    print(f"PCA k={a.k}: {100 * pca['explained']:.1f} % of the variance", flush=True)
    torch.save({"pca": pca, "model": a.model, "layer": site, "d_model": d_model,
                "videos": [p.split("/")[-1] for p in paths],
                "num_frames": a.num_frames, "fps": a.fps, "max_side": a.max_side,
                "tokens": int(X.shape[0]),
                "channel_std_ratio": float(ch_std.max() / ch_std.median())}, a.out)
    print("saved:", a.out)
    print(json.dumps({"k": a.k, "explained": pca["explained"], "layer": site,
                      "tokens": int(X.shape[0])}))


if __name__ == "__main__":
    main()
