"""Phase A: the frozen host's layer-L stream, projected, cached to disk once.

Why this is a separate step, and why it is cheap
------------------------------------------------
The branch reads the decoder's layer-``L`` **output**, and writes only after it.
So the memory's input does not depend on any trainable parameter: it is a constant
of the frozen host and the video. That is the fact the whole training design rests
on -- it means the 3B host runs **once per video, ever**, under ``no_grad``, and
every subsequent epoch reads a tensor off disk.

What is cached is ``z = down(h)``, not ``h``: ``down`` is the frozen PCA basis, so
it is part of the constant too, and ``z`` is ``d_mem``-wide instead of ``d_model``.
For a 7 h day at 16 frames per 30 s clip that is ~540 k tokens, i.e. **135 MB in
fp16** rather than 2.2 GB.

Clip boundaries are stored alongside. Training samples windows of *clips*, so it
needs to slice the stream at the same places the video was cut -- and EgoLife's
clips are consecutive 30 s slices of one recording, so a sorted glob is one
continuous timeline and the offsets are its index.

    python -m titans.graft.cache_z \
        --videos '<egolife>/A1_JAKE/DAY1/*.mp4' \
        --basis cache/qwen_l24_pca128.pt --frames 16 --out cache/jake_day1_z.pt
"""

from __future__ import annotations

import argparse
import glob as globmod
import os

import torch

from .branch import fold_pca
from .qwen import default_layer, find_decoder_layers


@torch.no_grad()
def cache_clips(model, processor, paths, layer: int, *, num_frames: int,
                max_side: int, prompt: str, device: str, W=None, b=None,
                report_every: int = 25) -> dict:
    """``z`` for every clip in ``paths``, concatenated, with clip boundaries.

    ``W``/``b`` are the folded PCA (:func:`titans.graft.branch.fold_pca`). Passing
    them here rather than projecting later keeps the 2048-wide tensor off the CPU
    entirely -- only the ``k``-wide result is ever moved or kept.

    A clip that fails to decode is **skipped and recorded**, not fatal: one bad file
    in 828 must not cost the whole day's pass.
    """
    from ..worldmodel.data import load_video_frames
    from .fit_basis import resize_frames, video_inputs

    layers = find_decoder_layers(model)
    grabbed = []

    def hook(_mod, _inp, out):
        grabbed.append((out[0] if isinstance(out, tuple) else out).detach())

    handle = layers[layer].register_forward_hook(hook)
    chunks, offsets, kept, skipped = [], [0], [], []
    fps = num_frames / 30.0                      # EgoLife clips are 30 s slices
    try:
        for i, path in enumerate(paths):
            grabbed.clear()
            try:
                frames = resize_frames(load_video_frames(path, num_frames=num_frames),
                                       max_side)
                inputs = video_inputs(processor, frames, prompt, fps)
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                model(**inputs)
            except Exception as e:
                skipped.append((os.path.basename(path), type(e).__name__))
                print(f"  saut {os.path.basename(path)}: {type(e).__name__}", flush=True)
                continue
            if not grabbed:
                raise SystemExit("le hook ne s'est jamais declenche -- mauvaise couche ?")
            h = grabbed[0].float()                            # [1, n, d_model]
            z = h if W is None else (h @ W.T.to(h) + b.to(h))  # [1, n, k]
            chunks.append(z[0].half().cpu())
            offsets.append(offsets[-1] + chunks[-1].shape[0])
            kept.append(path)
            del h, z
            if (i + 1) % report_every == 0 or i + 1 == len(paths):
                print(f"  [{i + 1}/{len(paths)}] {offsets[-1]} tokens cumules",
                      flush=True)
    finally:
        handle.remove()
    if not chunks:
        raise SystemExit("aucun clip n'a pu etre lu")
    return {"z": torch.cat(chunks, dim=0), "offsets": torch.tensor(offsets),
            "clips": kept, "skipped": skipped, "layer": layer,
            "frames_per_clip": num_frames, "max_side": max_side, "prompt": prompt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--videos", required=True, help="glob; les clips sont concatenes "
                                                    "dans l'ordre des noms")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--basis", default=None, help="base PCA figee (fortement conseille)")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--frames", type=int, default=16, dest="frames",
                    help="images par clip de 30 s. 16 = 0.53 im/s = ~22 tokens/s de "
                         "video, ce qui fixe le write_stride et alpha.")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--prompt", default="Describe what happens in this video.")
    ap.add_argument("--guard-gb", type=float, default=10.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    paths = sorted(globmod.glob(a.videos, recursive=True))
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise SystemExit(f"aucune video pour {a.videos}")
    print(f"{len(paths)} clips", flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer

    W = b = None
    if a.basis:
        pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
        W, b = fold_pca(pca)
        W, b = W.to(a.device), b.to(a.device)
        print(f"base PCA k={W.shape[0]} sur D={W.shape[1]}", flush=True)

    blob = cache_clips(model, processor, paths, site, num_frames=a.frames,
                       max_side=a.max_side, prompt=a.prompt, device=a.device, W=W, b=b)
    blob["basis"] = a.basis
    blob["model"] = a.model
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(blob, a.out)
    n, k = blob["z"].shape
    print(f"\n{len(blob['clips'])} clips gardes, {len(blob['skipped'])} sautes")
    print(f"z {tuple(blob['z'].shape)} fp16 -> {a.out} "
          f"({n * k * 2 / 2**20:.0f} Mio, {n / len(blob['clips']):.0f} tokens/clip)")


if __name__ == "__main__":
    main()
