"""Steps 2 and 3 of the graft: prove the identity, then measure what it costs.

Nothing here trains. It answers three questions on the real model, in one pass:

1. **Is the grafted model still the base model?** With ``up`` zero-init the logits
   must be equal *bit for bit*, not close. Any wiring mistake -- a dropped
   ``attention_mask``, a mangled cache tail, grafting onto the vision tower --
   shows up as a non-zero difference. This is the cheapest correctness test the
   project has and it only works before training starts.
2. **What does the branch cost in VRAM?** Measured at the real token counts with
   ``torch.cuda.max_memory_allocated``, base and grafted, forward and backward.
   Extrapolating this number was the alternative; measuring it is not expensive.
3. **What is the parameter bill?** As a share of the host, for the paper's
   "we added 0.0x % and touched nothing else".

Two input modes, and the difference matters. Random token ids give exact control
over the sequence length, which is what the VRAM sweep needs -- but they produce
activations from a distribution the model never sees, so **nothing about numerical
stability may be concluded from them**. ``--video`` runs a real clip through the
vision tower instead; that is the mode that settles whether the memory diverges.

    python -m titans.graft.check_qwen --tokens 512,2048 --out graft_check.json
    python -m titans.graft.check_qwen --video clip.mp4 --frames 16,32,64 \\
        --basis cache/qwen_l24_pca128.pt --write-stride 1,8
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .branch import MemoryBranch
from .qwen import branch_report, default_layer, find_decoder_layers, graft, ungraft


def load_host(model_id: str, dtype: torch.dtype, device: str):
    """The frozen VLM. Its text decoder is all the graft touches.

    Frozen here, at load, and not later at ``graft()``: an unfrozen host would make
    the base backward a full 3.75 B fine-tune, which is not the thing the graft is
    being compared against and whose peak is *larger* than the grafted one. The
    honest pair is "host forward" (what inference costs) against "grafted backward"
    (what training the branch costs).
    """
    from transformers import AutoModelForCausalLM

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except ImportError:                       # a host that is not Qwen2.5-VL
        Cls = AutoModelForCausalLM
    model = Cls.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def text_inputs(model, n_tokens: int, device: str, batch: int = 1) -> dict:
    """A token stream of exactly ``n_tokens``, inside the embedding table.

    Random ids give control over the length, which is what the VRAM sweep needs.
    They are *not* a substitute for real input when the question is numerical
    stability: garbage tokens produce activations from a distribution the model
    never sees, so anything measured about divergence has to be re-measured on
    video before it is believed. Use ``--videos`` for that.
    """
    vocab = int(getattr(model.config, "vocab_size", None)
                or model.config.text_config.vocab_size)
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(0, vocab, (batch, n_tokens), generator=g).to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def video_inputs_for(model_id: str, path: str, n_frames: int, max_side: int,
                     device: str, processor=None) -> dict:
    """One real video, as the model's forward expects it."""
    from transformers import AutoProcessor

    from .fit_basis import resize_frames, video_inputs
    from ..worldmodel.data import load_video_frames

    processor = processor or AutoProcessor.from_pretrained(model_id)
    frames = resize_frames(load_video_frames(path, num_frames=n_frames), max_side)
    inputs = video_inputs(processor, frames, "Describe what happens in this video.", 1.0)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


@torch.no_grad()
def _forward_peak(model, inputs, device) -> tuple:
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    logits = model(**inputs).logits
    torch.cuda.synchronize(device)
    return logits.float().clone(), torch.cuda.max_memory_allocated(device), time.time() - t0


def _backward_peak(model, inputs, device) -> tuple:
    """Peak with a real backward -- what training will actually pay.

    Returns ``(peak, seconds)``, or ``(None, None)`` on OOM: the point of this
    sweep is to find where the budget runs out, so hitting the wall is a result to
    record, not a crash to propagate.
    """
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        out = model(**inputs).logits
        out.float().square().mean().backward()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None, None
    model.zero_grad(set_to_none=True)
    return peak, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--layer", type=int, default=None, help="default: 2/3 of depth")
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--basis", default=None,
                    help="a basis from titans.graft.fit_basis; without it `down` is "
                         "random and the memory diverges on raw activations")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--write-stride", default="1", dest="write_stride",
                    help="comma-separated strides; the grid is tokens x strides")
    ap.add_argument("--tokens", default="256,1024",
                    help="comma-separated token counts to measure at")
    ap.add_argument("--guard-gb", type=float, default=14.0, dest="guard_gb",
                    help="refuse to start unless this much VRAM is free")
    ap.add_argument("--video", default=None,
                    help="a real .mp4; the only input that settles a question about "
                         "numerical stability")
    ap.add_argument("--frames", default="16,32,64",
                    help="with --video: frame counts, in place of --tokens")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dtype = getattr(torch, a.dtype)
    dev = a.device
    if dev.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(dev)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres "
                             "(les cartes sont partagees)")
    print(f"loading {a.model} ({a.dtype}) ...", flush=True)
    model = load_host(a.model, dtype, dev)
    layers = find_decoder_layers(model)
    site = default_layer(len(layers)) if a.layer is None else a.layer
    d_model = int(model.config.text_config.hidden_size
                  if hasattr(model.config, "text_config") else model.config.hidden_size)
    host_params = sum(p.numel() for p in model.parameters())
    print(f"host: {len(layers)} decoder layers, d_model={d_model}, "
          f"{host_params/1e9:.2f} B params; graft site = layer {site}", flush=True)

    pca = None
    if a.basis:
        blob = torch.load(a.basis, map_location="cpu", weights_only=False)
        pca = blob["pca"]
        if blob.get("layer") != site:
            raise SystemExit(f"basis was fitted at layer {blob.get('layer')}, "
                             f"the graft site is {site}")
        print(f"basis: k={pca['k']}, {100 * pca['explained']:.1f} % of the variance, "
              f"fitted on {blob.get('tokens')} tokens of real video", flush=True)

    branch = MemoryBranch(d_model, a.d_mem, chunk_size=a.chunk,
                          down_pca=pca).to(dev)          # fp32 on purpose
    rep = {"model": a.model, "layer": site, "d_model": d_model, "d_mem": a.d_mem,
           "chunk": a.chunk, "dtype": a.dtype, "n_decoder_layers": len(layers),
           "basis": a.basis, **branch_report(branch, host_params), "runs": []}
    print(f"branch: {rep['branch_trainable']:,} trainable params "
          f"= {rep['pct_of_host']:.4f} % of the host", flush=True)
    print("\nla note se lit en ECRITURES, pas en tokens : le backward garde un etat "
          "de memoire par ecriture\n", flush=True)

    if a.video:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(a.model)
        grid = [(int(f), int(s)) for f in a.frames.split(",")
                for s in a.write_stride.split(",")]
        print(f"entrees : vraie video {a.video.split('/')[-1]}", flush=True)
    else:
        grid = [(int(t), int(s)) for t in a.tokens.split(",")
                for s in a.write_stride.split(",")]
        print("entrees : identifiants de tokens aleatoires "
              "(longueur controlee, distribution irrealiste)", flush=True)

    for size, stride in grid:
        branch.write_stride = stride
        if a.video:
            inputs = video_inputs_for(a.model, a.video, size, a.max_side, dev, proc)
            n_tok = int(inputs["input_ids"].shape[1])
        else:
            inputs = text_inputs(model, size, dev)
            n_tok = size
        base_logits, base_peak, base_t = _forward_peak(model, inputs, dev)

        wrapper = graft(model, branch, layer=site)
        try:
            wrapper.reset()
            graft_logits, graft_peak, graft_t = _forward_peak(model, inputs, dev)
            same = bool(torch.equal(base_logits, graft_logits))
            diff = float((graft_logits - base_logits).abs().max())
            del base_logits, graft_logits
            torch.cuda.empty_cache()
            wrapper.reset()
            graft_bwd = ((None, None) if a.no_backward
                         else _backward_peak(model, inputs, dev))
        finally:
            ungraft(model, wrapper)
            torch.cuda.empty_cache()

        row = {
            "tokens": n_tok, "write_stride": stride,
            "writes": -(-n_tok // stride),
            "identical": same, "max_abs_diff": diff,
            # The inference floor: the frozen host alone.
            "fwd_peak_base_gb": base_peak / 2**30,
            "fwd_peak_graft_gb": graft_peak / 2**30,
            "fwd_extra_gb": (graft_peak - base_peak) / 2**30,
            "fwd_s_base": base_t, "fwd_s_graft": graft_t,
            # What training the branch costs: activations kept from the graft site
            # up, plus the branch's own graph. Nothing below layer `site` is stored.
            "bwd_peak_graft_gb": None if graft_bwd[0] is None else graft_bwd[0] / 2**30,
            "bwd_over_fwd_gb": (None if graft_bwd[0] is None
                                else (graft_bwd[0] - base_peak) / 2**30),
            "bwd_s_graft": graft_bwd[1],
        }
        rep["runs"].append(row)
        mark = "IDENTIQUE" if same else f"DIFFERENT (max {diff:.3e})"
        train = ("OOM" if row["bwd_peak_graft_gb"] is None else
                 f"{row['bwd_peak_graft_gb']:.2f} Go "
                 f"(+{row['bwd_over_fwd_gb']:.2f}) en {row['bwd_s_graft']:.1f}s")
        print(f"  {n_tok:>6} tok /{stride:>4} = {row['writes']:>5} ecritures : "
              f"{mark:<12} inference {row['fwd_peak_graft_gb']:.2f} Go "
              f"(+{row['fwd_extra_gb']:.3f}) | entrainement {train}", flush=True)

    ok = all(r["identical"] for r in rep["runs"])
    print(f"\nidentite a l'init : {'OK sur tous les points' if ok else 'ECHEC'}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print("json:", a.out)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
