"""LongVideoBench: the judge. The graft against the untouched host, at matched cost.

The comparison this makes, and the one it refuses to make
---------------------------------------------------------
The standing rule for this project is that the graft only ever **adds**: it never
replaces an attention block and never shortens the context. So the honest
comparison is not "memory instead of frames" but **the same frames, plus memory**:

    baseline : the frozen host sees ``--frames`` frames sampled uniformly
    graft    : the frozen host sees THE SAME ``--frames`` frames, and the branch
               additionally carries a memory written from a ``--mem-frames``
               sampling of the whole video

Both pay the same attention cost on the same context; one of them has also read
the parts that uniform sampling threw away. Any gain is attributable to the memory
and to nothing else, and the zero-init ``up`` makes the baseline *exact* rather
than a re-run of a similar model.

Why LongVideoBench and not EgoLifeQA: EgoLifeQA is the training signal now, so it
cannot also be the judge. LongVideoBench stratifies its questions by video
duration (15 / 60 / 600 / 3600 s) and its leaderboard publishes accuracy per
stratum, so the reach curve is directly comparable to GPT-4o, Gemini-1.5-Pro and
LLaVA-Video without our building anything.

The answer is scored by comparing the logit of each option letter at the first
generated position -- not by free generation, which would make the score depend on
decoding and on the model's willingness to follow a format.

    python -m titans.graft.eval_lvb --data data/longvideobench \
        --ckpt runs/graft_unit_s8.pt --basis cache/qwen_l24_pca128.pt \
        --frames 32 --mem-frames 256 --limit 200 --out results/lvb_unit.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .qwen import default_layer, disabled, find_decoder_layers, graft

LETTERS = "ABCDEFGH"


def build_prompt(item: dict) -> str:
    """The question and its options, in LongVideoBench's own order."""
    lines = [item["question"]]
    for i, c in enumerate(item["candidates"]):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


@torch.no_grad()
def letter_logits(model, processor, frames, prompt: str, n_opt: int, fps: float,
                  device: str) -> torch.Tensor:
    """Logit of each option letter at the first answer position.

    Scoring by letter logit rather than by generation keeps the metric independent
    of decoding: a model that would have rambled before answering is scored on what
    it believed, not on whether it obeyed the format.
    """
    from .fit_basis import video_inputs
    inp = video_inputs(processor, frames, prompt, fps)
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    logits = model(**inp, use_cache=False).logits[0, -1].float()
    ids = [processor.tokenizer.encode(LETTERS[i], add_special_tokens=False)[0]
           for i in range(n_opt)]
    return logits[torch.tensor(ids, device=logits.device)]


@torch.no_grad()
def write_memory(branch: MemoryBranch, model, processor, frames, prompt: str,
                 fps: float, device: str, layer: int, chunk: int, stride: int):
    """Run the *ungrafted* host over the dense sampling and write what it sees.

    The host must be ungrafted here: the branch reads layer L's output, which does
    not depend on the branch, so this is the same activation the graft would see --
    and running it grafted would advance the recurrence twice.
    """
    layers = find_decoder_layers(model)
    got = []

    def hook(_m, _i, out):
        got.append((out[0] if isinstance(out, tuple) else out).detach())

    h = layers[layer].register_forward_hook(hook)
    try:
        from .fit_basis import video_inputs
        inp = video_inputs(processor, frames, prompt, fps)
        inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
        model(**inp, use_cache=False)
    finally:
        h.remove()
    z = branch.down(got[0].float())[:, ::stride]              # [1, n_w, d_mem]
    state, S = branch.init_carry(1)
    if z.shape[1]:
        state, S = branch.memory.update(z, state, S, min(chunk, z.shape[1]))
    return state, S, int(z.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True, help="racine contenant lvb_val.json et videos/")
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--ckpt", default=None, help="runs/*.pt ; sans lui, seule la "
                                                 "reference est evaluee")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--frames", type=int, default=32,
                    help="images DANS LE CONTEXTE, identiques pour les deux modeles")
    ap.add_argument("--mem-frames", type=int, default=256, dest="mem_frames",
                    help="images ecrites dans la memoire (jamais dans le contexte)")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=None, dest="write_stride",
                    help="par defaut, celui de l'entrainement (lu dans le "
                         "checkpoint) : la portee de la memoire est calibree sur "
                         "un nombre d'ecritures, pas sur une duree de video")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--duration-group", type=int, default=None, dest="duration_group",
                    help="n'evaluer qu'une tranche (15 / 60 / 600 / 3600 s)")
    ap.add_argument("--guard-gb", type=float, default=9.0, dest="guard_gb")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from ..worldmodel.data import load_video_frames
    from .check_qwen import load_host
    from .fit_basis import resize_frames

    if a.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(a.device)
        print(f"{free / 2**30:.1f} Go libres sur {total / 2**30:.1f}", flush=True)
        if free / 2**30 < a.guard_gb:
            raise SystemExit(f"abandon : il faut {a.guard_gb} Go libres")

    data = json.load(open(os.path.join(a.data, a.split)))
    if a.duration_group:
        data = [d for d in data if d.get("duration_group") == a.duration_group]
    if a.limit:
        data = data[:a.limit]
    print(f"{len(data)} questions"
          + (f" (tranche {a.duration_group} s)" if a.duration_group else ""), flush=True)

    model = load_host(a.model, getattr(torch, a.dtype), a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model))) if a.layer is None else a.layer

    branch = wrapper = None
    if a.ckpt:
        blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        args = blob.get("args", {})
        pca = (torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
               if a.basis else None)
        d_model = int(model.config.text_config.hidden_size)
        mem_kw = memory_kwargs_from_args(args)
        if a.write_stride is None:
            a.write_stride = int(args.get("write_stride", 2))
        branch = MemoryBranch(d_model, args.get("d_mem", 128), chunk_size=a.chunk,
                              down_pca=pca, **mem_kw).to(a.device)
        branch.load_state_dict(blob["branch"])
        branch.eval()
        wrapper = graft(model, branch, layer=site)
        wrapper.writes_enabled = False        # la memoire est ecrite hors contexte
        print(f"greffe {a.ckpt} (pas {blob.get('step')}, val {blob.get('val_mem'):.4f}) "
              f"muon={mem_kw['orthogonalize_momentum']} "
              f"alpha_init={mem_kw['alpha_init']:.3g} pas-ecriture={a.write_stride} "
              f"(entrainement : {args.get('write_stride')})", flush=True)

    rows, t0 = [], time.time()
    for i, item in enumerate(data):
        path = os.path.join(a.data, "videos", item["video_path"])
        if not os.path.exists(path):
            continue
        n_opt = len(item["candidates"])
        prompt = build_prompt(item)
        dur = float(item["duration"])
        try:
            ctx = resize_frames(load_video_frames(path, num_frames=a.frames), a.max_side)
            fps_ctx = a.frames / dur
            with (disabled(wrapper) if wrapper else _nullcm()):
                base = letter_logits(model, processor, ctx, prompt, n_opt,
                                     fps_ctx, a.device)
            row = {"id": item["id"], "duration_group": item.get("duration_group"),
                   "duration": dur, "gold": item["correct_choice"],
                   "base": int(base.argmax())}
            if wrapper is not None:
                dense = resize_frames(load_video_frames(path, num_frames=a.mem_frames),
                                      a.max_side)
                with disabled(wrapper):
                    state, S, nw = write_memory(branch, model, processor, dense,
                                                "", a.mem_frames / dur, a.device,
                                                site, a.chunk, a.write_stride)
                wrapper.carry, wrapper.token_offset = (state, S), 0
                mem = letter_logits(model, processor, ctx, prompt, n_opt,
                                    fps_ctx, a.device)
                row["mem"] = int(mem.argmax())
                row["writes"] = nw
            rows.append(row)
        except Exception as e:
            print(f"  saut {item['id']}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            acc_b = sum(r["base"] == r["gold"] for r in rows) / max(1, len(rows))
            msg = f"  [{i + 1}/{len(data)}] reference {100 * acc_b:.1f} %"
            if wrapper is not None and rows and "mem" in rows[0]:
                acc_m = sum(r["mem"] == r["gold"] for r in rows) / len(rows)
                msg += f" | greffe {100 * acc_m:.1f} %"
            print(msg + f"  ({(time.time() - t0) / (i + 1):.1f} s/question)", flush=True)

    report = summarise(rows)
    print("\n" + format_report(report))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "report": report, "rows": rows}, f, indent=1)
        print("json:", a.out)


class _nullcm:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def summarise(rows: list) -> dict:
    """Accuracy overall and per duration stratum -- the axis that is the claim."""
    out = {}
    groups = sorted({r.get("duration_group") for r in rows if r.get("duration_group")})
    for key, sel in [("all", rows)] + [(str(g), [r for r in rows
                                                 if r.get("duration_group") == g])
                                       for g in groups]:
        if not sel:
            continue
        e = {"n": len(sel),
             "base": sum(r["base"] == r["gold"] for r in sel) / len(sel)}
        if "mem" in sel[0]:
            e["mem"] = sum(r["mem"] == r["gold"] for r in sel) / len(sel)
            e["delta"] = e["mem"] - e["base"]
            # The memory's reach is calibrated on a number of WRITES, not on a
            # duration. A stratum evaluated far from the training write count is
            # being asked something it was never tuned for, and the accuracy alone
            # does not show it -- so the count travels with the score.
            w = [r["writes"] for r in sel if "writes" in r]
            if w:
                e["writes"] = sum(w) / len(w)
        out[key] = e
    return out


def format_report(rep: dict) -> str:
    has_mem = any("mem" in v for v in rep.values())
    has_w = any("writes" in v for v in rep.values())
    head = (f"{'tranche':>9}{'n':>6}{'reference':>11}"
            + (f"{'greffe':>9}{'delta':>8}" if has_mem else "")
            + (f"{'ecritures':>11}" if has_w else ""))
    lines = [head]
    for k, v in rep.items():
        line = f"{k:>9}{v['n']:>6}{100 * v['base']:>10.1f}%"
        if "mem" in v:
            line += f"{100 * v['mem']:>8.1f}%{100 * v['delta']:>+7.1f}"
        if has_w:
            line += f"{v.get('writes', 0):>11.0f}"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
