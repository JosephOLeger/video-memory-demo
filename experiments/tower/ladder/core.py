"""Core of the memory-ladder harness: frames -> per-second visual tokens -> context.

Unit of time = one Qwen temporal token group = 2 frames at 2 fps = 1 s.
Every video segment is turned once into its post-merger visual tokens per group
([G, n_tok, 2048]); a memory is then just a list of groups, assembled without
re-running the vision tower. This is exact because the Qwen2.5-VL vision tower
attends within one temporal group only (cu_seqlens per grid_t; checked in smoke).

The LLM sees:  [text "memory" + video #1 = memory groups] [text "recent" + video #2
= window groups] [question]. Visual tokens are injected by replacing
``get_video_features``; input_ids / mm_token_type_ids / grid / second_per_grid_ts
are built here (second_per_grid_ts = 1.0 s: transformers 5.14 silently ignores
``fps=`` in the processor call and uses 1/12 s).
"""
from __future__ import annotations

import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

FPS = 2
HI, LO = 448, 224            # resolutions: 256 and 64 tokens per group
HIDDEN = 2048
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
FV = os.path.expanduser("~/work/titans-mac/data/finevideo")
LET = "ABCDEFGH"


# ------------------------------------------------------------------ host
class Host:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.tok = self.proc.tokenizer
        self.vpad = self.model.config.video_token_id
        self.lid = [self.tok.encode(c, add_special_tokens=False)[0] for c in LET]
        self._inject = None
        inner = self.model.model
        orig = inner.get_video_features

        def fake(pv, thw, **kw):
            if self._inject is None:
                return orig(pv, thw, **kw)
            return SimpleNamespace(pooler_output=tuple(self._inject))
        inner.get_video_features = fake

    # ---- vision
    @torch.no_grad()
    def encode(self, frames: np.ndarray, chunk: int = 32) -> torch.Tensor:
        """uint8 [T,H,W,3] (T even, H=W in {448,224}) -> bf16 [T/2, n_tok, 2048] on CPU."""
        assert frames.shape[0] % 2 == 0, frames.shape
        out = []
        for i in range(0, frames.shape[0], 2 * chunk):
            fr = frames[i:i + 2 * chunk]
            vp = self.proc.video_processor(videos=[fr], return_tensors="pt",
                                           do_sample_frames=False)
            pv = vp["pixel_values_videos"].to(self.device, torch.bfloat16)
            thw = vp["video_grid_thw"].to(self.device)
            emb = self.model.model.visual(pv, grid_thw=thw).pooler_output
            t, h, w = thw[0].tolist()
            out.append(emb.reshape(t, h * w // 4, -1).to("cpu"))
        return torch.cat(out)

    # ---- language
    def build(self, mem_side, n_mem, win_side, n_win, prompt):
        """input_ids etc. for [mem video][window video][prompt]; None counts mean absent."""
        content = []
        grids = []
        if n_mem:
            content += [{"type": "text", "text": "Memory of earlier parts of the video stream:"},
                        {"type": "video"}]
            grids.append((n_mem, mem_side // 14, mem_side // 14))
        if n_win:
            content += [{"type": "text", "text": "The most recent seconds of the video stream:"},
                        {"type": "video"}]
            grids.append((n_win, win_side // 14, win_side // 14))
        content.append({"type": "text", "text": prompt})
        text = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        parts = text.split("<|video_pad|>")
        assert len(parts) == len(grids) + 1
        ids = []
        for i, p in enumerate(parts):
            ids += self.tok.encode(p, add_special_tokens=False)
            if i < len(grids):
                t, h, w = grids[i]
                ids += [self.vpad] * (t * h * w // 4)
        ids = torch.tensor([ids])
        d = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "mm_token_type_ids": (ids == self.vpad).int() * 2}
        if grids:
            d["video_grid_thw"] = torch.tensor(grids)
            d["second_per_grid_ts"] = torch.ones(len(grids))
            d["pixel_values_videos"] = torch.zeros(1, 1176, dtype=torch.bfloat16)
        return d

    @torch.no_grad()
    def score(self, mem: torch.Tensor | None, win: torch.Tensor | None, prompt: str, n_opt: int):
        """mem/win: [G, n_tok, 2048] or None. Returns (letter logits [n_opt], n_ctx_tokens)."""
        side = lambda x: HI if x.shape[1] == 256 else LO          # noqa: E731
        d = self.build(side(mem) if mem is not None else None, 0 if mem is None else mem.shape[0],
                       side(win) if win is not None else None, 0 if win is None else win.shape[0],
                       prompt)
        inj = [x.reshape(-1, x.shape[-1]).to(self.device) for x in (mem, win) if x is not None]
        self._inject = inj
        try:
            d = {k: v.to(self.device) for k, v in d.items()}
            lg = self.model(**d, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            self._inject = None
        return lg[self.lid[:n_opt]].cpu(), int(d["input_ids"].shape[1])


# ------------------------------------------------------------------ filler
def fv_ids(n, seed=0, min_dur=200):
    rows = [json.loads(l) for l in open(os.path.join(FV, "manifest.jsonl"))]
    rows = [r for r in rows if r["duration"] >= min_dur]
    random.Random(seed).shuffle(rows)
    return [r["id"] for r in rows[:n]]


def square(frames: np.ndarray, side: int) -> np.ndarray:
    from PIL import Image
    T, H, W, _ = frames.shape
    s = min(H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    out = np.empty((T, side, side, 3), np.uint8)
    for i in range(T):
        out[i] = np.asarray(Image.fromarray(frames[i, y0:y0 + s, x0:x0 + s]).resize(
            (side, side), Image.BILINEAR))
    return out


def read_filler(vid: str, seconds: int, start_s: float = 10.0) -> np.ndarray:
    """``seconds`` s of FineVideo ``vid`` at 2 fps, 448 square crop. uint8 [2*seconds,448,448,3]."""
    import decord
    vr = decord.VideoReader(os.path.join(FV, "videos", vid + ".mp4"))
    src = float(vr.get_avg_fps())
    n = 2 * seconds
    idx = [min(len(vr) - 1, int(round((start_s + i / FPS) * src))) for i in range(n)]
    fr = vr.get_batch(idx).asnumpy()
    return square(fr, HI)


def lowres(frames: np.ndarray) -> np.ndarray:
    return square(frames, LO) if frames.shape[1] != LO else frames


class EmbCache:
    """Disk + RAM cache of per-group embeddings, keyed by a string."""

    def __init__(self, host: Host, root: str, ram: int = 64):
        self.host, self.root, self.ram, self.mem = host, root, ram, {}
        os.makedirs(root, exist_ok=True)

    def get(self, key: str, make_frames):
        if key in self.mem:
            return self.mem[key]
        path = os.path.join(self.root, key + ".pt")
        if os.path.exists(path):
            v = torch.load(path)
        else:
            fr = make_frames()
            hi = self.host.encode(fr)
            lo = self.host.encode(lowres(fr))
            v = {"hi": hi, "lo": lo}
            torch.save(v, path)
        if len(self.mem) >= self.ram:
            self.mem.pop(next(iter(self.mem)))
        self.mem[key] = v
        return v


def pad_even(frames: np.ndarray) -> np.ndarray:
    return frames if frames.shape[0] % 2 == 0 else np.concatenate([frames, frames[-1:]])


# ------------------------------------------------------------------ several labelled clips
def build_multi(host, segs, prompt):
    """segs: [(label_text, side, n_groups)] in order (the window is just the last seg)."""
    content, grids = [], []
    for lab, side, n in segs:
        content += [{"type": "text", "text": lab}, {"type": "video"}]
        grids.append((n, side // 14, side // 14))
    content.append({"type": "text", "text": prompt})
    text = host.proc.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    parts = text.split("<|video_pad|>")
    ids = []
    for i, p in enumerate(parts):
        ids += host.tok.encode(p, add_special_tokens=False)
        if i < len(grids):
            t, h, w = grids[i]
            ids += [host.vpad] * (t * h * w // 4)
    ids = torch.tensor([ids])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == host.vpad).int() * 2,
            "video_grid_thw": torch.tensor(grids), "second_per_grid_ts": torch.ones(len(grids)),
            "pixel_values_videos": torch.zeros(1, 1176, dtype=torch.bfloat16)}


@torch.no_grad()
def score_multi(host, segs, prompt, n_opt):
    """segs: [(label_text, tensor [G, n_tok, 2048])]."""
    side = lambda x: HI if x.shape[1] == 256 else LO          # noqa: E731
    d = build_multi(host, [(l, side(x), x.shape[0]) for l, x in segs], prompt)
    host._inject = [x.reshape(-1, x.shape[-1]).to(host.device) for _, x in segs]
    try:
        d = {k: v.to(host.device) for k, v in d.items()}
        lg = host.model(**d, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    finally:
        host._inject = None
    return lg[host.lid[:n_opt]].cpu(), int(d["input_ids"].shape[1])
