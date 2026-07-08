"""Test the sliding-window continuous encoding tiles frames contiguously.

Uses a fake processor/encoder (no V-JEPA download): frames carry their own index
as pixel value, and the fake encoder returns the per-tubelet mean, so each latent
is a known "signature" of the frames it came from. A correct continuous stream
must have signatures that increase by exactly one tubelet per step with no gap or
overlap -- i.e. arange(0.5, F, tubelet).
"""

import numpy as np
import torch

from titans.worldmodel.precompute_latents import video_to_latents_sliding

TUBELET = 2


class _FakeProcessor:
    def __call__(self, arr, return_tensors=None):
        # arr: numpy [T, H, W, C] uint8 -> pixel_values_videos [1, T, C, H, W]
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float()
        return {"pixel_values_videos": t}


class _FakeEncoder:
    """Returns [1, T', d]; each latent = mean pixel over its tubelet + space."""
    model_name = "fake"

    def __call__(self, pv):  # pv: [1, T, C, H, W]
        _, T, C, H, W = pv.shape
        Tp = T // TUBELET
        m = pv.view(1, Tp, TUBELET, C, H, W).mean(dim=(2, 3, 4, 5))  # [1, T']
        return m.unsqueeze(-1).expand(1, Tp, 4)  # [1, T', d=4]


def test_sliding_tiles_frames_contiguously():
    F, clip_len, stride = 24, 8, 4
    # frame f has constant pixel value f (< 255), so a latent over frames
    # [a, a+1] has signature a + 0.5.
    frames = torch.arange(F).view(F, 1, 1, 1).expand(F, 2, 2, 3).contiguous()

    z = video_to_latents_sliding(
        frames, _FakeEncoder(), _FakeProcessor(),
        clip_len=clip_len, stride_frames=stride,
    )
    sig = z[:, 0]  # signature per latent

    expected = torch.arange(0.5, F, TUBELET)  # 0.5, 2.5, ..., 22.5
    assert z.shape[0] == expected.shape[0], f"length {z.shape[0]} != {expected.shape[0]}"
    assert torch.allclose(sig, expected, atol=1e-5), f"non-contiguous tiling: {sig.tolist()}"
    # strictly increasing => no gaps, no overlaps, no discontinuity
    assert (sig[1:] > sig[:-1]).all()


def test_sliding_full_stride_matches_single_clip_span():
    # stride == clip_len => no overlap; still runs and covers the video.
    F, clip_len = 16, 8
    frames = torch.arange(F).view(F, 1, 1, 1).expand(F, 2, 2, 3).contiguous()
    z = video_to_latents_sliding(frames, _FakeEncoder(), _FakeProcessor(),
                                 clip_len=clip_len, stride_frames=clip_len)
    assert z.shape[0] == F // TUBELET
