"""Cropping a latent cache to shorter sequences.

The point of cropping is to fit ``[V;V]`` under full BPTT, so the two things that
must not break are the *content* (crop i really is the i-th slice, in order) and the
*metadata* (a cropped PCA cache must still look pre-reduced to the harness, or every
run silently re-standardises and the numbers stop being comparable).
"""

import pytest
import torch

from titans.worldmodel.crop_cache import crop_cache_file, crop_latents
from titans.worldmodel.train import load_latent_cache


def test_shape_and_ragged_tail():
    z = torch.randn(5, 144, 16)
    c = crop_latents(z, 64)
    assert c.shape == (10, 64, 16)          # 2 crops of 64, the last 16 dropped


def test_crops_are_the_slices_in_order():
    z = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    c = crop_latents(z, 2)
    assert c.shape == (6, 2, 3)
    # sequence 0 -> crops 0,1,2 ; sequence 1 -> crops 3,4,5
    for seq in range(2):
        for k in range(3):
            assert torch.equal(c[seq * 3 + k], z[seq, k * 2:(k + 1) * 2])


def test_T_larger_than_the_cache_is_an_error():
    with pytest.raises(ValueError, match="exceeds"):
        crop_latents(torch.randn(2, 8, 4), 16)


def test_metadata_survives_and_the_cache_stays_pre_reduced(tmp_path):
    pca = {"mean": torch.zeros(8), "std": torch.ones(8),
           "components": torch.eye(8)[:4], "sv": torch.ones(4),
           "whiten": True, "k": 4, "explained": 0.85}
    src = tmp_path / "src.pt"
    torch.save({"latents": torch.randn(3, 12, 4), "pca": pca, "k": 4,
                "whiten": True, "explained": 0.85,
                "segments": ["A1/d/a.mp4", "A1/d/b.mp4", "A2/d/c.mp4"]}, src)

    dst = tmp_path / "dst.pt"
    info = crop_cache_file(src, dst, 4)
    assert info["shape"] == (9, 4, 4) and info["crops_per_seq"] == 3

    z, meta = load_latent_cache(dst)
    assert z.shape == (9, 4, 4)
    assert "pca" in meta and meta["k"] == 4        # what the harness keys on
    assert len(meta["segments"]) == 9              # provenance stays row-aligned
    assert meta["segments"][:3] == ["A1/d/a.mp4"] * 3
    assert meta["crop"] == {"from_T": 12, "to_T": 4, "crops_per_seq": 3,
                            "source": str(src)}


def test_write_is_atomic(tmp_path):
    """No .tmp left behind, and the destination only appears complete."""
    src = tmp_path / "src.pt"
    torch.save({"latents": torch.randn(2, 8, 4)}, src)
    dst = tmp_path / "out" / "dst.pt"
    crop_cache_file(src, dst, 4)
    assert dst.exists()
    assert list(dst.parent.glob("*.tmp")) == []
