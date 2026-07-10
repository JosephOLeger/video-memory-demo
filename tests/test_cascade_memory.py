"""Tests for the opt-in metaplastic cascade momentum in NeuralMemory.

The cascade replaces the scalar momentum ``S_t`` by a K-reservoir state ``U_t``
(see ``titans/metaplastic.py``). It is strictly opt-in behind
``use_cascade_momentum``; with the flag off the memory must be bit-identical to
the original scalar-momentum path.
"""

import torch

from titans import NeuralMemory, TitansMACBlock


def make_memory(seed=0, **kwargs):
    torch.manual_seed(seed)
    mem = NeuralMemory(dim=8, hidden_dim=12, num_layers=2, theta_scale=0.5, **kwargs)
    mem.double()
    return mem


def test_flag_off_is_bit_identical():
    """Regression guard: default path unchanged by the cascade addition.

    Two memories built with identical seeds -- one constructed with the new
    (defaulted-off) kwargs spelled out -- must produce identical outputs.
    """
    torch.manual_seed(1)
    x = torch.randn(2, 16, 8, dtype=torch.float64)

    mem_a = make_memory(seed=42)
    mem_b = make_memory(seed=42, use_cascade_momentum=False)

    ya, sa = mem_a.forward_chunked(x, chunk_size=4)
    yb, sb = mem_b.forward_chunked(x, chunk_size=4)
    assert torch.equal(ya, yb)
    for key in sa:
        assert torch.equal(sa[key], sb[key])


def test_k1_cascade_matches_scalar():
    """A single-reservoir (K=1) cascade must reproduce the scalar momentum."""
    torch.manual_seed(7)
    x = torch.randn(2, 12, 8, dtype=torch.float64)

    scalar = make_memory(seed=99)
    casc = make_memory(seed=99, use_cascade_momentum=True, cascade_reservoirs=1)

    y_s, _ = scalar.forward_chunked(x, chunk_size=4)
    y_c, _ = casc.forward_chunked(x, chunk_size=4)
    assert torch.allclose(y_s, y_c, rtol=1e-10, atol=1e-10)


def test_cascade_runs_no_nan_both_paths():
    """With the flag on, both update() and the naive loop run cleanly."""
    for selective in (False, True):
        torch.manual_seed(3)
        mem = make_memory(
            seed=3, use_cascade_momentum=True, cascade_reservoirs=4,
            cascade_selective=selective,
        )
        x = torch.randn(2, 12, 8, dtype=torch.float64)

        # naive sequential path (ground truth)
        y, state = mem.forward_naive_sequential(x, chunk_size=4)
        assert y.shape == (2, 12, 8)
        assert torch.isfinite(y).all()

        # chunked update() path (used by MAC)
        st, S = mem.init_memory(2)
        st, S = mem.update(x, st, S, chunk_size=4)
        for key in S:
            # cascade carry has a trailing reservoir axis of size K=4
            assert S[key].shape[-1] == 4
            assert torch.isfinite(S[key]).all()
        for key in st:
            assert torch.isfinite(st[key]).all()


def test_cascade_chunked_vs_naive_agree():
    """chunked vs naive-sequential still agree when the cascade is enabled.

    Mirrors test_memory_equivalence.test_naive_chunked_equivalence. Selective
    mode is included because its per-token betas are shared by both paths.
    """
    for selective in (False, True):
        torch.manual_seed(5)
        mem = make_memory(
            seed=5, use_cascade_momentum=True, cascade_reservoirs=4,
            cascade_selective=selective,
        )
        x = torch.randn(2, 16, 8, dtype=torch.float64)
        for chunk_size in (4, 8, 16):
            y_ref, s_ref = mem.forward_naive_sequential(x, chunk_size=chunk_size)
            y_chk, s_chk = mem.forward_chunked(x, chunk_size=chunk_size)
            assert torch.allclose(y_ref, y_chk, rtol=1e-8, atol=1e-8), (
                f"selective={selective} chunk={chunk_size} "
                f"max|diff|={(y_ref - y_chk).abs().max().item():.3e}"
            )
            for key in s_ref:
                assert torch.allclose(s_ref[key], s_chk[key], rtol=1e-8, atol=1e-8)


def test_mac_block_with_cascade_smoke():
    """TitansMACBlock runs forward with the cascade enabled on a toy input."""
    for selective in (False, True):
        torch.manual_seed(11)
        block = TitansMACBlock(
            dim=8, num_heads=2, segment_size=4, num_persistent_tokens=2,
            memory_use_cascade=True, cascade_reservoirs=3,
            cascade_selective=selective,
        )
        x = torch.randn(2, 8, 8)
        out, state, S = block(x)
        assert out.shape == (2, 8, 8)
        assert torch.isfinite(out).all()
        for key in S:
            assert S[key].shape[-1] == 3
