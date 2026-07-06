"""Tests for the Titans neural-memory core (Phase 1)."""

import torch

from titans import NeuralMemory


def make_memory(dim=8, hidden_dim=12, num_layers=2, theta_scale=0.5, seed=0):
    torch.manual_seed(seed)
    mem = NeuralMemory(
        dim=dim, hidden_dim=hidden_dim, num_layers=num_layers, theta_scale=theta_scale
    )
    mem.double()  # tighten tolerances by running in float64
    return mem


def test_naive_chunked_equivalence():
    """forward_chunked must reproduce the reference loop for every chunk size.

    Both paths use chunk-start gradient semantics for a given chunk size, so the
    outputs are equal up to floating-point roundoff (the chunked path merely
    computes the same gradients in parallel and scans the same recurrence).
    """
    torch.manual_seed(1)
    dim, N, B = 8, 16, 2
    mem = make_memory(dim=dim, num_layers=2)
    x = torch.randn(B, N, dim, dtype=torch.float64)

    for chunk_size in (4, 8, N):
        y_ref, s_ref = mem.forward_naive_sequential(x, chunk_size=chunk_size)
        y_chk, s_chk = mem.forward_chunked(x, chunk_size=chunk_size)
        assert y_ref.shape == y_chk.shape == (B, N, dim)
        assert torch.allclose(y_ref, y_chk, rtol=1e-9, atol=1e-9), (
            f"outputs diverge at chunk_size={chunk_size}: "
            f"max |diff| = {(y_ref - y_chk).abs().max().item():.3e}"
        )
        for key in s_ref:
            assert torch.allclose(s_ref[key], s_chk[key], rtol=1e-9, atol=1e-9)


def test_chunk1_equals_fully_online():
    """chunk_size == 1 is the fully-online recurrence (default naive path)."""
    torch.manual_seed(2)
    dim, N, B = 10, 20, 3
    mem = make_memory(dim=dim, num_layers=3)
    x = torch.randn(B, N, dim, dtype=torch.float64)

    y_online, _ = mem.forward_naive_sequential(x)  # chunk_size=None
    y_chunk1, _ = mem.forward_chunked(x, chunk_size=1)
    assert torch.allclose(y_online, y_chunk1, rtol=1e-9, atol=1e-9)


def test_deeper_mlp_equivalence():
    """Same equivalence with a deeper (L_M=3) memory MLP."""
    torch.manual_seed(3)
    dim, N, B = 12, 24, 2
    mem = make_memory(dim=dim, hidden_dim=16, num_layers=3)
    x = torch.randn(B, N, dim, dtype=torch.float64)

    for chunk_size in (4, 8, N):
        y_ref, _ = mem.forward_naive_sequential(x, chunk_size=chunk_size)
        y_chk, _ = mem.forward_chunked(x, chunk_size=chunk_size)
        assert torch.allclose(y_ref, y_chk, rtol=1e-8, atol=1e-8)


def test_arbitrary_leading_dims():
    """A single unbatched sequence [N, d] is accepted and shaped correctly."""
    torch.manual_seed(4)
    dim, N = 8, 12
    mem = make_memory(dim=dim)
    x = torch.randn(N, dim, dtype=torch.float64)
    y, _ = mem.forward_chunked(x, chunk_size=4)
    assert y.shape == (N, dim)


def test_memorization_reduces_error():
    """Sanity: online updates should memorize repeated key/value associations.

    We feed a small set of distinct tokens repeated many times. Retrieval error
    at the final occurrence of each token (memory already updated on it) should
    be far lower than the no-update baseline (initial memory M_0).
    """
    torch.manual_seed(5)
    dim = 8
    mem = make_memory(dim=dim, hidden_dim=16, num_layers=2, theta_scale=1.0)

    # Pin the data-dependent gates to a stable regime so the sanity check is
    # about memorization, not about an untrained network's arbitrary gates:
    # no forgetting (alpha~0), no momentum (eta~0), small inner lr (theta~0.3).
    # This reduces the update to plain online gradient descent.
    import math

    with torch.no_grad():
        for head, bias in (
            (mem.alpha_head, -20.0),  # sigmoid -> ~0
            (mem.eta_head, -20.0),    # sigmoid -> ~0
        ):
            head.weight.zero_()
            head.bias.fill_(bias)
        mem.theta_head.weight.zero_()
        # softplus(b) = 0.05  =>  b = log(exp(0.05) - 1)
        mem.theta_head.bias.fill_(math.log(math.expm1(0.05)))
        # Tie the query projection to the key projection so retrieval probes the
        # learned key -> value association (otherwise we would query the memory
        # at a projection it was never trained on).
        mem.q_proj.weight.copy_(mem.k_proj.weight)

    n_distinct, repeats = 4, 12
    base = torch.randn(n_distinct, dim, dtype=torch.float64)
    seq = base.repeat(repeats, 1).unsqueeze(0)  # [1, n_distinct*repeats, dim]

    # Targets the memory is trying to learn: v_t = x_t W_V for the SAME tokens.
    with torch.no_grad():
        v = mem.v_proj(seq)  # [1, T, dim]
        q = mem.q_proj(seq)

    y, final_state = mem.forward_chunked(seq, chunk_size=8)

    # Updated-memory retrieval error on the last block (memory has seen each
    # token 11 times already).
    last_block = slice(-n_distinct, None)
    updated_err = ((y[:, last_block] - v[:, last_block]) ** 2).mean().item()

    # No-update baseline: retrieve the same queries from the initial memory M_0.
    q_last = q[:, last_block].reshape(-1, dim)  # [n_distinct, dim]
    v_last = v[:, last_block].reshape(-1, dim)
    m0 = mem.memory.init_state(q_last.shape[0])
    baseline = mem.retrieve(m0, q_last)
    baseline_err = ((baseline - v_last) ** 2).mean().item()

    assert updated_err < 0.5 * baseline_err, (
        f"memorization did not reduce error: updated={updated_err:.4e} "
        f"baseline={baseline_err:.4e}"
    )
