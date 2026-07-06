# titans-mac

A from-scratch PyTorch reimplementation of the **neural long-term memory** from
*Titans: Learning to Memorize at Test Time* (Behrouz, Zhong, Mirrokni — Google
Research, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)). The memory is a
small deep MLP whose *weights* are updated online at test time by an inner
meta-learning loop: each token produces an associative-memory loss
`‖M(k_t) − v_t‖²`, and the memory's weights follow a data-dependent
momentum ("surprise") + weight-decay ("forget") update
(`S_t = η_t S_{t-1} − θ_t ∇_W ℓ`, `M_t = (1−α_t) M_{t-1} + S_t`). `α_t, η_t, θ_t`
are scalars per token produced by small learned linear heads on the input.

`src/titans/memory.py` exposes a `NeuralMemory` module with two equivalent code
paths: `forward_naive_sequential` (an explicit for-loop reference using
`torch.func.grad` per step) and `forward_chunked` (the §3.2 parallel
reformulation — per-chunk batched gradients via nested `vmap`, then a scanned
momentum recurrence). Retrieval `y_t = M_t(q_t)` is exposed via `retrieve`.

## Running the tests

```bash
pip install -e '.[test]'   # or: pip install torch pytest
pytest
```

The tests check that the naive and chunked paths agree numerically (float64,
`atol/rtol` down to `1e-9`) across several chunk sizes, and a toy sanity check
that online updates actually memorize repeated key/value pairs.

## Status

This is **Phase 1** of a multi-phase implementation: the neural memory core
only. The MAC / MAG / MAL attention architecture, sequence segmentation, and
persistent memory tokens are future work (Phase 2).
