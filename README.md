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

## Phase 2: MAC (Memory As Context)

`src/titans/mac.py` adds `TitansMACBlock`, the **Memory As Context** architecture
(§4.1, Eqs 19-25). A sequence `[B, N, d]` is split into fixed segments of size
`C`; the neural-memory carry `(state, S)` (weights **and** momentum) is threaded
across segments *sequentially*, while parallelism lives *inside* each segment.
Per segment `i`:

1. **Historical retrieval** from the memory carried from earlier segments:
   `h_i = retrieve(state, q_proj(S_i))`.
2. Build the context window `[persistent tokens; h_i; S_i]` (`persistent.py`:
   learnable `PersistentMemory` tokens).
3. **Causal self-attention** over that window (`attention.py`,
   `CausalSelfAttention`), with the §4.4 details: SiLU on Q/K/V projections,
   L2-normalised (cosine) Q·K, and a depthwise-separable **causal** conv after
   each projection.
4. Take the last `C` positions as the segment's attention output `y_i`.
5. **Update the long-term memory using `y_i`** via the new retrieval-free
   `NeuralMemory.update(y_i, state, S, chunk_size)` — the memory learns from what
   attention selected.
6. **Output** = gate of `y_i` with a fresh retrieval `r_i` from the just-updated
   memory, then a final projection; a residual wraps the whole block.

Supporting change in `memory.py`: a shared chunked-scan helper now backs both the
Phase-1 `forward_chunked` and the new `update` (which skips retrieval), and the
momentum state `S` is exposed via `init_memory` / `update` so it can be carried
across segments. The Phase-1 tests are unchanged and still pass.

```bash
pytest        # runs Phase 1 (memory equivalence) + Phase 2 (MAC) tests
```

`tests/test_mac.py` checks output shape (several configs), **causality** (a change
at/after a segment boundary leaves every earlier output bit-identical — the key
correctness test), gradient flow to every outer parameter (the functional inner
loop does not detach the outer graph), and a small toy trainability check.

### Under-specified points (documented design choices)

* **`N % C == 0` is required** — no padding logic this phase; a clear `ValueError`
  is raised otherwise.
* **Update/output gate**: the paper only says "normalization and gating with a
  linear layer". We RMSNorm each branch and combine with a learnable per-channel
  sigmoid gate `o = norm(y)·σ(W_r norm(r)) + norm(r)·σ(W_y norm(y))`, then a final
  linear output projection with a block-level residual `x + out_proj(o)`.
* **Causal-conv kernel size** defaults to 4 ("a small kernel" in the paper).
* **Memory update chunk size** defaults to the segment size `C` (one parallel
  chunk per segment).
* **Not validated against an official reference** — none is public. Correctness
  here means *internally consistent, causal, and faithful to the paper's
  described mechanism*, not numerically matching a ground truth.

## Status

**Phase 1** (neural memory core), **Phase 2** (a single, well-tested MAC block)
and **Phase 3** (a small world model) are done.

Phase 3 lives in `src/titans/worldmodel/`: a `WorldModel` that stacks MAC blocks
to predict the next latent `Z_{t+1}` from past latents `Z_1..t`, trained with an
L1 loss. The core runs on plain latent tensors with no extra dependencies; an
optional lazily-loaded `VJEPA2Encoder` bridges real video clips to those latents.
A self-contained demo (synthetic latents, no downloads) confirms trainability:

```bash
python -m titans.worldmodel.train   # prints L1 dropping ~0.53 -> 0.03
```

The **real V-JEPA-2 chain** is wired and verified end to end (on synthetic video,
no dataset yet). Latents are precomputed once with the frozen encoder and cached,
then the world model trains on the cache:

* `frames.py` — temporal aggregation: average consecutive frames over a sliding
  `window` moved by a `stride` (a stride > 1 lets a fixed clip span a longer
  real-time window → longer effective videos), then chunk into fixed-size clips.
* `encoder.py` — frozen `facebook/vjepa2-vitl-fpc64-256` (the `fpc16` repos are
  SSv2 classifiers, not the base encoder). A clip `[B, 64, 3, 256, 256]` → `[B,
  32, 1024]`: 32 temporal groups (derived from the token grid, not hard-coded),
  each a spatial mean-pool of its 256 patches.
* `precompute_latents.py` — frames → aggregate → clips → processor → encoder →
  concatenate clip latents → cache `.pt`.
* `data.py` — **Kinetics-mini** loader: lists the `nateraw/kinetics-mini` `.mp4`
  files (train/val, 5 classes × 10 clips), downloads them, and decodes frames
  with `decord` into `[F, H, W, C]` tensors. `precompute_kinetics(...)` ties this
  to the encoder in one call.
* `train.py` — `load_latents` + per-channel `standardize_latents` (raw encoder
  features are large-magnitude and drive the memory's inner loop to `NaN`
  otherwise). On cached real Kinetics latents the L1 drops cleanly with no
  divergence.

The encoder runs on **CUDA** automatically when available (~17× faster than CPU:
~2.9 s vs ~49 s per 64-frame clip on an RTX 5070); latents are moved to CPU
before caching so the cache is portable.

**Memory stability fix** (`memory.py`): `mlp_forward` now RMS-normalises the
memory MLP output. Threading the online memory across many segments otherwise
lets `M`'s weights — and thus `M(q)` and the associative loss driving the update
— grow ~geometrically per segment and overflow to `NaN` (only visible once a
sequence spans ~8+ segments, i.e. the long-context regime this project targets).
Normalising the memory *function* bounds both retrieval and loss. Guarded by
`test_many_segments_stay_finite`; the Phase-1 equivalence tests still hold since
both code paths share `mlp_forward`.

Not yet built: longer-context clips concatenated across time to really showcase
Titans' long-memory advantage (Phase 4), the MAG and MAL variants, per-channel
(vs scalar) memory gates, and a true parallel scan for the momentum recurrence
(currently a Python scan across chunks).
