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

**Phase 1** (neural memory core), **Phase 2** (a single, well-tested MAC block),
**Phase 3** (a small world model over frozen V-JEPA-2 video latents) and **Phase
4** (long-term-memory ablations, a GPU/multi-seed experiment harness, a longer and
more diverse EgoLife data source) are done, alongside two smaller additions: the
**ATLAS Omega rule** (a sliding-window generalisation of the memory update) and
**membench** (a synthetic long-range-memory benchmark, independent of the video
world model).

Phase 3 lives in `src/titans/worldmodel/`: a `WorldModel` that stacks MAC blocks
to predict the next latent `Z_{t+1}` from past latents `Z_1..t`, trained with an
L1 loss. The core runs on plain latent tensors with no extra dependencies; an
optional lazily-loaded `VJEPA2Encoder` bridges real video clips to those latents.
A self-contained demo (synthetic latents, no downloads) confirms trainability:

```bash
python -m titans.worldmodel.train   # prints L1 dropping ~0.53 -> 0.03
```

The **real V-JEPA-2 chain** is wired and verified end to end, including a real run
against actual Kinetics-mini clips (a cached latent tensor from that run,
`cache/kinetics.pt`, sits on disk locally — `.pt` files are gitignored, so it is
not committed and no loss log from that run survives in the repo). Latents are
precomputed once with the frozen encoder and cached, then the world model trains
on the cache:

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

## Phase 4: long-term-memory ablations + experiment harness

`src/titans/worldmodel/ablation.py` asks what the *threaded* long-term memory
actually contributes, controlling for two confounds:

* **Parameter count**: the control is not a smaller model with the memory branch
  removed, but the same model with `reset_memory_each_segment=True`
  (`TitansMACBlock` / `WorldModel`) — identical weights, but the memory is reset
  at every segment so it carries no information *across* segments. Since
  attention only ever sees one segment window, "threaded" vs "reset" isolates the
  long-range pathway at zero parameter-count cost.
* **A single scalar loss hides *where* memory helps**: `per_position_l1` reports
  next-step L1 as a function of temporal position, bucketed by segment
  (`format_report`), so a real long-range effect should show the threaded-vs-reset
  gap *growing* with position, not sitting flat.

Two experiment functions, both using train/val splits from the dataset's separate
folders (no video-level leakage) with standardisation stats fit on train only:

* `run_ablation` trains **threaded** and **reset-control** models *separately*
  (same seed → identical init) and compares — conflates "memory helps at
  position t" with "training with memory yields different weights".
* `run_inference_ablation` trains **one** threaded model, then evaluates *the same
  trained weights* twice — once with memory threaded, once reset each segment at
  inference only. Segment 0 is identical by construction, so a growing gap is
  purely the accumulated memory paying off at inference, with no training
  confound.

`src/titans/worldmodel/harness.py` makes comparing variants fast and statistically
honest: `train_eval_once` trains one `WorldModel` config (on GPU when available,
optionally mini-batched) and reports train/val L1; `run_config` repeats that over
several seeds and reports `mean ± std`, plus the **persistence baseline**
(`z_hat[t+1] = z[t]`, from `persistence_baseline_l1`) so a number means something
— does the model even beat copying the last latent; `compare_variants` /
`format_comparison` run several named configs and tabulate them together, with
the train/val gap flagging overfitting (data-limited) vs underfitting.
`tests/test_harness.py` checks the baseline formula and that a tiny run completes
and stays finite.

## ATLAS Omega rule (arXiv:2505.23735)

`NeuralMemory(window_size=c)` (`memory.py`) generalises the memory's inner-loop
objective from ATLAS, *"Atlas: Learning to Optimally Memorize the Context at Test
Time"* (Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, Mirrokni — same
Google Research team as Titans). Titans' *Delta rule* (`c=1`, the default) fits the
memory to the **current token only**: `min_M ‖M(k_t) − v_t‖²`. ATLAS's **Omega
rule** (`c>1`) instead fits a **gated sliding-window sum** over the last `c`
tokens (Eq. 9):

```
min_M  sum_{i=t-c+1}^{t}  gamma_i^(t) · ‖M(k_i) − v_i‖²
```

so the memory learns a local *context* rather than isolated tokens; per-token
gates `gamma_i` (a `gamma_head`, created only when `c>1`) let the model prune which
past tokens in the window count. Because the gradient of a sum is the sum of
gradients, this reuses the same per-token gradients `_scan_chunked` already
computes — no extra backward pass, and `c=1` stays bit-identical to plain Titans
(all Phase 1/2/3 tests pass unchanged). The window is clamped at chunk boundaries;
long-range flow beyond the window still goes through the threaded memory state, as
in the base architecture. Plumbed through `TitansMACBlock(memory_window_size=)` →
`WorldModel` / `membench.RecallModel`. `tests/test_omega.py` checks `c=1` adds no
extra parameters, that `c>1` stays finite and changes the memory relative to
`c=1`, the **grad-of-sum == sum-of-grads correctness identity**, and that a
`WorldModel` with a non-trivial window still trains.

`docs/atlas_vs_titans.md` (French) is a fuller ATLAS-vs-Titans summary: Titans is a
degenerate special case of ATLAS (window `c=1`, identity feature map, first-order
inner optimiser). It tracks three ATLAS ideas and their status here — the Omega
rule (✅ implemented, above), polynomial feature maps `φ(k)` for higher memory
capacity (⬜ not built), and the second-order Muon/Newton-Schulz inner optimiser
(⬜ not built) — plus the DeepTransformers/SWDT connection (⬜ out of scope).

## EgoLife loader (longer, more diverse video)

`src/titans/worldmodel/egolife.py` adds `lmms-lab/EgoLife` as a second video
source, alongside Kinetics-mini: continuous first-person life-logging (6 people ×
7 days), cut into ~30 s / 600-frame / 20 fps segments
(`<person>/<day>/<...>.mp4`). That's 3× longer per segment and far more temporally
continuous than Kinetics-mini's short action clips — a better fit for a
long-memory model — and a bigger, more diverse pool against overfitting. The
train/val split is **by person** (`TRAIN_PERSONS` = 4 people, `VAL_PERSONS` = 2
people, disjoint — no identity leakage); within a person, segments are sampled
*evenly* across the whole week so training data spans diverse scenes rather than
consecutive near-duplicates. Decoding reuses `data.py`'s `load_video_frames`, and
`precompute_latents.precompute_egolife` ties it to the encoder, defaulting to the
continuous sliding-window encoding (`video_to_latents_sliding` — see
`precompute_latents.py` and `tests/test_sliding.py`, which uses a fake
encoder/processor to check the tiling is exactly contiguous: no gaps, no overlaps,
across window/stride combinations).

## membench: synthetic long-range-memory benchmark

`src/titans/membench/` answers a gap the world model can't: **next-step latent
prediction is nearly blind to long-term memory** — it's dominated by local
dynamics, so a working (or broken) memory barely moves that loss. The Titans/ATLAS
papers instead demonstrate memory with tasks that have an *explicit* long-range
dependency, scored by retrieval **accuracy** against a known chance level, not
perplexity. `membench` provides three synthetic token tasks (`tasks.py`), each
padded so the sequence is much longer than `segment_size` — attention only ever
sees within one segment, so solving the task *forces* reliance on the long-term
memory threaded across segments:

* `associative_recall` — show `num_pairs` (key, value) pairs, then query one key;
  predict its value.
* `copy_task` — show a random sequence, then reproduce it in order after a fixed
  lag.
* `needle` — a single marked (payload) token buried in random filler; query at the
  end retrieves the payload (needle-in-haystack).

`model.py`'s `RecallModel` is a minimal token model (embedding + learned
positions + stacked `TitansMACBlock`s + linear classifier head) built purely to
probe the memory — no video/world-model code involved. `run.py`'s `train_recall`
trains on freshly-sampled batches and reports held-out accuracy (mean/std over
unseen seeds) against chance; `accuracy_by_distance` buckets accuracy by how far
the answer sits from its dependency — a flat curve means memory holds at any
depth, a downward slope means it forgets with distance, which is the direct
long-term-memory diagnostic. `tests/test_membench.py` checks task shapes/padding,
that recall tasks have exactly one masked answer per row, and a tiny
forward+backward pass.

## Not yet built

The MAG and MAL variants from the paper (only MAC is implemented); per-channel
(vs scalar) memory gates (`alpha`/`eta`/`theta` are one scalar per token, via a
`Linear(dim, 1)` head, not per-channel); a true parallel scan for the momentum
recurrence (`_scan_chunked` is currently a Python loop across chunks, not a
hardware parallel scan); and, from the ATLAS side, polynomial feature maps `φ` on
keys/queries (for memory capacity beyond what `M(k) ≈ v` allows) and the
second-order Muon/Newton-Schulz inner optimiser (`docs/atlas_vs_titans.md` tracks
both as pending).
