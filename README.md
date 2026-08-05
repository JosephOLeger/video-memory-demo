# titans-mac

A from-scratch PyTorch implementation of Google Research's **Titans**
([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) and **ATLAS**
([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)) neural long-term memory,
applied to a **video world model in latent space**, plus one architectural idea
of our own: a **metaplastic momentum cascade** that comes from neither paper.

The memory is the research object; the world model is what stresses it.

---

## 1. What the memory is

The long-term memory `M` is a small MLP whose **weights are the state**, updated
online at test time by an inner meta-learning loop. At every token:

1. project: `k_t = W_K x_t`, `v_t = W_V x_t`, `q_t = W_Q x_t`;
2. measure the associative loss at the current weights: `ℓ_t = ‖M_{t-1}(k_t) − v_t‖²`;
3. update the weights with momentum ("surprise") + decay ("forget"):

   ```
   S_t = η_t S_{t-1} − θ_t ∇_W ℓ_t
   M_t = (1 − α_t) M_{t-1} + S_t
   ```

4. retrieve `y_t = M_t(q_t)` (forward pass, no update).

`α_t` (forget), `η_t` (momentum decay) and `θ_t` (inner learning rate) are scalars
per token, produced by small learned linear heads on `x_t`.

`src/titans/memory.py` exposes this as `NeuralMemory` with two paths:

| path | role |
|---|---|
| `forward_naive_sequential` | explicit for-loop reference, `torch.func.grad` per step |
| `forward_chunked` | §3.2 parallel form: per-chunk batched gradients via nested `vmap`, then a scanned momentum recurrence |
| `update(x, state, S, chunk)` | the same chunked recurrence **without** retrieval — what MAC uses |
| `retrieve(state, q)` | forward pass at a given weight state |

The two full paths agree to `atol/rtol = 1e-9` in float64 across chunk sizes and
across window sizes (see §7).

**Chunk-start semantics.** The parallel form evaluates every token's gradient at
the weights held at the *start* of its chunk (mini-batch gradient descent). For a
nonlinear MLP this equals the fully-online recurrence only at `chunk_size=1`, so
the reference path takes the same `chunk_size` argument and uses the same
semantics, which is what makes the equivalence test meaningful.

## 2. MAC — Memory As Context

`src/titans/mac.py` implements Titans §4.1 (Eqs 19-25). A sequence `[B, N, d]` is
split into segments of size `C`. The memory carry `(state, S)` — **weights and
momentum** — is threaded across segments *sequentially*; parallelism lives inside
each segment. Per segment `i`:

1. **historical retrieval** from the memory carried in: `h_i = retrieve(state, q_proj(S_i))`;
2. build the context `[persistent tokens ; h_i ; S_i]`;
3. **causal attention** over that window (§4.4 details: SiLU on Q/K/V, L2-normalised
   cosine Q·K, depthwise-separable causal conv after each projection);
4. take the last `C` positions as the segment output `y_i`;
5. **update the memory from `y_i`** — the memory learns from what attention selected;
6. **output** = gate of `y_i` with a fresh retrieval `r_i` from the just-updated
   memory, then a projection, wrapped in a residual.

Because attention only ever sees one segment window, **the threaded memory is the
block's only long-range pathway.** That is what every measurement below exploits.

### Under-specified points, resolved and documented

* `N % C == 0` is required (no padding logic; a clear `ValueError` otherwise).
* **Gate** (§4.4 says only "normalization and gating with a linear layer"): we
  RMSNorm each branch and combine with a learnable per-channel sigmoid gate
  `o = norm(y)·σ(W_r norm(r)) + norm(r)·σ(W_y norm(y))`, then a linear output
  projection with a block-level residual `x + out_proj(o)`.
* Causal-conv kernel size defaults to 4 ("a small kernel" in the paper).
* Memory-update chunk size defaults to the segment size `C`.
* **Not validated against an official reference** — none is public. Correctness
  here means internally consistent, causal, and faithful to the described
  mechanism; not numerically matching a ground truth.

## 3. The ATLAS additions

Titans is the degenerate case of ATLAS: window `c=1`, identity feature map,
first-order inner optimiser. All three ATLAS ingredients are implemented, so
Eq 57-58 is complete:

```
M_t = α_t M_{t-1} + NewtonSchulz5(S_t)
S_t = θ_t S_{t-1} − Σ_{i=t-c+1}^{t} η_i ∇‖M_{t-1}(φ(k_i)) − v_i‖²
```

| ATLAS component | where | flag |
|---|---|---|
| Omega rule, sliding window `c` | `memory.py` | `memory_window_size=c` |
| Muon / Newton-Schulz inner optimiser | `memory.py::newton_schulz5` | `memory_orthogonalize_momentum=True` |
| polynomial feature map `φ` | `features.py` | `feature_degree=p` |
| memory MLP recipe (Table 7) | `memory.py` | `memory_expansion=4`, `memory_activation="gelu"`, `memory_residual=True` |

Every flag is **opt-in and defaults off**, so the Titans paths stay bit-identical
and earlier results remain comparable.

### The Omega rule and its cost

`c=1` (the default) is Titans' delta rule: fit the memory to the **current token
only**. `c>1` fits a gated sliding-window sum over the last `c` tokens (Eq 9):

```
min_M  Σ_{i=t-c+1}^{t}  γ_i^(t) · ‖M(k_i) − v_i‖²
```

so the memory learns a local *context* rather than isolated tokens, with learned
per-token gates `γ_i` pruning what counts.

**Cost.** Because the gradient of a sum is the sum of gradients, `c>1` adds **no
extra backward pass** — the per-token gradients are already computed. It does add
a `c`-term weighted sum per token per weight tensor in the scan, and creates the
`gamma_head` gates. `c=1` takes a dedicated branch with no window arithmetic and
no extra parameters, so **if a run is too slow, `memory_window_size=1` recovers
Titans' original cost exactly** — it is the default, so this is the "do nothing"
option.

**Two defects found and fixed (2026-07-29).** `forward_chunked` never passed
`gamma`, so at `c>1` it summed the window *unweighted* and `gamma_head` received
no gradient at all — a parameter that existed, was counted, and did nothing. And
`forward_naive_sequential` ignored `window_size` outright, returning bit-identical
output for any `c`, so the "reference path" was not a reference for Omega. Both
are fixed and the equivalence test now runs at `c>1`, where the two paths agree
exactly in float64. No published result used those paths, so nothing was
invalidated — but the pattern (a flag that is accepted and silently inert) is the
recurring failure mode of this codebase and is why §7 tests what it tests.

**Known approximation, documented not fixed.** The window is truncated at chunk
starts: at `c=4, chunk=8` the per-position widths are `[1,2,3,4,4,4,4,4]` rather
than a uniform 4. Fixing it needs gradients carried across a chunk boundary, which
conflicts with evaluating them all at the chunk-start weights.

### Documented deviation: `φ` is the diagonal subset

The paper's `φ` is every monomial of degree ≤ `p`. That is unusable here: 2145
dimensions at `d=64, p=2`, 47905 at `p=3` — and because our `M` is an *MLP* rather
than a matrix, we cannot use the kernel trick to avoid materialising it, so the
first memory layer would destroy the parameter budget `φ` exists to protect. We
keep the pure powers and drop the cross terms:

```
φ(x) = concat([c_1·x, c_2·x², …, c_p·x^p]),   dim = p·d,   c_i init 1/√(i!)
```

so `φ(q)·φ(k) = Σ_i c_i² ⟨q^i, k^i⟩` rather than `Σ_i a_i (qᵀk)^i`. The structure
of Eq 5 (a learned, gated mixture of increasing degrees) survives; the exact kernel
identity does not. The tensor-sketch of Kacham et al. (2024a), which ATLAS cites,
is the faithful alternative — not implemented. Also ours and not the paper's:
RMS-normalising before taking powers, without which `x^p` is numerically explosive
for unconstrained keys.

`docs/atlas_vs_titans.md` (French) is a fuller component-by-component comparison.

## 4. Our own contribution: the metaplastic momentum cascade

`src/titans/metaplastic.py`. Titans has a single momentum scalar `S_t`, so one
decay `η_t` controls both reactivity *and* retention — the plasticity-stability
tradeoff. We replace it with a **K-reservoir cascade** inspired by synaptic
metaplasticity (Fusi/Abbott/Drew 2005; Benna & Fusi 2016). Each momentum entry
becomes a vector `u = (u¹,…,uᴷ)`; only `u¹` sees the gradient, and consecutive
reservoirs exchange mass by diffusion:

```
u_t = B·u_{t-1} + e₁·∇ℓ_t          B tridiagonal, b_k = β₁·g^-(k-1)
```

`read(u) = u¹` is the visible momentum fed into `M_t = (1−α_t)M_{t-1} + read(u_t)`.
Opt-in via `memory_use_cascade=True`; **default off**, and `K=1` collapses exactly
onto the scalar rule, so the cascade is a strict generalisation.

**`η_t`'s new role.** Reservoir 1 has no self-decay, so a run of same-sign
gradients grows it linearly. We keep `η_t` as a per-token multiplicative leak on
the *whole* cascade state: `U_t = B_t(η_t U_{t-1}) + e₁(−θ_t ∇ℓ)`. This restores
the missing leak, preserves `η_t`'s data-dependent semantics, and is what makes
`K=1` exact. `cascade_leak` selects where the leak applies (`full`, `shallow`,
`none`, `slow`, `surprise`).

**Two coupling modes.** A fixed geometric `B`, and a **selective** `B_t` where a
learned head gates the couplings *per edge*. A single global scalar gate would give
`B_t = I + s_t·D` for fixed `D` — all such `B_t` commute and reduce to a fixed bank
of `K` exponential filters, no more expressive than the fixed cascade. **Per-edge**
gating reshapes the diffusion so the `B_t` genuinely do not commute. This is a
deliberate deviation from a literal "per-token scalar", documented in the module.

**Stability of `B`, checked not assumed.** `B` is symmetric and row-stochastic ⇒
doubly stochastic ⇒ all eigenvalues in `(−1, 1]` with the top one exactly 1. For
`K=6, β₁=0.5, g=2` the spectrum is `[0.40, 0.77, 0.90, 0.96, 0.99, 1.00]`. The
only caveat is the eigenvalue exactly 1: `u¹` has no self-decay, so under sustained
same-sign gradients the conserved total grows at most *linearly* in sequence
length, and the memory's own `(1−α_t)` forget gate supplies the missing leak.

**Retention, measured.** Impulse response at `K=6` against a single exponential
matched to the same *initial* decay:

| t | cascade `u¹` | exp `0.75ᵗ` | ratio |
|---|---|---|---|
| 10 | 0.389 | 5.6e-2 | 6.9× |
| 100 | 0.208 | 3.2e-13 | 6.5e11× |
| 500 | 0.167 | 3.4e-63 | 4.9e61× |

`u¹` relaxes toward the equilibrium `1/K ≈ 0.167`, never to 0. Over `t ∈ [2,30)`
the excess above equilibrium is **power-law-like** (log-log slope ≈ −0.52,
`R² = 0.994`, against `R² = 0.895` for the matched exponential), then crosses over
to the slowest eigenvalue — the finite-cascade Benna-Fusi picture.

## 5. The world model

`src/titans/worldmodel/`. A `WorldModel` stacks MAC blocks to predict the next
latent `Z_{t+1}` from `Z_1..t`, trained with L1. The core runs on plain latent
tensors with no extra dependencies:

```bash
python -m titans.worldmodel.train    # synthetic-latent demo, L1 drops ~0.53 -> 0.03
```

### Real video

The chain is wired and verified end to end against actual video:

* `encoder.py` — frozen `facebook/vjepa2-vitl-fpc64-256`. A clip `[B,64,3,256,256]`
  → `[B,32,1024]`: 32 temporal groups, each a spatial mean-pool of its 256 patches.
  Runs on CUDA when available (~17× faster: 2.9 s vs 49 s per 64-frame clip).
* `precompute_latents.py` — `video_to_latents_sliding` slides a `clip_len` window by
  `stride_frames` and keeps each window's central band, so adjacent bands tile the
  video into one **continuous** latent stream. We feed *real* frames: V-JEPA-2 is
  motion-aware, so averaging frames before it would destroy the signal it reads.
* `egolife.py` — `lmms-lab/EgoLife`: continuous first-person life-logging, 6 people
  × 7 days, cut into ~30 s / 600-frame segments. The train/val split is **by
  person** (4 vs 2, disjoint — no identity leakage), and within a person segments
  are sampled *evenly* across the week so training spans diverse scenes.
* `data.py` — decode any `.mp4` with `decord`; source-agnostic.

### Input representation: a fixed PCA basis, no learned projection

By default the model learns `in_proj` (1024 → dim) and `out_proj` (dim → 1024).
Two problems with that: **most of a small model's parameters go into I/O rather
than into the memory**, and the projections **co-adapt with the memory**, so "what
did the memory learn?" is entangled with "what did the projection show it".

Measured with `sane_ns` at depth 2:

| config | total params | I/O | **memory** |
|---|---|---|---|
| learned I/O, `dim=64` | 258 k | **51.2 %** | 16.1 % |
| PCA `k=128`, no I/O | 498 k | **0 %** | **33.1 %** |
| PCA `k=256`, no I/O | 1.98 M | **0 %** | **33.2 %** |

`bottleneck.py` + `make_pca_cache.py` fit a PCA on **train only** and write reduced
caches; `WorldModel(io_projection=False)` then runs directly on them with `in_proj`,
`out_norm` and `out_proj` all identities. The reduction becomes a property of the
*dataset*.

```bash
python -m titans.worldmodel.make_pca_cache \
    --train cache/egolife_train_8000.pt --val cache/egolife_val_8000.pt \
    --k 128 --k 256 --out-prefix cache/egolife_pca
```

On the 8000-sequence cache this takes **~20 s per k** (SVD 17 s): `k=128` keeps
**85.7 %** of the variance, `k=256` keeps **92.9 %**. The basis is stored with the
latents so any prediction is invertible to the 1024-D space (`inverse_pca`).

The harness **auto-detects** a pre-reduced cache from its metadata and sets `dim`,
`io_projection=False` and skips standardisation — deliberately detected from the
data rather than trusted to a flag, because this repo has shipped flags that were
silently inert. Note `out_norm` is dropped together with `out_proj`: an RMSNorm
with no projection after it would pin every prediction to unit RMS while the
targets keep their own scale.

**L1 values in PCA space are not comparable to non-PCA runs.** `format_comparison`
prints the space and the explained variance in its header so a number cannot be
misread.

## 6. How memory is measured

Three instruments, in decreasing order of how much they are trusted.

### 6.1 The echo benchmark — `echo.py` (headline)

Next-step latent L1 is nearly blind to long-term memory: it is dominated by local
dynamics. The echo test is not. Train on `[V;V]` — a video concatenated with
**itself**, so the sequence is twice what attention can ever see — predicting `K`
steps ahead. Then score the second copy under two prefixes:

```
[V1 ; V1]   second half = V1, and the memory has just seen V1     -> second_repeat
[V2 ; V1]   second half = V1, but the memory has seen unrelated V2 -> second_cross

memory_gain = second_cross − second_repeat        (larger = better)
```

Both are pure inference on one trained model, and **both are scored on the
identical frames of `V1`**. If the memory retained V1, predicting V1 must be easier
when V1 is what it holds.

A second control, `mem_effect = second_memoff − second`, re-runs the same weights
with `reset_memory_each_segment=True` so nothing crosses a segment boundary: `> 0`
means the threaded memory helps at all.

> **Train this with full BPTT.** A read at position `T+j` in copy 2 and the write at
> position `j` it must credit are exactly `T` apart, and `tbptt_step` detaches the
> carry at every window boundary — so **any `--bptt-window <= T` severs every
> read/write pair** and the benchmark measures nothing. It is not enough for the
> window merely to *cross* the copy boundary. Measured: at `T=144` with
> `--bptt-window 120`, `memory_gain` collapsed from +0.215 to +0.032 on all arms at
> once, with the echo itself gone (`first` 0.690 → `second` 0.684, against
> 0.675 → 0.542 when trained correctly). The lever against VRAM is `T`, via
> `crop_cache.py` — never the window. Windowing stays fine for ordinary next-latent
> training, where the useful dependencies are short.

> **Superseded metric.** An earlier version ranked on
> `content_specificity = second_VW − second`, comparing predicting `W` after `V`
> against predicting `V` after `V`. Those two scored halves hold *different
> content*, so the gap conflated "the memory helps" with "W happens to be harder
> than V". It has been deleted rather than kept as a second ranking. Result files
> written before the change still carry the field; they are historical and not
> comparable to `memory_gain`.

### 6.2 MQAR — `membench/`

`associative_recall`: show `num_pairs` (key, value) pairs, then query one key.
Sequences are padded far beyond `segment_size`, so the only pathway from an early
pair to the late query is the threaded memory. Scored by held-out accuracy against
a known chance level.

`accuracy_by_distance` buckets accuracy by how far the answer sits from its
dependency, which separates "solved inside the attention window" from "solved by
the memory" — the distinction that turned out to explain an earlier plateau.

### 6.3 Memory health — `harness.memory_health`

`pr` (participation ratio of `w1`'s singular values; 1.0 = rank-1 collapse), `norm`,
and `cos_ret` (mean pairwise cosine of retrievals for distinct probe queries;
1.0 = `M` has become a constant function of the query). Probed by running the
block's *own* segment loop on real latents, so the state is the one training
actually produces.

**Health has repeatedly failed to predict task performance** — see §8. It is kept
as a diagnostic of *what broke*, not as a target.

### 6.4 Inner-loop stability — `stability.py`

Not a memory metric: a probe for §8.5. It samples **per chunk**, which is the
granularity the recurrence advances at — the divergence happens inside a single
forward, so anything sampled per optimisation step reads it as gradual and misses it.

`trace_forward` threads the real recurrence and reports, for every chunk: the two
candidate curvatures `λ_max(2 Xᵀ diag(w) X)` for `X = K` (ATLAS's Gram) and
`X = H₀` (the deep one), `‖h₀‖`, `‖W₀‖₂`, `‖W₁‖₂`, `‖S‖`, the realised gates, `κ`,
and the resulting `ϱ(B)`. It uses the **per-token** weights `w_i` rather than a
chunk-level Gram, for the reason in §8.6 — the identity `Σ_i w_i = κ` is a test.

Every raw quantity is returned, not just the derived `ϱ`, so a spectral radius can
be recomputed offline under a corrected formula without re-running anything. Sanity
checks against the numbers the theory note measured independently: `λ_max(KᵀK)`
lands at 7.3-9.0 (note: 7.5-7.9) and `κ` at 14.63 / 28.60 for `η = 0.534 / 0.9`.

## 7. Tests

```bash
pip install -e '.[test]'
pytest                       # 161 tests
```

Beyond shapes and trainability, the suite guards the properties that turned out to
be easy to break:

* the two memory paths agree in float64 (`atol/rtol` to `1e-9`) across chunk sizes
  **and** across window sizes;
* MAC's **causality** across segment boundaries — a change at/after a boundary
  leaves every earlier output bit-identical;
* `grad-of-sum == sum-of-grads` for the Omega window, with and without `φ`;
* every opt-in flag **actually changes the computation** rather than merely being
  accepted — four options were once found silently inert once assembled, and two
  more in the Omega paths (§3);
* the echo probe's defining property: the two scored halves are bit-identical while
  the prefixes differ, and a deliberately memoryless predictor scores exactly
  `memory_gain = 0`;
* PCA is fitted on train only, and `io_projection=False` really leaves no learned
  I/O parameter behind;
* `normalize_hidden` holds `‖h₀‖` to its target *and* leaves `M(k)`'s scale free —
  the pair of properties that separate it from `normalize_output` (§8.8);
* cropping a cache preserves both content order and the `pca` metadata the harness
  keys on, so a cropped corpus cannot silently stop being pre-reduced.

That last class — "does the flag do anything?" — is the one worth extending when
adding a flag.

## 8. Results

### 8.1 The gate/normalisation set (SANE) is what matters

`atlas.py` holds every configuration as a **named** config, diffed against each
other by construction, because an experiment's actual configuration is otherwise
easy to get wrong. The decisive one:

| ingredient | what it fixes |
|---|---|
| `alpha_init=0.002` | forget-gate half-life of **one token** → ~346 tokens |
| `normalize_qk="l2"` | `λ_max ~ m`, so ATLAS's `2/λ_max` bound is ~0.25, not ~0.008 |
| `theta_scale=0.03` | puts the inner step inside that bound |
| `normalize_output=False` | the output norm pins `‖M(k)‖ = 1` against a target at ~0.01, emptying the inner loss |
| `normalize_retrieval=True` | stands in for the paper's chunk-end layer norm; bounds the `h_i` feedback |
| Muon + bias freeze | rank and retrieval diversity |
| `learned_qk_scale` | attention stops averaging its window |

**Every one is inert or harmful on its own.** The gains below are the assembled
result; no single-ingredient claim is supported.

Two of these are worth their own note. The attention temperature: Q/K are
L2-normalised so `q·k ∈ [−1,1]`, and `scaled_dot_product_attention` then shrinks
them *again* by `1/√head_dim`, leaving the softmax near-uniform — measured max
attention weight 0.0556 against 0.0500 for perfectly uniform. Attention was doing
little more than averaging its window, which is what made the memory's per-token
gradients collinear. And `alpha_init`: the default `sigmoid(0) = 0.5` is a
half-life of **one token**, retaining `0.5²⁴ ≈ 6e-8` over a 24-token sequence — a
one-token buffer, not a long-term memory.

### 8.2 Video world model, 6 configs × 5 seeds, 10 epochs

On the 5999-sequence EgoLife cache, with memory rank probed *during* training:

| config | val L1 | vs persistence (0.6561) | PR, start → end |
|---|---|---|---|
| **`atlas_cascade_sane`** | **0.3419 ± 0.0039** | **+47.9 %** | 16.1 → 22.8 |
| **`sane_ns`** | **0.3422 ± 0.0006** | **+47.8 %** | 64.2 → 50.4 |
| `sane` | 0.3754 ± 0.0013 | +42.8 % | 64.2 → 48.2 |
| `atlas_cascade` | 0.4112 ± 0.0291 | +37.3 % | 10.7 → 3.9 |
| `titans` | 0.4566 ± 0.0203 | +30.4 % | 1.0 → 1.3 |
| `atlas` | 0.4733 ± 0.0239 | +27.9 % | 2.9 → 1.2 |

```
titans vs sane_ns              +0.1145   p = 0.008
titans vs atlas_cascade_sane   +0.1147   p = 0.008
titans vs sane                 +0.0813   p = 0.008
```

The three configs that keep the memory healthy are the three best; the three that
collapse are the three worst. **Once SANE is in place the cascade adds nothing on
this metric** — `sane_ns` (no cascade, no `φ`) equals `atlas_cascade_sane`. The
cascade's earlier apparent win was compensating for broken gates, which SANE fixes
at the root. Caveat: next-step L1 is nearly blind to retention, which is precisely
why the echo benchmark exists.

### 8.3 MQAR — out-of-window memory from 0.23 to 0.99

`sane_ns` at 12 pairs, where plain Titans fails to learn the task on half its seeds:

| | in-window (control) | out-of-window (memory) |
|---|---|---|
| `sane_ns`, seed 1 | 1.0000 | **0.9941** |
| `sane_ns`, seed 0 | 1.0000 | **0.9858** |
| same config, **memory branch off** | 1.0000 | **0.0531 / 0.0731** |
| `titans` | 1.0000 | 0.132 |

*(chance = 0.0625)*

The anti-leak control is the point: disable the memory branch and out-of-window
accuracy falls to **exactly chance** while in-window stays at 1.0. Nothing but the
threaded memory can carry the answer. Full unsegmented attention scores 0.997 — so
in its good mode the memory does as well as seeing the entire sequence.

**It is bimodal, and that is the open problem.** Roughly half the seeds reach the
high mode; the rest sit at the Titans baseline. The two modes are
**indistinguishable by rank** (PR 25.15 vs 25.18), so the bimodality is not the
collapse. The memory is healthy either way; one seed learns to use it and the other
does not. The open question is an optimisation one.

### 8.4 Echo sweep, 14 configs × 2 horizons × 3 seeds (75/84 usable cells)

Measured on the **superseded** `content_specificity` metric (§6.1), at `K=24`,
which is the horizon where any signal exists (`K=8` gives ~+0.004 for every config):

| rank | config | K=24 | n |
|---|---|---|---|
| 1 | `cas_r4_seg4` | +0.1404 ± 0.0244 | 3 |
| 2 | `cas_r2` | +0.1310 ± 0.0267 | 3 |
| 3 | `cas_tag` | +0.1300 ± 0.0123 | 2 |
| 5 | `cas_r4` | +0.1257 ± 0.0241 | 2 |
| 6 | **`cas_indep`** (null test) | **+0.1254 ± 0.0090** | 3 |
| 12 | `cas_leakfull` | +0.0691 ± 0.0205 | 2 |
| 13 | `nocas_seg4` | +0.0603 ± 0.0135 | 3 |
| 14 | `nocas_seg16` | +0.0403 ± 0.0047 | 2 |

Three readings, in order of importance:

1. **The cascade helps, but the coupling does not.** `cas_indep` — `K`
   *independent* exponential filters, no coupling at all — scores +0.1254 against
   +0.1257 for the coupled chain. Indistinguishable, and with lower variance. The
   benefit comes from **having multiple timescales**, not from the non-commuting
   diffusion. The operator-set argument for per-edge gating stands mathematically;
   this experiment does not see it.
2. **More reservoirs is worse**, monotonically: r2 > r4 > r6 > r8.
3. **`leak` is the real lever.** `cas_leakfull` collapses to the no-cascade level
   while `leak=none` and `leak=surprise` lead — not draining the reservoirs is what
   the Benna-Fusi retention argument predicts, and it holds.

`PR` predicts nothing here: `nocas_seg16` has the best rank (24.2) and the worst
score.

**Caveat.** These numbers use the confounded metric. Conclusion 1 compares two
cascade configs on the same videos, so the bias largely cancels; the
cascade-vs-no-cascade ranking is more exposed and is being re-measured with
`memory_gain`.

### 8.5 SANE's failure mode: an abrupt in-forward divergence

`normalize_output=False` is SANE's fragile ingredient. On some seeds and not
others, at the same dim, a run diverges to all-NaN. Localised per-step:

```
step 1639:  logits=BAD  loss=BAD  grad=BAD  params_before_step=OK  params_after_step=BAD
```

The outer weights were **still finite entering the forward**, and the forward
itself produced NaN. It is not an optimiser blow-up and not a gradient explosion —
it is the online memory's inner loop overflowing *within a single forward pass*.
The 30 preceding steps are healthy (loss 3.73-4.23, grad-norm 1.5-2.6), then
everything is NaN in one step. **Any probe coarser than per-step mis-reads this as
gradual.**

Both candidate fixes work (baseline dies at accuracy 0.000):

| arm | divergence | final acc |
|---|---|---|
| baseline SANE | step 1639 | 0.000 |
| `+ normalize_output=True` | none | 0.375 |
| `+ theta_scale=0.01` (halved) | none | 0.375 |

`theta_scale=0.01` is the better lead: it stays inside the SANE design, whereas
restoring the output norm re-introduces exactly what SANE removes.

The divergence is **deterministic per seed**: re-running the same seed reproduces
the same NaN bit-for-bit, so a failed cell must be replaced with a *different* seed,
never retried. It affected ~13 % of echo-sweep cells, and it is what invalidated an
earlier capacity sweep — so **memory capacity remains unmeasured**.

**Reproducing it needs the right task.** The cell above is `sane_ns`, `dim=32`,
`segment_size=8`, `num_pairs=8`, batch 32, and **`key_vocab = val_vocab = 64`**.
That last one is not incidental: re-running at `vocab=16` (chance 1/16 instead of
1/64) is an easier task with a different key geometry and gave **0 divergences in
25 cells**. `membench/nan_sweep.py` is the runnable cell, with those values as
defaults.

### 8.6 Why ATLAS's bound does not cover this

`θ` sat at 0.021-0.022 against ATLAS's inner-loop bound `2/λ_max(K Kᵀ)` — a
*factor ~10 inside* it — yet halving `θ` prevents the blow-up. Working out why is a
separate note (`TheoTitanFr.pdf`); what it establishes, and what it rules out:

**The convention first.** ATLAS's p.26 iterate `M − η(MK − V)Kᵀ` is the gradient of
`½‖·‖²`, so their `2/λ_max(KKᵀ)` presupposes the ½. Our inner loss (`_grad_fn`) has
no ½, so **our budget is `1/λ_max(KKᵀ)`, half theirs**. Every number below is stated
in `λ_max(H)` with `H = 2 KᵀK` to keep that explicit. (ATLAS is internally
inconsistent here: its Prop. 1 writes `ℓ = ‖MK − V‖²` without the ½.)

Two structural departures from the setting ATLAS's bound is stated in, and **neither
explains the divergence**:

* **Momentum + decay.** The exact characteristic polynomial of the
  `S_t = η S_{t−1} − θ g`, `M_t = (1−α)M_{t−1} + S_t` recurrence (§1) gives stability iff
  `0 ≤ θ λ_max(H) < (1+η)(2−α)`. Momentum *widens* the admissible step (`2 ↦ 2(1+η)`)
  — the classic heavy-ball result. It cannot cause a blow-up inside ATLAS's bound.
* **Stale chunk gradients.** Every gradient in a chunk is evaluated at the chunk-start
  anchor, so the same error is acted on `m` times without re-evaluation, inflating the
  effective step by `κ = Q_m/(1−η)` (14.6 at rest, **28.6 at `eta_init=0.9`**). But the
  chunk map takes the **per-token** Hessians `H_i = 2 k_i k_iᵀ`, weighted
  `w_i = Σ_{j≥i} a^{m−j} η^{j−i}` with `Σ_i w_i = κ`. Multiplying `κ` by a chunk-level
  `λ_max(KᵀK)` — already summed over the `m` tokens — double-counts by ~`m`. Done
  correctly, `ϱ(B) ≈ 0.32`: `θ` at 26 % of the chunk budget, a margin of ×3.9.
  Verified against exact simulation of the recurrence.

What is left is **the deep memory's own curvature**. Write `M`'s own layers
`W₀, b₀, W₁, b₁` (not to be confused with the outer projections `W_K/W_V/W_Q`, which
are frozen for the inner loop). For `ℓ = ‖M(k) − v‖²` the Gauss-Newton curvature of
the *last layer's* block is `λ_max(H_GN^(W1)) = 2‖h₀‖²` with `h₀ = σ(W₀k + b₀)`. ATLAS's `λ_max(KKᵀ)` is the curvature of a **matrix**
memory, where `J = k` and the keys' Gram matrix *is* the Hessian. For a deep memory
it is not: `normalize_qk` pins `‖k‖ = 1` but bounds neither `‖h₀‖` nor `‖W₁‖`, both
online-updated and free to grow inside the inner loop. **The bound sits on the wrong
Gram matrix** — the precise content of "necessary but far from sufficient".

Predicted thresholds at `θ = 0.0208`, `η = 0.534`, `m = 8`: unstable once
`θ·Λ_chunk > 2.37`, i.e. **`‖h₀‖ > 1.97`** against 0.29 at initialisation; a
sufficient condition is `‖W₀‖₂ + ‖b₀‖₂ < 1.97`. This also explains both known fixes
as one mechanism: `normalize_output=True` bounds the residual so `2Jᵀr` cannot
inherit `‖h₀‖`'s growth (at the cost of the associative signal, §8.1), and halving
`θ_scale` attacks the same product `θλ` from the other side.

`stability.py` measures all of it per chunk (§6.4), and `normalize_hidden` bounds
`‖h₀‖` directly (§8.8).

### 8.7 Initialising the momentum gate matches a whole cascade — and it replicates

On `memory_gain` (§6.1), `T=64`, `K=24`, full BPTT, batch 8, 2500 steps, PCA `k=128`.
`nocas_eta10` is `nocas` plus one changed bias: `--eta-init 0.9`, i.e. a momentum
horizon `1/(1-η)` of 10 tokens instead of the default 2.1. Permutation tests, paired
by seed. **0 NaN in either sweep** — the PCA representation appears to suppress §8.5.

| arm | small corpus, 0.38 M latents (n=24) | full EgoLife, 2.80 M latents (n=8) |
|---|---|---|
| `nocas` | +0.2154 ± 0.0249 | +0.1666 ± 0.0120 |
| `nocas_eta10` | +0.2377 (**+0.0223**, p=0.0027) | +0.2236 (**+0.0571**, p=0.008) |
| `cas_tag` | +0.2475 (**+0.0321**, p=0.0002) | +0.2063 (**+0.0397**, p=0.016) |
| `eta_init` − cascade | −0.0098, p=0.20 **ns** | +0.0173, p=0.29 **ns** |

Three readings:

1. **One bias buys what the architecture buys.** Both beat the baseline in both
   sweeps, and the two are never distinguishable from each other — at 7× the data
   the sign of their gap even flips, which is what "indistinguishable" looks like.
   This generalises the `alpha_init`/SANE lesson (§8.1): gains credited to
   multi-timescale machinery are largely recoverable by initialising the gates that
   are already there. `η` had gone unfixed while `α` was fixed, so the memory's
   *accumulation* stayed short-range however long its retention became.
2. **The effect grows with scale.** What moved is the baseline: `nocas` fell 23 %
   while `nocas_eta10` fell 6 %. The harder corpus punishes the un-tuned gate, so
   this does not wash out — the usual worry about a small-corpus result.
3. **`eta_init` carries more seed variance** (SD 0.0295 against 0.0120), so it buys
   a higher mean at a wider spread.

**Caveat.** The two corpora differ on two axes at once: 7.3× more latents *and*
208 ms per latent instead of 469 ms. The baseline's drop could be the finer
resolution rather than the volume; separating them needs a run at matched
resolution. And "not significantly different" is not "equal" — resolving the
residual gap at 3 SE would need ~120 seeds per arm.

### 8.8 `normalize_hidden`: free on plain SANE, expensive on `eta_init`

§8.6 says the unbounded quantity is `‖h₀‖`, so bound it: `normalize_hidden`
L2-normalises the memory's hidden activation, making `λ_max(H_GN^(W1)) = 2‖h₀‖²` a
constant. **L2, not RMS** — an RMSNorm leaves `‖h₀‖ = √d_h ≈ 22` at ATLAS's
expansion 4, against a threshold near 2, so it would not bound the quantity that
matters. The target norm is a parameter: at init `‖h₀‖ ≈ 0.3`, so a ceiling of 1.0
caps the curvature while *raising* the resting point ~10×.

Unlike `normalize_output` it costs **no associative signal**: `M(k) = W₁h₀ + b₁`
keeps its scale free, so `W₁` can still shrink to meet `‖v‖ ≈ 0.01`. Pinned by test —
fitting a small-norm target converges with `normalize_hidden` and does not with
`normalize_output`.

Echo benchmark, full EgoLife `T=64` corpus, **n=8 per arm, 0 NaN in all 40 cells**,
paired permutation tests:

| arm | `memory_gain` | vs its control | p (paired) |
|---|---|---|---|
| `nocas` | +0.1666 ± 0.0120 | — | — |
| `nocas_nh` (target 1.0) | +0.1677 ± 0.0135 | **+0.0011** | 0.88 **ns** |
| `nocas_nh03` (target 0.3) | +0.1655 ± 0.0113 | −0.0010 | 0.87 **ns** |
| `nocas_eta10` | +0.2236 ± 0.0295 | +0.0571 vs `nocas` | 0.0078 |
| `nocas_eta10_nh` | +0.1911 ± 0.0127 | **−0.0326 vs eta10** | **0.0078** |

1. **On plain SANE it is free** (+0.0011, p=0.88) — which `theta_scale=0.01` is not
   (it costs `memory_gain` at p=0.015) and `normalize_output` is not (it empties the
   loss). The target norm does not matter (0.3 vs 1.0: p=0.72), so the result does
   not hinge on tuning.
2. **On the `eta_init` arm it costs**: −0.0326, removing 57 % of eta_init's gain.
   Still clearly above `nocas` (+0.0245, p=0.0007), so not a cancellation.
3. **It halves that arm's seed variance** (SD 0.0295 → 0.0127, back to `nocas`'s
   0.0120). Consistent with — not proof of — part of `eta_init`'s gain coming from
   operating near the instability edge: `κ` is 28.6 at `eta_init=0.9` against 14.6 at
   rest, and §8.6 puts the resting point already past the predicted threshold there.
   The theory said `eta10` is where the fix would bite hardest, and it is.

The controls reproduced §8.7 to four decimals, which also confirms the flag is inert
when off.

**What this does not yet show is that it prevents the divergence it was built for.**
PCA space gives 0 NaN everywhere, so the sweep above had nothing to prevent; that
question lives on MQAR (§8.5) and is open.

## 9. Data/parameter proportions: an accepted deviation

ATLAS trains at **40-88 tokens per parameter** (Table 7). Our video corpus is
`5999 × 64 = 383,936` latents, so that band would cap the model at **4.4k-9.6k
parameters**; ours are 143k-2M, i.e. 15× to 130× off. Closing it on a 1M-parameter
model would need ~5,200 h of video against the ~8,000 segments we have.

**We accept the deviation, and state the consequence: loss comparisons against the
paper are not interpretable; only internal A/B comparisons between our own variants
are.**

But the ratio is weak evidence here and should not be over-read. It is an
LLM-scaling heuristic assuming near-independent tokens, and consecutive video
latents are massively redundant. Empirically it mis-predicts: at a ratio of 0.01 the
measured train/val gap was 0.043 and the model beat the persistence baseline. The
honest indicator is the measured generalisation gap, not the ratio. On MQAR the
question is moot — tasks are generated on the fly.

The intrinsic dimensionality of the V-JEPA-2 latents, measured on 128k vectors:

| variance kept | components |
|---|---|
| 50 % | 9 |
| 90 % | 112 |
| 95 % | 204 |
| 99 % | 484 |
| **effective dimension (participation ratio)** | **~23** |

So the latents are "small" in variance terms but need 200-500 dimensions for
faithful reconstruction — which is what `k=128` / `k=256` bracket.

## 10. Running experiments

```bash
# named world-model configs on a latent cache (PCA or raw, auto-detected)
python -m titans.worldmodel.atlas \
    --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
    --configs sane_ns,sane_ns_cascade --seeds 0,1,2 --epochs 10 --health-every 200

# the echo benchmark for one config
python -m titans.worldmodel.echo --label cas_r4 \
    --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
    --horizon 24 --seed 0 --cascade --reservoirs 4 --leak none --readout mean \
    --out result.json
```

`harness.py` provides the statistical scaffolding: `run_config` trains one config
over several seeds and reports `mean ± std`, plus the **persistence baseline**
(`ẑ[t+1] = z[t]`) so a number means something; `compare_variants` /
`format_comparison` tabulate several configs, with the train/val gap flagging
overfitting (data-limited) against underfitting.

### Preparing a corpus

```bash
# encode the full corpus into resumable shards (one GPU slice per process)
python -m titans.worldmodel.encode_egolife --split train --out-dir cache/egolife_full --gpu-slice 0/4

# fit the PCA basis on train only, in O(d^2) memory, and write both reduced caches
python -m titans.worldmodel.sharded --shard-dir cache/egolife_full --k 128 \
    --out-prefix cache/egolife_full_pca

# cut to shorter sequences so [V;V] fits under FULL BPTT (see §6.1)
python -m titans.worldmodel.crop_cache --src cache/egolife_full_pca128_train.pt \
    --dst cache/egolife_full_pca128_T64_train.pt --T 64
```

### Running a sweep on the shared box

Write one command per line into a queue file, then let a cron-driven pool drain it:

```bash
# 1. size the guard on the HEAVIEST arm -- never extrapolate from the lightest
python -m titans.worldmodel.measure_vram --train ...T64_train.pt --val ...T64_val.pt \
    --horizon 24 --batch 8
#    nocas 6.36 / nocas_eta10 6.36 / cas_tag 9.34 GiB  -> guard 13000 MiB

# 2. eight workers, two per card, restored every ten minutes if any die
JOBS=jobs_ref64.txt GUARD=13000 PREFIX=S bash scripts/keepalive.sh
crontab -l | { cat; echo "*/10 * * * * JOBS=jobs_ref64.txt GUARD=13000 PREFIX=S /bin/bash \$HOME/work/titans-mac/scripts/keepalive.sh"; } | crontab -
```

The MQAR divergence cells are the same shape of job, and cheap (under a GiB each,
8-25 min depending on whether Muon is on):

```bash
python -m titans.membench.nan_sweep --config sane_ns --seed 0 --out cell.json
python -m titans.membench.nan_sweep --config sane_ns --seed 0 --normalize-hidden 1.0
```

Arms must share the seed grid: the divergence is deterministic per seed, so a NaN
rate only means something against a control at the *same* seeds. Raise `WORKERS`
for these — twelve fit where two echo jobs would.

`scripts/pool4.sh` is the worker: a free-VRAM guard, one requeue on failure then a
drop, and a per-job timeout so a hung job cannot hold a card for days. Two hard-won
details are baked into it, both of which have cost whole sweeps:

* **The guard must be sized on the heaviest arm.** `cas_tag` needs +47 % over
  `nocas` — from `--readout gated` and `--selective`, about +24 % each; the cascade
  *reservoirs* are nearly free. A guard set from `nocas` puts two heavy jobs on one
  card. Cost: 74 jobs (2026-07-29), 22 (07-31).
* **The guard is a check-then-act race.** Workers polling a cold pool in the same
  second all read the card as free and all launch. `pool4.sh` holds a per-card
  `flock` across *read free → pop job → let it allocate* (`SETTLE=120 s`), so the
  second worker on a card reads a post-allocation number.

Swapping a running pool is safe: `kill` the `pool4.sh` **parent** (not the process
group) and its `timeout … python …` child survives to finish and write its JSON,
while the worker stops pulling new work. Give the replacements a new `PREFIX` or the
keepalive's `pgrep` will think they are already running.

## 11. Not built

* The **MAG** and **MAL** variants (only MAC is implemented).
* **Per-channel** memory gates — `α`/`η`/`θ` are one scalar per token, via a
  `Linear(dim, 1)` head.
* A **true parallel scan** for the momentum recurrence (`_scan_chunked` is a Python
  loop across chunks).
* The **tensor-sketch feature map** that would restore `φ`'s cross terms.
* A **VLM / language head** — the model predicts latents, not text. Grafting the
  memory into a pretrained open multimodal model (frozen backbone, zero-init output
  gate, memory on a per-frame pooled stream) is the natural next scope; sketched, not
  started.
* A **bound on the first layer's curvature**. `normalize_hidden` (§8.8) bounds the
  last layer's block by construction; the first layer's
  `≤ 2‖k‖²‖W₁ diag σ′‖²` still depends on `‖W₁‖`, which nothing constrains.
* **Open problems from the theory note**: lifting the linearisation to genuinely
  per-token `(H_i, v_i)`; measuring the predicted threshold `θ⋆ ∝ (1−η)/m` by
  sweeping `θ_scale` and `m`; a switched-system certificate (common Lyapunov `P`, or
  a bound on the joint spectral radius) for the data-dependent gates; and the
  admissible region in the `(α, η)` plane, where `α → 0` removes the contraction and
  `η → 1` makes `κ` diverge — long retention and long accumulation are in direct
  tension.
