# titans-mac

A from-scratch PyTorch implementation of Google Research's **Titans**
([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) and **ATLAS**
([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)) neural long-term memory,
applied to a **video world model in latent space**, plus one architectural idea
of our own: a **metaplastic momentum cascade** that comes from neither paper.

The memory is the research object; the world model is what stresses it.

---

## 0. Orientation — read this first

**The question.** Can a memory carried in *weights* hold episodic information that
attention cannot hold at matched cost? Not "is it the best world model" — the object
is an associative memory, judged on recall, not on prediction loss. Never call our own
chunked block "quadratic"; it isn't.

**Where the project stands.**

| | |
|---|---|
| settled | the divergence is a **curvature feedback** in the memory MLP's own hidden layer, and ATLAS's un-rescaled Newton-Schulz removes it — 104 800 writes with no wall against 80 for the control (§9.13, §9.14); reach is set by `α` alone under a bounded step, `α* ≈ 1/(2d)`, `SNR* ≈ 0.61√(P/d)` (§9.15); the gate/normalisation set that works (`SANE`, §9.1); the graft is bit-identical to the frozen host at init and costs 0.0092 % of its parameters (§6) |
| live | what the bound **costs**: `unit` gave −59 % `memory_gain` on the world model, and whether that is the price of stability or a mis-scaled step is what the α-law sweep and the `squash` arm are measuring; whether the cascade still *functions* under a bound that discards its readout magnitude — a stability trace cannot tell, only retention can |
| refuted | more memory buys more reach — it bought **less** (3.8 → 0.5 min, h32 → h512), because wider diverges sooner (§9.16); the stability *margin* as a survival criterion (§9.9); `normalize_hidden` as a fix — it is a 54× delay, still exponential (§9.14) |
| abandoned | the synthetic MQAR/copy/needle benchmark and its `membench` package, removed 2026-08-14. EgoLifeQA is the benchmark; **we do not invent new ones** |
| next | train the branch on EgoLife with the host frozen, `unit` (or `squash`) + cascade, `α` set from the closed form. The evaluation benchmark is still unchosen and must not be EgoLife |

**Standing constraints on new experiments.**

* Every new arm is a **delta on `sane_ns`** — no runs outside the SANE gate set.
* Cascade results must never rest on **r = 0 pure accumulators** (`cas_none`,
  `cas_r0.03`); they are diagnostics of the no-leak limit, not the architecture.
* The BPTT window must **exceed the copy length `T`**, or `memory_gain` measures
  nothing (§7.1).
* The GPUs are shared. Size the VRAM guard on the **heaviest** arm, never extrapolate
  from the lightest (§11).
* Register predictions **before** launching anything that could be read either way —
  `results/predictions_*.md`. The margin was refuted this way and it is the reason we
  trust the refutation.

**Where things are.** Code here; real runs on the workstation
(`~/work/titans-mac`, synced by rsync — verify with `md5sum`, not `git status`).
Raw JSON is mirrored into `results/box/`, and `scripts/nightly_report.sh` renders the
CSVs, figures and `ETAT.md` into Josep's results folder every 30 minutes. Multi-phase
sweeps self-drive from one cron entry through `scripts/tau_chain.sh` (§11).

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
across window sizes (see §8).

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
recurring failure mode of this codebase and is why §8 tests what it tests.

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
caches; `WorldModel` runs directly on them. The reduction is a property of the
*dataset*, not a learned layer.

**The learned I/O projection was removed on 2026-08-06** (`in_proj`, `out_norm`,
`out_proj` are gone, not identities). A PCA cache is now the only accepted input:
`latent_dim` must equal `dim`, and a raw 1024-D cache raises rather than being
standardised into a model whose results are comparable to nothing. Runs in the old
learned-projection space are retired — they are also where 12 of the 15 non-finite
echo runs came from.

```bash
python -m titans.worldmodel.make_pca_cache \
    --train cache/egolife_train_8000.pt --val cache/egolife_val_8000.pt \
    --k 128 --k 256 --out-prefix cache/egolife_pca
```

On the 8000-sequence cache this takes **~20 s per k** (SVD 17 s): `k=128` keeps
**85.7 %** of the variance, `k=256` keeps **92.9 %**. The basis is stored with the
latents so any prediction is invertible to the 1024-D space (`inverse_pca`).

The harness **auto-detects** a pre-reduced cache from its metadata and sets `dim`
from it — deliberately detected from the data rather than trusted to a flag, because
this repo has shipped flags that were silently inert. `out_norm` went with
`out_proj`: an RMSNorm with no projection after it would pin every prediction to
unit RMS while the
targets keep their own scale.

**L1 values in PCA space are not comparable to non-PCA runs.** `format_comparison`
prints the space and the explained variance in its header so a number cannot be
misread.

## 6. The graft — Qwen2.5-VL

The paper's actual claim. Everything before this section is evidence that the block
is sound; this is where it is attached to something that already sees the world.

**Host: Qwen2.5-VL-3B-Instruct**, read from its config rather than assumed — 36
decoder layers, `d_model` 2048, 3.75 B parameters. Graft site is layer **24** (⅔ of
depth): late enough that the representation is abstract, early enough that twelve
layers can still use what the memory returns.

### 6.1 The block only ever adds

`graft/branch.py`. The branch reads the decoder's hidden states at one layer, writes
them into a `NeuralMemory`, retrieves, and **adds** the result back into the residual
stream. It does not replace an attention block, does not shorten the context, and
does not touch a single host parameter (`freeze_host` returns the count).

`up` is zero-initialised, so at step 0 the grafted model is **bit-identical** to the
frozen host — `identity_check` asserts `torch.equal` on the logits, not `allclose`.
That is the cheapest correctness test the project has (it catches a dropped
`attention_mask`, a mangled cache tail, a graft onto the vision tower), and it only
exists before training starts. MAC's `_gate` cannot do this: it mixes `y` and `r`
multiplicatively and has no setting that returns `y` unchanged.

**The guarantee has one condition, and it is not cosmetic: `0 × NaN = NaN`.** A
diverged memory propagates straight through a zero `up`. Kept deliberately — a
divergence must be loud — but never state the identity without it.

### 6.2 The input basis is a fitted PCA, not a learned projection

`graft/fit_basis.py` fits the memory's input basis on **the frozen host's own
layer-24 activations**, over real EgoLife video. Same move as §5: the reduction
becomes a property of the data instead of parameters free to co-adapt with the
memory, and freezing it drops the branch from 606 979 to **344 835 trainable
parameters = 0.0092 % of the host**.

Fitted on 24 clips, 12 288 activations, `k=128` → 70.5 % of the variance. It also
puts a number on a real pathology: **per-channel std ratio max/median = 119×** on the
residual stream, which is why `fit_pca` standardises before the SVD.

What it does **not** do is fix the divergence. Measured both ways: without the basis
the wall sits at ~256 writes, with it ~328. It moves the wall; §9.13 removes it.

Qwen2.5-VL uses *dynamic* resolution and will not downscale for you — 32 native
EgoLife frames ask the vision tower for 11.3 GB in one allocation. `resize_frames`
caps the longest side (256 → 1324 tokens per 32 frames).

### 6.3 Read cadence is decoupled from write cadence

A memory holds a trace for a finite number of **writes**, not tokens. Writing once
every `write_stride` tokens multiplies reach by that stride at **zero VRAM cost**,
while the host still sees every frame and the branch still reads at every position.
The price is temporal resolution, never context. Streaming callers must pass
`token_offset` or the stride's phase silently resets at every call.

This matters for the honest version of the GPU ask: the claim is not "impossible
without bigger cards" but "at 23 GB we get X minutes **at full temporal
resolution**".

### 6.4 What it costs, measured on the real model

`graft/check_qwen.py` runs base-then-graft in one process (no second weight copy)
and reports identity, VRAM and time at real token counts.

| | |
|---|---|
| inference overhead | +0.39 GB at 676 tokens, +0.75 at 1324, +1.48 at 2620 |
| training, 1024 tokens | 1024 writes → 14.3 GB / 7.2 s; 64 → 11.4 GB / 0.6 s; 16 → 11.2 GB / 0.3 s |

**VRAM is read in writes, not tokens** — the backward keeps one memory state per
write. Separately, at 2620 video tokens the backward OOMs *even at 11 writes*: that
one is the host's own backward through layers 24-35, so training needs shorter
windows or checkpointing regardless of the memory.

**`GraftedLayer` is not checkpoint-safe as written.** Its forward mutates `carry` and
`token_offset`, and `torch.utils.checkpoint` replays the forward during the backward —
which would advance the memory a second time over the same tokens and double the
stride's phase, silently. Checkpoint `self.layer` only: that is where the OOM actually
is, it is a pure function with `use_cache=False`, and the branch's inner loop is the
most expensive thing per byte saved in the whole model. HuggingFace's
`gradient_checkpointing_enable()` cannot be used, since it wraps the modules in the
`ModuleList` — which is now the wrapper.

---

## 7. How memory is measured

Three instruments, in decreasing order of how much they are trusted.

### 7.1 The echo benchmark — `echo.py` (headline)

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

### 7.2 Memory health — `harness.memory_health`

`pr` (participation ratio of `w1`'s singular values; 1.0 = rank-1 collapse), `norm`,
and `cos_ret` (mean pairwise cosine of retrievals for distinct probe queries;
1.0 = `M` has become a constant function of the query). Probed by running the
block's *own* segment loop on real latents, so the state is the one training
actually produces.

**Health has repeatedly failed to predict task performance** — see §9. It is kept
as a diagnostic of *what broke*, not as a target.

### 7.3 Inner-loop stability — `stability.py`

Not a memory metric: a probe for §9.4. It samples **per chunk**, which is the
granularity the recurrence advances at — the divergence happens inside a single
forward, so anything sampled per optimisation step reads it as gradual and misses it.

`trace_forward` threads the real recurrence and reports, for every chunk: the two
candidate curvatures `λ_max(2 Xᵀ diag(w) X)` for `X = K` (ATLAS's Gram) and
`X = H₀` (the deep one), `‖h₀‖`, `‖W₀‖₂`, `‖W₁‖₂`, `‖S‖`, the realised gates, `κ`,
and the resulting `ϱ(B)`. It uses the **per-token** weights `w_i` rather than a
chunk-level Gram, for the reason in §9.5 — the identity `Σ_i w_i = κ` is a test.

Every raw quantity is returned, not just the derived `ϱ`, so a spectral radius can
be recomputed offline under a corrected formula without re-running anything. Sanity
checks against the numbers the theory note measured independently: `λ_max(KᵀK)`
lands at 7.3-9.0 (note: 7.5-7.9) and `κ` at 14.63 / 28.60 for `η = 0.534 / 0.9`.

**`ϱ` is meaningless on a bounded-step path.** The 2×2 chunk map is derived for a step
*proportional to the gradient*; under `unit` the step no longer scales with the state,
so `ϱ` reads 1.3-1.8 while the state is provably flat. Read the measured norms
(`w1_fro`, `S_norm`, `h0_norm_max`) there, as the cascade path already does by
returning `None`.

**The probe must survive what it measures.** `weighted_gram_lambda_max` and
`chunk_transfer` report `inf` on a non-finite or unfactorable input rather than
raising (`‖G‖_F ≥ λ_max` is the fallback), and `trace_forward` takes an `on_row`
callback so a caller can stream rows as they are produced. Both come from a
40-minute trace that died on `linalg.eigh: failed to converge` and lost every row.

`graft/trace_stability.py` drives all of this on the real host. Use `--videos` (a
sorted glob of consecutive EgoLife clips, i.e. one continuous recording) for anything
that claims a wall; `--repeat` tiles a single clip and makes the probe **easier**, not
harder — at period `P` the memory learns the loop and the driving gradient dies.

## 8. Tests

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
* PCA is fitted on train only, no learned I/O parameter survives (every parameter
  is under `blocks.`), and a width mismatch raises instead of being projected away;
* `normalize_hidden` holds `‖h₀‖` to its target *and* leaves `M(k)`'s scale free —
  the pair of properties that separate it from `normalize_output` (§9.7);
* cropping a cache preserves both content order and the `pca` metadata the harness
  keys on, so a cropped corpus cannot silently stop being pre-reduced.

That last class — "does the flag do anything?" — is the one worth extending when
adding a flag.

## 9. Results

> **Read every "η = x" label below as "`eta_init` = x", not as the momentum the
> model ended up using.** Measured 2026-08-14 on 58 frozen artefacts: the outer loop
> moves the momentum gate far off its initialisation, and it does so **bimodally** —
> either back to ~0 or roughly to where it started, nothing between.
>
> | `eta_init` | learned `η` per seed |
> |---|---|
> | 0.7 | 0.000, 0.000, 0.043 |
> | 0.9 | 0.001, 0.000, 0.961, 0.631, 0.032, 0.000 (6 seeds) |
> | 0.95-0.97 | 0.93-0.98 on two seeds, 0.05 on the third |
> | 0.995-0.999 | 0.98-0.997 on all seeds |
>
> So `eta_init` selects a **basin**, it does not set `η`, and the boundary sits near
> 0.95. Two consequences, both unresolved at the time of writing: any statement here
> of the form "at η = 0.9, so `τ_acc` = 10 tokens" is about a model whose real
> `τ_acc` is ~1 for most seeds; and the `τ*` bracket of §9.9 is computed on that
> mislabelled axis, so its *numbers* are provisional even though the transition it
> brackets is real. Worse for §9.8: the single 40 000-step arm has `η = 6e-6` **and**
> `α = 2e-6`, i.e. no momentum and no forgetting — a pure accumulator on `M`. Its
> convergence numbers are measured and stand; the mechanism attributed to them does
> not.
>
> Note also the anticorrelation, which is the §9.11 compensation loop appearing on the
> scalar arm: the seeds that keep `η ≈ 0.96` train `θ_max ≈ 0.004`, those that drop to
> `η ≈ 0` train `θ_max ≈ 0.024-0.043`. Momentum and step size are substitutes, and the
> outer loop picks one.

### 9.1 The gate/normalisation set (SANE) is what matters

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

### 9.2 Video world model, 6 configs × 5 seeds, 10 epochs

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

### 9.3 Echo sweep, 14 configs × 2 horizons × 3 seeds (75/84 usable cells)

Measured on the **superseded** `content_specificity` metric (§7.1), at `K=24`,
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

### 9.4 The divergence, as phenomenology

> Mechanism in §9.13, fix in §9.14. This section is the shape of the failure, which
> is what dictates how it must be probed.

On some seeds and not others, at the same dim, a run goes all-NaN. Localised
per-step:

```
step 1639:  logits=BAD  loss=BAD  grad=BAD  params_before_step=OK  params_after_step=BAD
```

The outer weights were **still finite entering the forward**, and the forward itself
produced NaN. Not an optimiser blow-up, not a gradient explosion — the online memory's
inner loop overflowing *within a single forward pass*. The 30 preceding steps are
healthy (loss 3.73-4.23, grad-norm 1.5-2.6). **Any probe coarser than per-chunk
mis-reads this as gradual**, which is why `stability.py` samples per chunk (§7.3).

**Deterministic per seed**: the same seed reproduces the same NaN bit-for-bit, so a
failed cell must be replaced with a *different* seed, never retried. It affected ~13 %
of echo-sweep cells.

This section long claimed the divergence had *never been reproduced in PCA space* —
0 NaN in 160+ runs. **That is now false**: the capacity sweep diverges on 11 of 30
PCA-video seeds (§9.16), and it reproduces on the real Qwen residual stream at 80
writes (§9.14). The two fixes once listed here, `theta_scale=0.01` and
`normalize_output=True`, are both superseded and neither is a default.

### 9.5 The bound sits on the wrong Gram matrix (prediction, later confirmed)

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
inherit `‖h₀‖`'s growth (at the cost of the associative signal, §9.1), and halving
`θ_scale` attacks the same product `θλ` from the other side.

`stability.py` measures all of it per chunk (§7.3). **This prediction was later
confirmed by direct measurement** on Qwen2.5-VL — `‖h₀‖` turns the trajectory over
right where `θΛ` crosses 2 (§9.13).

What the title used to say — that ATLAS has no bound covering this — is only half
right, and the wrong half was load-bearing. ATLAS's *stated* bound `2/λ_max(KKᵀ)` is
indeed the curvature of a **matrix** memory and does not cover a deep one. But ATLAS's
*optimiser* bounds the update itself, independently of any curvature — and we had
disabled exactly that by rescaling Newton-Schulz back to `‖G‖`. §9.14.

### 9.6 Initialising the momentum gate matches a whole cascade — and it replicates

On `memory_gain` (§7.1), `T=64`, `K=24`, full BPTT, batch 8, 2500 steps, PCA `k=128`.
`nocas_eta10` is `nocas` plus one changed bias: `--eta-init 0.9`, i.e. a momentum
horizon `1/(1-η)` of 10 tokens instead of the default 2.1. Permutation tests, paired
by seed. **0 NaN in either sweep** — the PCA representation appears to suppress §9.4.

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
   This generalises the `alpha_init`/SANE lesson (§9.1): gains credited to
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

### 9.7 `normalize_hidden`: free on plain SANE, expensive on `eta_init`

> **Superseded as a fix by §9.14**: bounding `‖h₀‖` is a 54× delay (80 → 4 328
> writes), still exponential at ×1.082/chunk, because it bounds the last layer's
> curvature but not the residual the gradient is still proportional to. Bounding the
> *update* is total. Kept for its cost measurement, which still stands and is the
> reason not to add it on top of `unit`.

§9.5 says the unbounded quantity is `‖h₀‖`, so bound it: `normalize_hidden`
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
   rest, and §9.5 puts the resting point already past the predicted threshold there.
   The theory said `eta10` is where the fix would bite hardest, and it is.

The controls reproduced §9.6 to four decimals, which also confirms the flag is inert
when off.

**What this does not yet show is that it prevents the divergence it was built for.**
PCA space gives 0 NaN everywhere, so the sweep above had nothing to prevent. With the
synthetic harness retired, that question is closed by abandonment rather than
answered — and `normalize_hidden` has since found a different job (§9.11).

### 9.8 2500 steps was not an equilibrium — 40 000 is

`--steps 2500` entered the repo as an unjustified default and every number in §9.3-8.7
was measured there. One trajectory (`nocas` η=0.9, snapshots every 5000 steps, **n=1
seed**) says what that cost:

| steps | `mem_effect` | L1 (2nd copy) | margin | PR | train/val gap |
|---|---|---|---|---|---|
| 5 000 | 0.2321 | 0.4719 | 11.46 | 54.3 | −0.0037 |
| 10 000 | 0.3557 | 0.3512 | 5.42 | 50.1 | +0.0024 |
| 20 000 | 0.4719 | 0.2418 | 3.42 | 49.9 | +0.0068 |
| 40 000 | **0.5331** | **0.1753** | **1.44** | 56.1 | +0.0089 |

A saturating exponential fits to RMS 0.005: **asymptote 0.530, `τ_train` ≈ 8 900
steps.** So 40 000 steps is converged (100.7 % of the asymptote), 10 000 reaches only
**67 %**, and 2500 about 30 %. Read every number in §9.3-8.7 as roughly a third of its
converged value. The *orderings* held all along the trajectory; the magnitudes did not.

**No overfitting at 53 epochs.** The train/val gap on `mem_effect` is +0.0089, i.e.
1.7 % of the signal. Criterion "stop before you memorise" never bites on this corpus.

**The margin falls as `steps^-1.00`** (an exact power law) while survival does not
move at all — see §9.9. It is not a stopping criterion.

### 9.9 Divergence and collapse are different failures, and the margin certifies neither

Two distinct ways the memory stops working, and only one shows up in a finiteness
check:

| | divergence | collapse |
|---|---|---|
| `‖S‖` | grows geometrically → NaN | bounded |
| PR | — | falls to ~5 from ~55 |
| when | within 200-400 tokens | over 10 000s of tokens |

Long-stream probes (`long_run`, `loop` = one clip repeated, `cycle` = a rotating set,
the harsher one). Scalar arms up to η=0.95 hold 64 000 tokens and **collapse**; η≥0.98
**diverge** before 400 tokens. The memory-off control holds PR at 54.27 exactly over
300 repetitions, so the instrument is clean.

**`τ*`, the critical accumulation horizon** (`τ_acc = 1/(1−η)`), is bracketed between
**20 and 50 tokens** on `loop` and between **3 and 10** on `cycle` — it depends on the
input statistics, so a single number would be wrong. The sweep that narrows it runs
from `scripts/tau_chain.sh`.

**The margin was tested blind and refused.** Nine predictions written before the runs,
**7 wrong**: two arms with margin < 8 held 64 000 tokens, one with margin 5.81 died at
384. Then a monotone control: along the §9.8 trajectory the margin fell **×8** (11.46 →
1.44) and survival did not move a token (300/300 finite at every step count). Use it as
a within-family diagnostic, never as a stability certificate.

**What does discriminate**: `d log‖S‖/dt` on a short probe — ~0 for survivors,
+0.013 to +0.065 for divergent arms. Two orders of magnitude, no overlap, ~27 s for
1920 tokens.

**Training longer does not fix collapse.** Across 5 000 → 40 000 steps the final PR is
4-6 regardless. What training buys is that the surviving directions stay useful:
`cos_ret` at the end of the stream rises 0.29 → 0.64 and the tail penalty halves. Rank
collapse and functional collapse are not the same thing.

### 9.10 Past distance 0 the memory is a net cost, and it is interference not decay

Scored with the memory-off control (`mem_effect = L1(memory off) − L1(repeat)`), at
10 000 steps, distractors between the two presentations of V:

| gap (latents) | +0 | +64 | +128 | +2048 |
|---|---|---|---|---|
| `mem_eta0.9` | **+0.3475** | −0.2895 | −0.3940 | −0.5977 |

The memory-off L1 is **0.7068 at every gap, exactly constant** — so the degradation is
not the model losing the thread. At gap 0, i.e. exactly the training length with no
drift possible, the *cross* condition already costs +26 % over memory-off. The
dominant term is **content interference**, not length-driven state decay.

Two controls worth knowing before designing anything here. A **full-attention baseline**
with `seg ≥ 2T` scores `memory_gain` between −9e-5 and +1e-4 with random sign, and
trains in 21 s against ~6 h for a memory arm. And **8 attention heads change nothing**
(`mem_effect` 0.0832 vs 0.0849 at 1 head) while *lowering* PR (56.0 vs 63.7) — the
head count is not the fix for anything measured here.

### 9.11 The cascade sits past the bound because the outer loop puts it there

`B` is doubly stochastic — spectral radius exactly 1 — so the cascade **cannot** grow
geometrically on its own. Yet every cascade arm dies within 200-400 tokens with
`d log‖S‖/dt` between +0.013 and +0.065. An audit of the gates the outer loop *chose*
(58 frozen artefacts, `margin.py`) locates the cause:

| separator | cascade (real leak, n=18) | scalar (n=40) | Cohen d |
|---|---|---|---|
| `θ_max` | 0.0630 ± 0.0305 | 0.0165 ± 0.0132 | **1.98** |
| `‖h₀‖ max` | 2.185 ± 0.560 | 1.194 ± 0.480 | **1.90** |
| `α` mean | 0.00040 | 0.00120 | −1.20 |
| `θ·λ_hidden` | 3.25 | 0.144 | 1.16 |

`cas_none` and `cas_r0.03` are excluded as pure accumulators (r = 0, and r < 0.1 where
the surprise leak does nothing); excluding them *strengthens* the separation.

**Raising the leak rate cannot fix this, because the outer loop compensates it.**
`cas_r1.0` leaks 33× harder than `cas_r0.03` and ends with 2.3× the `θ` and 4.5× less
margin. That is why sweeping r ∈ {0, 0.03, 0.1, 0.3, 1.0} is flat in survival: every
increment of leak is repaid in step size.

**Working hypothesis, under test.** In the `chain` topology the gradient enters
reservoir 0 only, which then sheds `beta1·g⁻¹ = 0.25` of its mass per token to the
deeper ones, and every readout except `sum` stays at single-filter scale. The outer
loop would be paying for that dilution in `θ`. The arms that change the *structure* and
leave `θ`/`α` freely learned (`readout=sum`, `beta1=0.1`, `normalize_hidden`) are
queued in phase 3 of `tau_chain.sh`; predictions and the falsifier were registered
before launch in `results/predictions_cascade_dilution.md`.

**`normalize_hidden`'s second life.** It was built to stop a divergence PCA space never
produced (§9.7), and on the scalar arm it costs 16 % of `mem_effect` for nothing. But
it bounds `‖h₀‖` by construction, and `‖h₀‖` is where the cascade's excess actually
sits (d = 1.90 above) — the one place it was never tested. That is arm `casE_nh`.

Note the two divergence mechanisms are **not** the same: scalar arms diverge by η
accumulation with *small* θ; cascade arms by θ compensation with margin < 1. They do
not belong on one axis.

### 9.12 Seed variance is larger than most tables here admit

Measured across 6 seeds per arm: `cas_none` spans a factor **24** on `θ_max`. Several
headline tables in this README are n=1 or n=3, including the §9.8 trajectory. Treat
any single-seed number as a direction, not a magnitude, and prefer spending budget on
seeds over arms.

### 9.13 The divergence is a curvature loop, and ATLAS already had the fix

§9.5 predicted the mechanism from theory and named a threshold. It is now **measured
directly**, on a different model (Qwen2.5-VL-3B), a different stream (real EgoLife
video through the frozen host), with `graft/trace_stability.py`.

`theta` never moves — 0.016 to 0.033 for the whole run, right up to the chunk that
goes NaN. What explodes is the **curvature**:

| chunk (m=8) | 0 | 3 | 6 | 10 |
|---|---|---|---|---|
| `‖h₀‖` | 0.29 | 0.93 | 4.0 | 2.6e12 |
| `λ ≈ ‖h₀‖²` | 2.3 | 26 | 301 | 1.2e26 |
| `θ·κ·λ` | 0.04 | 0.84 | 8.9 | 3.8e24 |

> `W₀` grows → `h₀ = σ(W₀k + b₀)` grows → `λ` grows → the effective step `θκλ` grows
> → `W₀` grows more.

`normalize_qk` pins `‖k‖ = 1`, the MLP's *input*; **nothing bounds its hidden layer**.
§9.5's predicted threshold (`θΛ > 2`, i.e. `‖h₀‖ ≈ 2`) is where the trajectory turns
over — a prediction registered before the measurement, confirmed on a model it was
never fitted to.

**Chunk size sets the entry point, not the outcome.** `κ → m(m+1)/2` as `η → 1`
(measured 35.2 at `m=8, η=0.995`, formula 36), so `θλ` at chunk 0 scales like `m²`:
0.037 at `m=8`, 0.32 at 16, 1.02 at 64. All three still die after 5-11 *chunk steps*.

### 9.14 Four magnitude conventions for the inner step

`newton_schulz5` restored `‖G‖_F` — so orthogonalisation was purely **directional**
and bounded nothing. "We already use Muon, so the step is bounded" was false for this
codebase. ATLAS Eq 32 / Table 1 do **not** rescale:

```
S_t = θ_t S_{t−1} − ∇ℓ            M_t = α_t M_{t−1} − η_t · NS-5(S_t)
```

The learning rate multiplies an *already-orthogonalised* matrix whose singular values
are ≈1, so `‖ΔM‖₂ ≈ η_t` **independently of the gradient**. Neither paper states this
as a stability result — ATLAS motivates Muon as second-order information, and neither
paper contains the words "stability", "divergence" or "clipping" at all. Structurally
it is the only hard bound either has on the inner loop.

`orthogonalize_momentum` now takes four values. `True` still means `rescale`,
bit-for-bit, so every frozen artefact reproduces.

| | `‖ΔM‖` | surprise hierarchy |
|---|---|---|
| `rescale` | `θ‖∇ℓ‖` — scales with the state | full, unbounded |
| `unit` | `≈ θ` | only the learned gate `θ_t` |
| `clip=c` | `min(‖S‖, c)` | flat above `c` |
| `squash=c` | `(2c/π)·arctan(π‖S‖/2c)` | **monotone everywhere** |

`squash` is Josep's. `φ_c` is monotone at every scale, bounded by `c` (so `‖M‖` has a
fixed point at **`c/α`** — which is how `c` should be chosen), and `≈ x` below `c` so
normal operation is unchanged. arctan rather than tanh on purpose: polynomial rather
than exponential saturation keeps ~1e6× more discriminability at `10c`, which is
exactly where the surprising tokens are. It is also a **homotopy between the two
papers**: `c → ∞` is Titans (unbounded), `c → 0` is ATLAS (magnitude discarded).

**The wall, on a continuous stream** — 40 consecutive EgoLife clips × 64 frames =
104 800 fresh writes, ~20 min of egocentric video, `d_mem=128`, layer 24, chunk 8:

| | survived | `‖W₁‖` start → mid → end | growth |
|---|---|---|---|
| `rescale` (control) | **80** | 6.5 → 48 → 1.4e10 | explodes |
| `rescale` + `normalize_hidden` | **4 328** | 6.5 → 1.7e11 → 2.7e20 | ×1.082/chunk |
| **`unit`** | **104 800, no break** | 6.5 → 41.0 → **42.7** | **×1.00001 = bounded** |
| `unit` + `normalize_hidden` | 104 800, no break | 6.6 → 44.0 → 48.4 | ×1.00001 |
| `unit` + cascade | 26 200, no break | 6.6 → 47.6 → 51.3 | ×1.00005 |
| `clip=1.0` + cascade | 26 200, no break | 6.5 → 153 → 161 | bounded but `θλ = 119` |

**≥1310× the control, and no wall was found** — the probe stops where the video stops.
`‖h₀‖` settles at 1.96 *by itself*: once the step is decoupled from the state, nothing
feeds the curvature, so `normalize_hidden` becomes redundant (and it separately costs
57 % of `eta_init`'s gain, §9.7). On its own it is only a 54× delay.

**`clip` at `c=1` failed twice, both times for the same reason.** On the world model
it was inert (seeds 1-4 bit-identical to `rescale`); on the cascade path it saturates
(`‖S‖ ≈ 9e3` against `c = 1`), which discards `θ_t` too. `c` must be chosen from
`c/α`, not by feel — the natural deposit size under SANE is `θ·√rank ≈ 0.24`.

**Two methodological corrections worth keeping.** Tiling one clip (`--repeat`) makes
the probe *easier*, not harder: at period `P` the memory learns the loop, the
associative loss collapses, and nothing drives the state. Measured cost of that error:
an extrapolated 9 000 writes for `rescale + normalize_hidden` against a real 4 328.
Use `--videos`. And a probe must survive what it measures — a 40-minute trace died on
`linalg.eigh: failed to converge` and took every row with it, because rows were only
returned at the end. Both fixed, with tests.

### 9.15 Reach is set by α alone, in closed form

Under a bounded step every deposit into `M` has the same size, so `θ` **cancels** out
of the trace-to-interference ratio (verified numerically over six decades of `θ`).
With quasi-orthogonal deposits — which is what `NS₅` produces — the trace laid down
`d` writes ago reads out at

```
SNR(d) = √P · (1−α)^d · √(α(2−α))          P = memory weight entries
```

Maximising over `α` gives a closed form, checked against simulation:

```
α* = 1 − √(d/(1+d)) ≈ 1/(2d)     half-life = 1.39·d     SNR* ≈ 0.61·√(P/d)
```

**The ratio 1.39 is constant** — the half-life should be ~1.4× the recall distance,
not a fraction of the sequence. `alpha_init = 0.002` is optimal for `d ≈ 250` writes;
EgoLifeQA's median question at `write_stride 9` is ~50 000, which wants `α ≈ 1e-5`.

Two consequences:

* **`SNR* ∝ √(P/d)`**: holding a distance `d` at constant quality needs `P ∝ d`,
  linear in the horizon. That is the GPU ask in closed form, and it is the first
  version of it that is defensible — see §9.16 for why the previous one was not.
* The **envelope** of `max_α SNR(d,α)` is a power law `d^{−1/2}`, but a *fixed* `α`
  touches it at one point only and falls off exponentially either side. Tuning `α` to
  a horizon buys one point; **Benna-Fusi's cascade is a construction that sits on the
  whole envelope at once** — and BF themselves note `t^{−1/2}` is the critical case
  where the variance just starts to diverge, i.e. the envelope is the best achievable.
  This reframes the cascade ablation: not "does it remember better" (it cannot beat
  `0.61√(P/d)`) but "does it dispense with choosing `d` in advance". Our measured
  readout decays `t^{−0.07}`, so we are far from it.

At `α = 1e-5` the gate is saturated (`σ⁻¹ = −11.5`, gradient `≈ 1e-5`), so at long
horizons `α` stops being learned and becomes a hyperparameter. Say so when reporting.

### 9.16 More memory bought less reach — because it diverged sooner

`worldmodel/capacity_reach.py` turns retention curves into reach in minutes of video
(1 latent = 0.469 s, measured from the cache, not assumed). Swept h32→h512, 6 seeds
each, on 30 pre-existing frozen artefacts — **zero training cost**:

| | h32 | h64 | h128 | h256 | h512 |
|---|---|---|---|---|---|
| reach (min, median) | **3.8** | 1.7 | 1.0 | 0.7 | **0.5** |
| gain@0 (median) | 0.088 | 0.196 | 0.223 | 0.241 | 0.255 |
| healthy seeds /6 | 1 | 0 | 2 | 2 | 2 |

**16× more memory, 7.6× less reach**, while storage at distance 0 improves
monotonically. This is not a capacity trade-off: widening raises `‖h₀‖`, which raises
`λ ≈ ‖h₀‖²`, which raises the effective step (§9.13). It also refutes §9.4's claim
that the divergence never appears in PCA space — 11 of these 30 seeds diverge.

So **the GPU ask cannot be written from this table**: today a bigger card buys a
memory that dies sooner. The sweep is worth redoing under a bounded step, where the
`√(P/d)` law of §9.15 should finally point the right way.

Caveat for any write-up: these artefacts ran ~2500 steps, so every magnitude is
roughly half its converged value (§9.8). Absolute reaches are a lower bound.

## 10. Data/parameter proportions: an accepted deviation

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
honest indicator is the measured generalisation gap, not the ratio — and it has since
been measured directly along a 40 000-step trajectory (§9.8): +0.9 % at 40 000 steps,
i.e. **no overfitting at 53 epochs**, which settles the question for this corpus.

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

## 11. Running experiments

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

```bash
# --- the graft ---------------------------------------------------------------
# 1. fit the memory's input basis on the host's own layer-24 activations
python -m titans.graft.fit_basis --videos '<egolife>/**/*.mp4' \
    --k 128 --num-videos 24 --max-side 256 --out cache/qwen_l24_pca128.pt

# 2. prove the graft is still the base model, and price it (identity + VRAM)
python -m titans.graft.check_qwen --basis cache/qwen_l24_pca128.pt \
    --video clip.mp4 --frames 16,32 --write-stride 1,8 --guard-gb 14

# 3. find where the inner loop breaks, on a CONTINUOUS stream (never --repeat)
python -m titans.graft.trace_stability --videos '<egolife>/**/*.mp4' \
    --num-videos 40 --basis cache/qwen_l24_pca128.pt --frames 64 --chunk 8 \
    --override orthogonalize_momentum=unit --print-every 50 --out trace.json
```

New axes on `echo.py`, all deltas on `sane_ns` (§9.14, §9.15):

| flag | |
|---|---|
| `--muon {rescale,unit,clip,squash}` | which magnitude convention bounds the inner step |
| `--muon-scale c` | the scale for `clip`/`squash`; pick it from `‖M‖ → c/α` |
| `--alpha-init a` | the reach knob. For a recall distance `d` writes, `a ≈ 1/(2d)` |

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

# cut to shorter sequences so [V;V] fits under FULL BPTT (see §7.1)
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

**Multi-phase studies run from one cron entry.** `scripts/tau_chain.sh` is the live
example: each phase is gated on the previous one's jobs being *gone* (`pgrep`), not on
its queue file being empty — the in-flight cells of a drained queue still hold cards,
and starting the next phase then is what caused the OOMs of 2026-07-29. A phase that
builds its own queue from earlier results (`build_traj_queue.py`) must refuse to emit
anything when the input does not support it, rather than queue a plausible-looking
sweep.

Arms must share the seed grid: the inner-loop divergence is deterministic per seed, so
a failure rate only means something against a control at the *same* seeds. Raise
`WORKERS`
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

## 12. Not built, and retired

**Retired 2026-08-14: the synthetic benchmark.** `src/titans/membench/` (MQAR,
associative recall, copy, needle-in-a-haystack) and its tests are deleted. It served
its purpose — it is where the SANE gate set was found (§9.1) and where the in-forward
divergence was first characterised (§9.4) — but the project is judged on real video
latents and, eventually, on EgoLifeQA. Two consequences to know: the `normalize_hidden`
divergence question is closed by abandonment rather than answered (§9.7), and any
sub-2500-step number quoted from that era is on a different task. **We do not build
replacement benchmarks.**

Never built:

* The **MAG** and **MAL** variants (only MAC is implemented).
* **Per-channel** memory gates — `α`/`η`/`θ` are one scalar per token, via a
  `Linear(dim, 1)` head.
* A **true parallel scan** for the momentum recurrence (`_scan_chunked` is a Python
  loop across chunks).
* The **tensor-sketch feature map** that would restore `φ`'s cross terms.
* A **language head** — the world model predicts latents, not text. (The *graft* is
  built, §6; what is missing is training it and an answer-token loss.)
* A **bound on the first layer's curvature**. `normalize_hidden` (§9.7) bounds the
  last layer's block by construction; the first layer's
  `≤ 2‖k‖²‖W₁ diag σ′‖²` still depends on `‖W₁‖`, which nothing constrains. Largely
  moot under §9.14: bounding the *update* bounds every block at once, which is why
  `‖h₀‖` settles by itself there.
* **Open problems from the theory note**: lifting the linearisation to genuinely
  per-token `(H_i, v_i)`; a switched-system certificate (common Lyapunov `P`, or a
  bound on the joint spectral radius) for the data-dependent gates; and the
  admissible region in the `(α, η)` plane. Two of its items are now closed: the
  predicted threshold is measured (§9.13), and `κ` **does not diverge as `η → 1`** at
  finite `m` — it saturates at `m(m+1)/2`, because a gradient injected at step `i` can
  only be re-applied on the `m−i+1` steps left in the chunk, so the chunk truncates
  the momentum's horizon and the triangular number falls out by counting. Confirmed
  three times at `m=8`: 14.6 at `η=0.534`, 28.6 at `η=0.9` (naive form says 80),
  35.2 at `η=0.995` against the bound 36. So "long retention and long accumulation in
  direct tension" does not hold as stated, and **`m` is the high-leverage knob**, not
  `η` — `κ` grows at least linearly in `m` in both regimes while `η` saturates.
