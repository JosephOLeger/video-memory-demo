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

The repo now also implements **Atlas** (*Learning to Optimally Memorize the
Context at Test Time*, [arXiv:2505.23735](https://arxiv.org/abs/2505.23735), same
Google Research team), of which Titans is the degenerate case: window `c=1`,
identity feature map, first-order inner optimiser. All three of Atlas's
ingredients are in — the Omega sliding-window rule, the polynomial feature map
`φ`, and the Muon/Newton-Schulz inner optimiser — so Eq 57-58 is complete. Every
one is **opt-in and defaults off**, so the Titans paths stay bit-identical and
earlier results remain comparable. See *Assembling Atlas* below, and
`docs/atlas_vs_titans.md` for the component-by-component status.

Read the *Memory collapse* section before trusting any memory-health claim: the
online memory degenerates to a rank-1, query-independent function within one
segment, this is now well characterised and repairable, and repairing it has so
far not improved any downstream task.

## Running the tests

```bash
pip install -e '.[test]'   # or: pip install torch pytest
pytest                      # 100 tests
```

Beyond the two paths agreeing numerically (float64, `atol/rtol` down to `1e-9`)
across several chunk sizes and a toy memorisation check, the suite guards the
properties that turned out to be easy to break: MAC's causality across segment
boundaries, `grad-of-sum == sum-of-grads` for the Omega window (with and without
`φ`), and — after four options were found to be silently inert once assembled —
that each opt-in flag actually *changes the computation* rather than merely being
accepted. That last class is the one worth extending when adding a flag.

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
   each projection. Note the temperature subtlety: normalising Q/K bounds the
   logits to `[−1, 1]`, and `F.scaled_dot_product_attention` then shrinks them
   again by its default `1/√head_dim`, leaving the softmax near-uniform.
   `learned_qk_scale=True` replaces that with a learned per-head temperature
   (default **off**, so the previous behaviour is bit-identical). See the
   collapse section for the measurements.
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

## Metaplastic momentum cascade (`metaplastic.py`) — opt-in momentum for the neural memory

`src/titans/metaplastic.py` explores replacing Titans' single momentum scalar
`S_t` (one decay `η_t` controls both reactivity *and* retention — the
plasticity-stability tradeoff) by a **K-reservoir cascade** inspired by
biological synaptic metaplasticity (Fusi/Abbott/Drew 2005; Benna & Fusi 2016).
Each scalar momentum entry becomes a length-`K` vector `u=(u¹,…,uᴷ)`; only `u¹`
sees the gradient, and consecutive reservoirs exchange mass by diffusion:
`u_t = B·u_{t-1} + e₁·∇ℓ_t` with `B` tridiagonal (`b_k = β₁·g^-(k-1)`).
`read(u)=u¹` is the "visible" momentum fed into `M_t=(1−α_t)M_{t-1}+read(u_t)`. The
recurrence is a `cascade_step`/`cascade_scan` pure-function pair (sequential Python
scan for now — consistent with the existing single-momentum scan; the split is the
seam where a Mamba-style parallel associative scan could drop in). **Status: wired
into `NeuralMemory`/`TitansMACBlock` as an opt-in** — `use_cascade_momentum=True` /
`memory_use_cascade=True`, **default off** so it stays bit-identical to scalar Titans
when unused. When on, `η_t` becomes a per-token leak on the *whole* cascade state
(`U_t = B_t·(η_t·U_{t-1}) + e₁·(−θ_t∇ℓ)`), so `K=1` collapses exactly onto the scalar
rule. It is exercised today by the `membench` sweep (`coupled`/`uncoupled` configs)
and covered by `tests/test_cascade_memory.py`; it is **not yet activated in the video
world model** (`worldmodel/` never sets the flag).

Two coupling modes: a **fixed geometric** `B` (mode 1), and a **selective** `B_t`
(mode 2) where a learned `Linear(dim, K−1)`+sigmoid head gates the couplings *per
edge*. A single global scalar gate would give `B_t=I+s_t·D` for a fixed `D` — all
such `B_t` commute and reduce to a fixed bank of `K` exponential filters, no more
expressive than the fixed cascade; **per-edge** gating reshapes the diffusion so
the `B_t` genuinely don't commute (event-triggered consolidation). This is a
deliberate deviation from a literal "per-token scalar", documented in the module.

**Stability of `B` (checked, not assumed).** `B` is symmetric and row-stochastic
⇒ doubly stochastic ⇒ all eigenvalues in `(−1, 1]` with the top one exactly `1`
(conserved diffusion mass). For the defaults `K=6, β₁=0.5, g=2` the spectrum is
`[0.40, 0.77, 0.90, 0.96, 0.99, 1.00]` (spectral radius 1, all positive — far
from the `−1` boundary). So, unlike the geometric per-segment blow-up fixed in
`memory.py`, there is **no** eigenvalue > 1; the only caveat is the eigenvalue
exactly 1 — `u¹` has no self-decay, so under sustained same-sign gradients the
conserved total grows at most *linearly* in sequence length (the memory's own
`(1−α_t)` forget gate would supply the missing leak downstream).

**Diagnostics** (`tests/test_metaplastic.py`, run with `pytest -s`):

* *Exact regression* — the hand-worked `K=2, β=0.3`, unit gradient at `t=1`:
  `u¹ = 1, 0.70, 0.58, 0.532, 0.5128 → 0.5` reproduced to `atol=1e-6` (vs a plain
  θ=0.7 exponential that keeps decaying `1, 0.7, 0.49, 0.343, 0.24`).
* *Impulse response, `K=6`* vs a single exponential matched to the same *initial*
  decay (`λ = 1−b₂ = 0.75`). The cascade retains far more of a one-shot memory:

  | t   | cascade `u¹` | exp `0.75ᵗ` | ratio     |
  |-----|--------------|-------------|-----------|
  | 10  | 0.389        | 5.6e-2      | 6.9×      |
  | 100 | 0.208        | 3.2e-13     | 6.5e11×   |
  | 500 | 0.167        | 3.4e-63     | 4.9e61×   |

  `u¹` relaxes toward the equilibrium `1/K ≈ 0.167` (mass redistribution), never
  to 0. Over the intermediate window `t∈[2,30)` the *excess above equilibrium*
  `u¹−1/K` is **power-law-like**: log-log slope ≈ −0.52 with `R²=0.994`, vs the
  matched exponential's `R²=0.895` (visibly curved on log-log). For finite `K`
  the very-long tail eventually crosses over to the single slowest eigenvalue
  (0.989), so it is power-law over the window set by the number of reservoirs,
  then a slow exponential — the Benna-Fusi finite-cascade picture.

## Memory collapse: rank-1 degeneracy, root cause, and mitigations

Motivating question: does the online-updated `M` actually *discriminate by
query* (`M(k)` genuinely depends on `k`), or does long inner-loop threading
degrade it into a near-constant function? Empirically, for plain scalar
momentum: **yes, it degenerates.** Probing a trained `M`'s retrieval on
several distinct queries (including queries far outside the training
distribution) gives pairwise cosine similarity `≈ 1.0000` after only a few
segments — `M(k)` stops depending on `k` at all.

### Diagnosis

* **Symptom, precisely localised.** `mlp_forward`'s final layer is
  `h = SiLU(k·w0ᵀ+b0)·w1ᵀ + b1`, then RMS-normalised. Splitting `h` into its
  `k`-dependent term (`signal`) and its additive term (`b1`), the ratio
  `‖b1‖ / ‖signal‖` grows **geometrically across segments** (observed up to
  `~10⁴×` within 8 segments) for the plain scalar recurrence — but `b1` is not
  the culprit *growing*; `w0, b0, w1` all decay toward **exactly zero** under
  `(1−α_t)` weight decay, `b1` merely decays slower (a single multiplicative
  factor per step vs. a two-stage multiplicative chain for the `k`-dependent
  path), so the ratio explodes by the numerator shrinking faster than expected,
  not the denominator growing.
* **`w1` itself collapses to (numerical) rank 1**, not just small: its
  spectrum has one dominant singular value and the rest at hardware-precision
  noise (`sv_bot ~ 1e-15`–`1e-18`), from essentially the *first* segment of
  online updating onward — confirmed with a participation-ratio diagnostic
  `PR(W) = (Σsᵢ²)² / Σsᵢ⁴` (`1.0` = pure rank-1, `dim` = flat/healthy).
* **This is an inner-loop phenomenon, not an outer-loop one.** `M₀` itself
  (the outer-trained initial weights, before any online update) is healthy
  (signal-to-bias ratio `~0.01`, i.e. signal dominates 100×). The collapse
  develops *during* the online recurrence over a single sequence, and resets
  every new sequence (state re-initialises to `M₀`).
* **Not fixed by genuine input diversity.** The same rank-1 collapse occurs
  whether the input is a smooth synthetic random walk, i.i.d. Gaussian tokens,
  or a literal constant ("black-screen") sequence — ruling out "the data just
  happens to be too correlated" as a sufficient explanation on its own.

### Root cause

The per-token gradient of `w1` is an outer product,
`∇_{w1} ℓ_t = e_t ⊗ h0_t` (error direction ⊗ hidden activation) —
**rank 1 by construction**, at *every* step, regardless of how diverse the
underlying video/data is. So the rank of `M` is decided entirely by how many
*distinct* such outer products actually survive into the momentum sum
`S_T = Σ_t θ_t (e_t ⊗ h0_t)`. Three measured effects each suppress that count;
they compound.

> **Superseded hypothesis (kept because it shaped the code).** This section
> previously blamed a self-referential feedback loop: `h_i = retrieve(state)`
> feeds `y_i = attn([persistent; h_i; segment])`, which produces the very
> `k_i, v_i` that update the memory, so the memory contaminates its own training
> signal — a power-iteration/Oja's-rule attractor. That story motivated
> `decouple_memory_update`. **It does not survive measurement**: the collapse is
> already complete after the *first* segment (`PR(w1)` goes `16.17 → 1.000`
> across one 8-token segment, before any feedback can accumulate), and severing
> the loop does not restore query-dependence (see the table below). The feedback
> loop is real, but it is not what sets the rank.

* **Attention is a smoother, and it is what writes to the memory.** The memory's
  write signal is `y_i`, an *attention output*. Measured on one segment with
  i.i.d. Gaussian input, mean pairwise cosine across the segment's tokens:
  `x` `−0.057` → `y` `+0.904` → keys `+0.864` → `h0_t` `+0.818` → `e_t` `+0.830`.
  The 8 outer products of a segment are therefore near-identical and their sum
  has `PR = 1.03` out of a possible 8. This is why the collapse is indifferent to
  the data (random walk, i.i.d., or a constant "black screen" all behave alike) —
  the correlation is manufactured by the block, not inherited from the input.
* **The cosine-attention temperature makes that smoothing near-maximal.**
  `attention.py` L2-normalises Q and K (so `q·k ∈ [−1, 1]`) and then lets
  `F.scaled_dot_product_attention` apply its default `1/√head_dim` *on top*. At
  `head_dim=16` that is `1/4`, so logits live in `[−0.25, 0.25]` and the softmax
  is nearly uniform: measured max attention weight `0.0556`, against `0.0500` for
  a perfectly uniform distribution over the same 20-token prefix. Attention is
  doing little more than averaging its window. The QK-normalisation literature
  (Henry et al. 2020; ViT-22B; Swin-V2) pairs the normalisation with a *learned*
  temperature for exactly this reason — normalised logits need scaling **up**,
  not down by `1/√d`. Sweeping the scale `0.25 → 50` moves `cos(y)` `0.90 → 0.17`
  and the momentum sum's `PR` `1.03 → 2.65`. Addressed by
  `learned_qk_scale=True` (below).
* **The momentum recurrence is a geometric sum, so few gradients survive it.**
  `S_t = η_t S_{t−1} − θ_t g_t` weights a segment's gradients by
  `(…, η², η, 1)`. With `η_t ≈ 0.515` the normalised weights over an 8-token
  segment are `0.005, 0.009, 0.018, 0.034, 0.067, 0.129, 0.251, 0.487` — the last
  token alone carries half the sum, and the **effective number of contributing
  gradients is 3.09 out of 8** (participation ratio of the weights). Since each
  is rank 1, this caps the achievable rank near 3 no matter how diverse the data
  is. The momentum's effective horizon is `1/(1−η) ≈ 2` tokens.
* **The momentum also swamps the initialisation.** After one segment
  `‖S‖ = 41` against `‖w1(M₀)‖ = 3.3` — a factor 12 — so `M₁ ≈ S` and the
  outer-trained weights (healthy, `PR = 16.17`) are simply overwritten. This is
  why `M₀`'s health is irrelevant to what the recurrence converges to.
* **Consequence, confirmed experimentally.** A late-sequence "regime shift"
  (second half of a sequence drawn from a content direction near-orthogonal
  to the first half) does **not** measurably redirect the already-locked
  dominant direction — new content is absorbed as noise around the existing
  attractor rather than treated as an independent signal.

**Measurement caveat.** The gate values and per-segment traces above were
measured on an *untrained* block (`dim=32`, `segment_size=8`, 16 segments), so
`α ≈ 0.46`, `η ≈ 0.515`, `θ ≈ 0.68` are the head initialisations
(`sigmoid(0)`, `softplus(0)`), not learned values. The collapse itself reproduces
on trained models too (the original probes), but whether training moves `η`
toward a longer horizon has **not** been checked — and if it does not, the short
horizon is structural rather than learned. That check is the cheapest open item
here.

### Mitigations tried, and what they actually fix

| approach | flag(s) | fixes direction lock-in? | fixes magnitude→0 decay? | cost | effect on recall (MQAR) |
|---|---|---|---|---|---|
| freeze `M`'s bias online (`b0,b1` stay at their outer-trained value) | `NeuralMemory(update_bias=False)` | no (alone) | no (alone) | free | — |
| + Newton-Schulz orthogonalisation of the scalar momentum before it updates `w0/w1` | `NeuralMemory(orthogonalize_momentum=True)` (`memory.py::newton_schulz5`, the same quintic iteration Muon/ATLAS's inner optimiser uses) | **yes**, combined with bias-freeze (retrieval diversity cosine `1.00 → −0.03`) | yes (combined) | **~2.8× slower** per step | **no measurable change** (`0.588±0.071` vs `0.575±0.086`, within noise) |
| metaplastic-cascade momentum with no leak | `TitansMACBlock(memory_use_cascade=True, cascade_leak="none", cascade_readout="mean")` | **no** — direction still drifts away from its start at a slow, roughly constant rate, unaffected by a regime shift | **yes** — state provably reaches a stable nonzero fixed point (bit-identical across segments in a constant-latent probe) | cheap | no measurable change |
| ATLAS Omega rule, sliding window `c>1` | `TitansMACBlock(memory_window_size=c)` | no | no | cheap | untested here (see `docs/atlas_vs_titans.md`) — rank diagnostic shows no improvement even with i.i.d. per-token content, because `retrieve`'s contamination taints every token *inside* the window too, not just across steps |
| decouple the update path from `h_i` | `TitansMACBlock(decouple_memory_update=True)` | **partly** — stops the *drift* lock (`cos`-to-segment-0 falls from `1.0` to ~`0`), but `M` stays a near-constant function of the query (retrieval-diversity cosine stays `1.0000`) | **no** — `w1` still decays to `1.8e−11` by segment 16 | one extra `attn()` call per segment (same weights, no new params) | untested |
| **learned cosine-attention temperature** | `TitansMACBlock(learned_qk_scale=True)` | **no, alone** (`PR` still `1.000`) — but it is the largest *contributor* in combination: added to NS+bias-freeze it takes `PR` from `3.8` to `5.4` (peaks `8.1`), the best measured | no, alone | one parameter per head | untested end-to-end |

Two rows deserve care, because both were previously overstated:

* **`decouple_memory_update` is not the mechanism-targeted fix it was billed as.**
  It severs the `retrieve → attend → update` path, and that genuinely stops the
  dominant direction from being frozen at its segment-0 value. But `M` remains a
  constant function of `q` (the metric that matters for retrieval), and it does
  not beat what the temperature achieves at lower cost and without deviating from
  Eq 21-25. **The coupled, paper-faithful path is the recommended default.**
* **The combination previously flagged as "the natural next step" does not work.**
  `decouple_memory_update=True` + `cascade_leak="none"` was predicted to fix both
  pathologies since each addresses one. Measured over 16 segments: magnitude is
  indeed stabilised (`‖w1‖ ≈ 55`, flat), but direction is *not* — `PR = 1.031`,
  retrieval cosine `1.0000`. The two fixes do not compose.

**What actually restores internal health.** The two ingredients repair
*different* things and neither suffices alone (measured over 16 segments, coupled
path):

| config | `PR(w1)` | `‖w1‖` | retrieval cos |
|---|---|---|---|
| baseline | 1.000 | 2.8e−08 | 1.0000 |
| Newton-Schulz alone | 1.888 | 3.5e−05 | **1.0000** (still constant) |
| bias-freeze alone | **1.000** (still rank-1) | 1.2e+00 | −0.0714 |
| NS + bias-freeze | 3.817 | 7.6e−01 | −0.0771 |
| **NS + bias-freeze + temperature** | **5.395** | 4.8e+01 | 0.1756 |

Newton-Schulz raises the rank but leaves retrieval query-independent;
bias-freeze fixes retrieval and magnitude but leaves the matrix rank-1. All
three together are the best configuration found. Notably, none of the three is
an architectural deviation: the temperature corrects a `1/√d` applied on top of
already-normalised Q/K, Newton-Schulz *is* ATLAS Eq 57-58, and freezing the bias
moves *toward* the papers, whose `M_t` recurrence has no bias term at all.

### The health fixes do not pay off downstream — and the reason is visible

Three independent downstream measures, all null or negative:

| measure | without fix | with NS + bias-freeze | cost |
|---|---|---|---|
| MQAR, 8 pairs, 6000 steps | `0.5875 ± 0.0710` | `0.5750 ± 0.0864` | 982 s → 2761 s |
| needle, 6000 steps | `0.0656 ± 0.0381` (chance `0.0625`) | `0.0656 ± 0.0381` | 2907 s → 8502 s |
| world-model val L1 | `0.5835` | `0.6457` (**worse**; persistence `0.6678`) | ~2.8× |

`accuracy_by_distance` on the MQAR run explains the headline number and is the
most important measurement in this section (`segment_size = 8`):

```
distance  3 | 0.9967     <- inside the attention window
distance  6 | 0.9882     <- inside
distance  8 | 0.2675     <- beyond it
distance 10 | 0.2887
distance 13 | 0.2674
distance 15 | 0.2553
```

The model solves ~99% of what attention can see and drops to ~26% (chance
`0.062`) the moment the answer requires the memory. The much-discussed "56%
plateau" is just the mixture of those two regimes — **not** a memory-capacity
ceiling. So the collapse programme has been optimising an organ that has not
been shown to sit on the critical path.

**Net picture.** Direction lock-in and magnitude decay are two separate
pathologies, both real, both reproducible, and both now fixable (NS +
bias-freeze, best with the temperature). What is *not* established is that
either one limits any task measured so far.

### Is the memory on the critical path at all? Yes — measured

The decisive test: train **one** model, then evaluate the *same weights* twice —
once threaded, once with `reset_memory_each_segment=True` so nothing crosses a
segment boundary. Three seeds, MQAR 8 pairs, `segment_size=8`:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| out-of-window, threaded | 0.2705 | 0.2735 | 0.2445 |
| out-of-window, **reset** | **0.0499** | **0.0435** | **0.0649** |
| in-window (dist 3), threaded | 0.9967 | 0.9902 | 0.5928 |
| in-window (dist 3), reset | 0.6287 | 0.0358 | 0.1531 |

Chance is 0.0625. **Severing the carry drops out-of-window accuracy to chance on
all three seeds** — the threaded memory really is what carries the long-range
signal. So the memory is load-bearing; it is simply weak (0.26 where ATLAS
reports state-of-the-art on MQAR).

**The control fails, so the attribution is not clean.** In-window accuracy was
supposed to stay ~0.99 (attention alone can solve those) and instead collapses
too, catastrophically on seed 1. The cause is a design flaw in the ablation, not
in the model: `reset_memory_each_segment` also changes `h_i`, the historical
retrieval fed *into* attention, so from segment 1 onward the whole computation
changes rather than just the long-range pathway. The direction of the result is
solid; the magnitude is not attributable. A clean version needs a
parameter-matched control **trained** with the carry severed, so it co-adapts.

**Reading this against ATLAS.** ATLAS reports strong MQAR ("best performance per
memory size compared to state-of-the-art models such as DeltaNet") and attributes
it to polynomial kernels and deep memory. So the right reading of our null
results is *not* "memory does not help" — it is **"we fixed the wrong thing"**.
The collapse was a real defect, reproducibly repaired, but it was not the binding
constraint. ATLAS names the constraint as capacity, and names the components that
address it — `φ` and deep memory — which is what the `features.py` work above
builds.

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

The **real V-JEPA-2 chain** is wired and verified end to end against actual video
(EgoLife egocentric clips, decoded on the fly). Latents are precomputed once with
the frozen encoder and cached (`.pt`, gitignored), then the world model trains on
the cache:

* `encoder.py` — frozen `facebook/vjepa2-vitl-fpc64-256` (the `fpc16` repos are
  SSv2 classifiers, not the base encoder). A clip `[B, 64, 3, 256, 256]` → `[B,
  32, 1024]`: 32 temporal groups (derived from the token grid, not hard-coded),
  each a spatial mean-pool of its 256 patches.
* `precompute_latents.py` — `video_to_latents_sliding`: slide a `clip_len`-frame
  window by `stride_frames` (with overlap), encode each window, and keep its
  central band, so adjacent bands tile the video into one **continuous** latent
  stream → cache `.pt`. We feed *real* frames, not a pre-encoder temporal average:
  V-JEPA-2 is motion-aware, so averaging frames before it would destroy the signal
  it reads (an earlier `frames.py` aggregate-and-chunk path, now removed).
* `data.py` — `load_video_frames`: decode any `.mp4` to `[F, H, W, C]` with
  `decord` (uniform temporal subsampling to a fixed length). Source-agnostic;
  dataset listing/download lives with each source (see `egolife.py`).
* `train.py` — `load_latents` + per-channel `standardize_latents` (raw encoder
  features are large-magnitude and drive the memory's inner loop to `NaN`
  otherwise). On cached real video latents the L1 drops cleanly with no
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

Note the caveat now attached to this section: the plain sum over the window also
scales the inner step, so a `c>1` run changes two things at once. See *Assembling
Atlas* for the measurement and for `window_normalize`, which separates them.

`docs/atlas_vs_titans.md` (French) is a fuller ATLAS-vs-Titans summary: Titans is a
degenerate special case of ATLAS (window `c=1`, identity feature map, first-order
inner optimiser). All three ATLAS ideas are now implemented — the Omega rule, the
polynomial feature map `φ`, and the Muon/Newton-Schulz inner optimiser — alongside
Table 7's memory recipe; the DeepTransformers/SWDT connection stays out of scope.

## EgoLife loader (longer, more diverse video)

`src/titans/worldmodel/egolife.py` uses `lmms-lab/EgoLife` as the video source:
continuous first-person life-logging (6 people × 7 days), cut into ~30 s /
600-frame / 20 fps segments
(`<person>/<day>/<...>.mp4`). That's 3× longer per segment and far more temporally
continuous than short action clips — a better fit for a
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
* `selective_copy` — MAD-style selective copying: content tokens scattered at
  random positions among BLANK filler; after a SEP, reproduce only the content
  tokens in order. Random positions make positional copying useless, so this
  probes content-addressed selection (contrast `copy_task`, a fixed-lag shift).
* `fuzzy_recall` — MAD-style fuzzy recall: associative recall where every real key
  is shadowed by a near-duplicate lure key (its adjacent id) carrying a *different*
  value, so retrieval must be robust to same-neighbourhood interference, not just
  exact-match lookup. (Embeddings are learned, so adjacency is a soft near-duplicate
  cue; the dominant added difficulty is interference — documented in the docstring.)

### Difficulty sweep: 3 memory-update configs at a matched budget (`sweep.py`)

`src/titans/membench/sweep.py` compares three ways the neural memory carries its
momentum, all sharing the *same* memory-MLP geometry so the parameter budget is
matched: **baseline** (scalar momentum `S_t`), **uncoupled** (the genuine
`K`-independent-filter null hypothesis — `cascade_topology="independent"`: `K`
uncoupled exponential filters, the gradient injected into all of them, readout =
mean; so it is *not* bit-identical to baseline but isolates what the cascade's
*coupling* buys over merely having `K` filters), and **coupled** (the full
data-dependent, non-commuting selective cascade). Configs baseline/uncoupled have identical parameter counts; coupled adds
exactly one small `cascade.beta_head = Linear(dim, K−1)` — a small, disclosed delta,
not hidden. `run_sweep` trains a small matched-budget model per (task, difficulty,
config) cell over MQAR (`associative_recall`), `selective_copy`, and `fuzzy_recall`,
and dumps a JSON of per-cell held-out accuracy to `results/`. Run with
`python -m titans.membench.sweep` (add `--smoke` for the tiny harness-check scale;
`tests/test_sweep.py` covers that). No plotting — raw numbers only.

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

## Assembling Atlas: `φ` + Omega + Muon

With the polynomial feature map (`features.py`) the **Atlas update rule is
complete** — ATLAS Eq 57-58:

```
M_t = α_t M_{t-1} + NewtonSchulz5(S_t)
S_t = θ_t S_{t-1} − Σ_{i=t-c+1}^{t} η_i ∇‖M_{t-1}(φ(k_i)) − v_i‖²
```

| ATLAS component | where | flag |
|---|---|---|
| Omega rule, sliding window `c` | `memory.py` | `memory_window_size=c` |
| Muon / Newton-Schulz inner optimiser | `memory.py::newton_schulz5` | `memory_orthogonalize_momentum=True` |
| **polynomial feature map `φ`** | `features.py` | `feature_degree=p` |
| memory MLP recipe (Table 7) | `memory.py` | `memory_expansion=4`, `memory_activation="gelu"`, `memory_residual=True` |

`φ_p(x) = [x^β]_{|β|≤p}` raises the *effective* key dimension — and with it the
number of key-value pairs `M` can store (`O(d_k^p)` against `O(d_k)`, ATLAS
Prop. 2) — at **zero cost to the input projections**, which is the whole point.
Measured: `w0` goes `16×16 → 16×48` at `p=3` while `k_proj`/`q_proj` are
untouched; a membench model goes 196k → 279k parameters with the delta sitting
in the memory block (20.8k → 62.0k), not in attention or I/O.

**Documented deviation — the diagonal subset, not the full basis.** The paper's
`φ` is every monomial of degree ≤ `p`. That is unusable here: 2145 dimensions at
`d=64, p=2`, 47905 at `p=3`, and because our `M` is an *MLP* rather than a matrix
we cannot use the kernel trick to avoid materialising it — the first memory layer
would destroy the parameter budget `φ` exists to protect. We keep the pure powers
and drop the cross terms:

```
φ(x) = concat([c_1·x, c_2·x², …, c_p·x^p]),   dim = p·d,   c_i init 1/√(i!)
```

so `φ(q)·φ(k) = Σ_i c_i² ⟨q^i, k^i⟩` rather than `Σ_i a_i (qᵀk)^i`. The structure
of Eq 5 (a learned, gated mixture of increasing degrees) survives; the exact
kernel identity does not. The tensor-sketch of Kacham et al. (2024a), which ATLAS
cites, is the faithful alternative — not implemented. Also ours and not the
paper's: RMS-normalising before taking powers, without which `x^p` is numerically
explosive for unconstrained keys.

### Four silent defects found by auditing the assembled stack

Every flag above was tested in isolation and worked. Assembled, four did nothing
at all — and because they failed *silently*, no test caught them. Recorded here
because the pattern, not the individual bugs, is the lesson:

| defect | symptom | status |
|---|---|---|
| `memory_residual` skipped whenever `φ` was on | i.e. in exactly the config ATLAS specifies — the skip demanded equal in/out widths, which `φ` breaks | fixed: adds `φ`'s degree-1 block (the un-mapped key), which degenerates to the paper's formula when `φ` is the identity |
| `orthogonalize_momentum` scoped to the scalar branch | cascade + Newton-Schulz ran as plain cascade, bit-identical | fixed: NS now applies to the cascade readout, the quantity playing `S_t`'s role in Eq 57 |
| `window_normalize` never plumbed into `TitansMACBlock` | every experiment configured through the block ran unnormalised | fixed |
| `NeuralMemoryMLP.forward` hardcoded SiLU / no residual | the module and the functional inner loop could disagree for the same config | fixed; a test now asserts they agree |

**Consequence for a past result.** Eq 58 *sums* `c` gradients, so a wider window
enlarges the inner step as well as the context — measured `‖ΔM‖_F` = 119 / 190 /
299 for `c` = 1 / 4 / 8. The July finding *"Omega degrades recall"* (8 pairs
`56% → 33%`) therefore confounded "more context" with "≈1.6× bigger step" and
**cannot be read as evidence about Omega**. `window_normalize=True` divides by the
window's gate mass; measured, it flattens the step to 119 / 112 / 112. Default
**off**, since the plain sum is what the paper writes.

**Still open, documented not fixed.** The window is truncated at chunk starts: at
`c=4, chunk=8` the per-position widths are `[1,2,3,4,4,4,4,4]` rather than a
uniform 4, and since `chunk = segment_size` by default this happens every
segment. Fixing it needs gradients carried across a chunk boundary, which
conflicts with evaluating them all at the chunk-start weights.

## Results (2026-07-21/22): the collapse is fixed, and it buys something

Two measured outcomes, on two different tasks, each with its own control.

### Video world model -- the metaplastic cascade wins

10 seeds per config, 5 epochs on the 5999-sequence EgoLife cache, permutation tests:

| config | n | val L1 | vs persistence (0.6561) |
|---|---|---|---|
| **`atlas_cascade`** | 10 | **0.4293 ± 0.0168** | **+34.6%** |
| `titans` | 10 | 0.4711 ± 0.0398 | +28.2% |
| `atlas` | 10 | 0.5109 ± 0.0378 | +22.1% |

```
atlas_cascade vs atlas      +0.0816   p < 0.0001   ***
atlas_cascade vs titans     +0.0418   p = 0.0093   ***
atlas         vs titans     -0.0398   p = 0.044      *
```

**The ATLAS stack without the cascade is significantly *worse* than plain Titans**
on next-latent prediction; adding the cascade reverses that and beats Titans.
`atlas_cascade` also has the lowest variance. The cascade is this project's own
idea, from neither paper, and it is the only component that clearly wins here --
after being written off in July on the strength of an under-converged sweep.

### MQAR -- out-of-window memory goes from 0.23 to 0.99

`sane_ns` (`SANE` + Muon + learned attention temperature) at 12 pairs, where
plain Titans fails to learn the task at all on half its seeds:

| | in-window (control) | out-of-window (memory) |
|---|---|---|
| `sane_ns`, seed 1 | 1.0000 | **0.9941** |
| `sane_ns`, seed 0 | 1.0000 | **0.9858** |
| same config, **memory branch off** | 1.0000 | **0.0531 / 0.0731** |
| `titans` | 1.0000 | 0.132 |

*(chance = 0.0625)*

The anti-leak control is the point: disable the memory branch and out-of-window
accuracy falls to **exactly chance**, while in-window stays at 1.0. Nothing but
the threaded memory can be carrying the answer. For reference, full (unsegmented)
attention scores 0.997 on this task -- so in its good mode the memory does as well
as seeing the entire sequence.

**It is bimodal, and that is the open problem.** Roughly half the seeds reach the
high mode; the rest sit at the Titans baseline and ~12% die outright:

```
 8 pairs:  4/8 high (0.99, 0.99, 0.99, 0.76)   3 low (~0.27)   1 dead
12 pairs:  2/5 high (0.99, 0.99)               2 low (0.22)    1 dead
```

### The rank collapse is resolved -- but it was never the whole story

`PR(w1)` under `sane_ns` during real training at 12 pairs, against 1.000 for
Titans:

| step | seed 0 (ends high) | seed 1 (high) | seed 2 (ends low) | seed 3 (low) |
|---|---|---|---|---|
| 0 | 30.68 | 32.29 | 32.65 | 32.26 |
| 4000 | 25.15 | 22.57 | 25.18 | 25.54 |

Rank holds at 22-30 and **survives training**, unlike the earlier mitigations
(Newton-Schulz + bias-freeze + temperature eroded from 4.7 to 2.0 over 10k steps).

**But the two modes are indistinguishable by rank** -- 25.15 against 25.18. The
bimodality is not the collapse. The memory is healthy either way; one seed learns
to use it and the other does not. So the link between internal health and task
performance still does not exist, for the third time in this project. What
changed is that the *ceiling* moved: a healthy memory made 0.99 reachable, which
it was not before.

The open question is now an optimisation one -- why a healthy memory is exploited
only half the time -- not an architectural one.

### What the configuration is, and why each piece is there

| ingredient | what it fixes | alone? |
|---|---|---|
| `alpha_init=0.002` | forget-gate half-life of **one token** -> ~346 | fixes magnitude only |
| `normalize_qk="l2"` | `lambda_max ~ m` so ATLAS's bound `2/lambda_max` is ~0.25, not ~0.008 | -- |
| `theta_scale=0.03` | puts the inner step inside that bound | -- |
| `normalize_output=False` | the output norm pins `\|M(k)\| = 1` against a target at ~0.01, emptying the inner loss | diverges alone |
| `normalize_retrieval=True` | stands in for the paper's chunk-end layer norm; bounds the `h_i` feedback | -- |
| Muon + bias freeze | rank and retrieval diversity | doesn't survive training alone |
| `learned_qk_scale` | attention stops averaging its window | holds one segment alone |

Every one is inert or harmful on its own. This is the assembled set, and the
gains above are the assembled result -- no single-ingredient claim is supported.

## Data/parameter proportions: a deliberate, documented deviation

ATLAS trains at **40–88 tokens per parameter** (Table 7: 170M/15B, 340M/15B,
760M/30B, 1.3B/100B). Our video corpus is `5999 × 64 = 383,936` latents, so that
band would cap the model at **4.4k–9.6k parameters**. Our models are 143k–1.26M,
i.e. a real ratio of **0.3–2.7** — 15× to 130× off. Closing it on a 1M-parameter
model would need ~625,000 segments (~5,200 h of video, ~1,200 h of encoding)
against the 8,000 we have.

**We accept the deviation.** The consequence is explicit: *loss comparisons
against the paper are not interpretable; only internal A/B comparisons between
our own variants are.*

**But the ratio is weak evidence here, and should not be over-read.** It is an
LLM-scaling heuristic that assumes near-independent tokens. Consecutive video
latents are massively redundant — 64 latents from one 30 s segment are not 64
samples. Empirically the heuristic mis-predicts badly: in July, at a ratio of
**0.01** (10,240 latents, ~1M parameters) the measured train/val gap was **0.043**
and the model beat the persistence baseline. The honest indicator is the measured
generalisation gap, not the ratio, and by that measure the world model is not
data-limited. On `membench` the question is moot — tasks are generated on the fly,
so 20,000 steps puts a 279k-parameter model at **55 tokens/parameter**, inside
ATLAS's own band.

## Not yet built

The MAG and MAL variants from the paper (only MAC is implemented); per-channel
(vs scalar) memory gates (`alpha`/`eta`/`theta` are one scalar per token, via a
`Linear(dim, 1)` head, not per-channel); a true parallel scan for the momentum
recurrence (`_scan_chunked` is currently a Python loop across chunks, not a
hardware parallel scan); and, from the ATLAS side, the tensor-sketch feature map
that would restore `φ`'s cross terms (`docs/atlas_vs_titans.md` tracks what
remains).
