# titans-mac

A from-scratch PyTorch implementation of Google Research's **Titans**
([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) and **ATLAS**
([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)) neural long-term memory,
applied to a **video world model in latent space**, plus two ingredients of our
own: the **`unit_squash` / ψ_c magnitude rule** (§7) — the write-size law the
stability results rest on — and a **metaplastic momentum cascade** (§4; out of
scope for the current paper).

The memory is the research object; the world model is what stresses it. The
repo's main product is the **stability and rank-collapse analysis** behind the
theory paper (ICLR). The Qwen graft (§6) is the second paper's thread; it trains
on a separate machine and is journaled outside this README.

---

## 0. Orientation — read this first

**The question.** Can a memory carried in *weights* hold episodic information that
attention cannot hold at matched cost? The object is an **associative memory**,
judged on recall, not on prediction loss. (Never call our own chunked block
"quadratic"; it isn't.)

**Where the project stands.**

- **Settled.** The divergence is a **curvature feedback** in the memory MLP's own
  hidden layer; what removes it is bounding the write's **magnitude**, not its
  direction (§7, §9). Un-rescaled Newton-Schulz (`unit`) survives 104 800 writes
  with no wall vs 80 for the rescale control; the **literal published rule is
  untrainable** (non-finite on 8/8 seeds within ≤8 outer steps) — no trainable
  unbounded reference exists. **Rank collapse is a learned gate pathology**: the
  outer loop drives the momentum gate to η≈0, the η head alone carries the
  healthy/collapsed difference (chimera swaps, 3/3 seed pairs), and no
  replay-time intervention restores function — stability is a **training-time
  design condition**. Reach is set by **α alone**, `α* ≈ 1/(2d)`,
  `SNR* ≈ 0.61·√(P/d)`. The **SANE** gate set is what makes the memory healthy
  (§9). Reproduced on the lucidrains reference stack, where retraining with ψ_c
  improves LM from 6.42 → 3.25 bpc. The graft is bit-identical to the frozen host
  at init and costs 0.0092 % of its parameters.
- **Live.** *Why* the outer loop abandons momentum when ψ is absent — the one open
  mechanism link, stated as open in the paper.
- **Refuted.** "More memory buys more reach" (it bought **less**: 3.8→0.5 min,
  h32→h512, because wider diverges sooner); the stability *margin* as a survival
  criterion; `normalize_hidden` as a fix (a 54× delay, still exponential); "η too
  large" as the collapse story (inverted — in-replay η is effectively **zero** on
  the collapsed arm); κ diverging as η→1 (it saturates at `m(m+1)/2`, `m` is the
  knob); CV(θ) as a lever.
- **Abandoned.** The synthetic MQAR/copy/needle benchmark; the
  `content_specificity` metric; **every rescale-era result** (see the purge below).

**Standing constraints on new experiments.**

- Every new arm is a **delta on `sane_ns`** — no runs outside the SANE gate set.
- Cascade results must never rest on **r = 0 pure accumulators** (`cas_none`,
  `cas_r0.03`) — diagnostics of the no-leak limit, not the architecture.
- The BPTT window must **exceed the copy length `T`**, or `memory_gain` measures
  nothing (§7).
- The GPUs are shared. Size the VRAM guard on the **heaviest** arm (§10).
- Register predictions **before** launching anything readable either way
  (`results/predictions_*.md`). The margin was refuted this way.
- **The rescale purge.** No result measured under `orthogonalize_momentum=rescale`
  (the default before 2026-08-17) may be used: those runs' step scaled with the
  state, so their stability was an accident. The Titans-literal arm is the only
  legitimate *unbounded* reference.
- **Gates are measured in-replay only** — never by feeding raw latents into
  `_project`. The heads are bimodal; the raw-latent distribution reads η≈0.995 on
  an arm whose real in-replay η is ~1e-6 (§7).

**Where things are.** Code here; real runs on the workstation (`~/work/titans-mac`,
synced by rsync — verify with `md5sum`, not `git status`). Frozen artefacts of the
collapse campaign in `frozen_muon/` on the box; outputs in `results/prlaw3/`. The
live lab log is **`predictions_collapse_2026-08-28.md`** (predictions, verdicts,
refutations, retractions, in order). Paper drafts live outside the repo.

---

## 1. What the memory is

The long-term memory `M` is a small MLP whose **weights are the state**, updated
online at test time by an inner meta-learning loop. At every token: project
`k,v,q = W_K x, W_V x, W_Q x`; measure `ℓ_t = ‖M_{t-1}(k_t) − v_t‖²`; update with
momentum ("surprise") + decay ("forget"):

```
S_t = η_t S_{t-1} − θ_t ∇_W ℓ_t          M_t = (1 − α_t) M_{t-1} + S_t
```

then retrieve `y_t = M_t(q_t)`. `α_t` (forget), `η_t` (momentum), `θ_t` (inner lr)
are per-token scalars from small learned heads on `x_t`.

`src/titans/memory.py::NeuralMemory` has two equivalent paths —
`forward_naive_sequential` (explicit reference) and `forward_chunked` (per-chunk
batched gradients via nested `vmap`, then a scanned recurrence) — plus
`update(x, state, S, chunk)` (the chunked recurrence without retrieval, what MAC
uses) and `retrieve(state, q)`. The two full paths agree to `1e-9` in float64
across chunk and window sizes (§8). The parallel form evaluates each token's
gradient at its **chunk-start** weights, so it equals the fully-online recurrence
only at `chunk_size=1`; the reference path takes the same argument, which is what
makes the equivalence test meaningful.

Read-time probe hooks: `rank_cap` (cap the online write of `w1` to a fixed
key-subspace) and `eta_override` / `alpha_override` / `theta_override` (force a
constant gate at replay). All default `None` → bit-identical.

## 2. MAC — Memory As Context

`src/titans/mac.py` (Titans §4.1). A sequence is split into segments of size `C`;
the memory carry `(state, S)` is threaded across segments **sequentially**,
parallelism lives inside each segment. Per segment: historical retrieval from the
carried memory → context `[persistent ; h_i ; segment]` → causal attention → last
`C` positions are `y_i` → **update the memory from `y_i`** (it learns from what
attention selected) → output = gated `y_i` + fresh retrieval, residual-wrapped.
Because attention only ever sees one segment window, **the threaded memory is the
block's only long-range pathway** — which every measurement below exploits.
Resolved under-specified points (gate form, conv kernel, `N % C == 0`) documented
in the module. No official reference exists, so correctness = internally
consistent, causal, faithful to the mechanism.

## 3. The ATLAS additions

Titans is the ATLAS degenerate case (`c=1`, identity feature map). All three ATLAS
ingredients are implemented, each **opt-in and default off** so Titans paths stay
bit-identical:

| component | flag |
|---|---|
| Omega rule, sliding window `c` | `memory_window_size=c` |
| Muon / Newton-Schulz inner optimiser | `memory_orthogonalize_momentum` |
| polynomial feature map `φ` | `feature_degree=p` |
| memory MLP recipe (Table 7) | `memory_expansion=4`, `gelu`, `memory_residual` |

`c>1` fits a gated sliding-window sum over the last `c` tokens — a local *context*
rather than isolated tokens — and adds **no extra backward pass** (the per-token
gradients already exist); `c=1` recovers Titans' cost exactly. Our `φ` keeps the
**diagonal subset** (pure powers `c_i·x^i`, RMS-normalised before powers), not the
full monomial set, which would blow the parameter budget for an MLP memory; the
kernel identity is lost, Eq 5's structure survives. Documented deviation.

## 4. Our contribution: the metaplastic momentum cascade

`src/titans/metaplastic.py`. Titans' single momentum scalar forces one decay to
control both reactivity and retention. We replace it with a **K-reservoir cascade**
(synaptic metaplasticity; Benna & Fusi 2016): `u_t = B·u_{t-1} + e₁·∇ℓ_t`, `B`
tridiagonal doubly-stochastic, `read(u)=u¹`. Opt-in, **default off**, `K=1`
collapses exactly onto the scalar rule. `η_t` becomes a per-token leak on the whole
state (restores the missing self-decay of reservoir 1). Two coupling modes: fixed
geometric `B`, and a **selective** per-edge-gated `B_t` (per-edge, so the `B_t`
genuinely don't commute — a scalar gate would reduce to a fixed filter bank).
Impulse response is power-law-like (log-log slope ≈ −0.52) toward equilibrium
`1/K`, never to 0. **Out of scope for the theory paper**; see §9 for why it dies in
practice (a conserved mode, not a leak-rate problem).

## 5. The world model

`src/titans/worldmodel/`. A `WorldModel` stacks MAC blocks to predict `Z_{t+1}`
from `Z_1..t` (L1 loss). Real-video chain: `encoder.py` (frozen
`facebook/vjepa2-vitl-fpc64-256`, clip → `[B,32,1024]`), `precompute_latents.py`
(sliding window → one **continuous** latent stream, real frames — V-JEPA is
motion-aware), `egolife.py` (`lmms-lab/EgoLife`, split **by person**, no identity
leakage).

**Input is a fixed PCA basis, no learned projection.** Learned I/O put >50 % of a
small model's params into I/O and let the projections **co-adapt** with the memory.
The learned I/O was removed 2026-08-06; a PCA cache (fit on **train only**) is the
only accepted input (`latent_dim` must equal `dim`, raw 1024-D raises). `k=128`
keeps 85.7 % variance, `k=256` 92.9 %. The reduction is a property of the *dataset*,
not a layer, and the harness auto-detects it from cache metadata. L1 in PCA space is
not comparable to non-PCA runs (the header prints the space + explained variance).

## 6. The graft — Qwen2.5-VL

The paper's actual claim: attach the block to something that already sees the world.
Host **Qwen2.5-VL-3B-Instruct** (36 layers, read from config), graft site layer
**24** (⅔ depth). `graft/branch.py`:

- **Only ever adds.** The branch reads one layer's hidden states, writes them into a
  `NeuralMemory`, retrieves, and **adds** the result back. It replaces no attention,
  shortens no context, touches no host parameter. `up` is zero-init, so at step 0
  the model is **bit-identical** to the host (`identity_check` asserts `torch.equal`
  on logits). One condition, non-cosmetic: `0 × NaN = NaN`, so a diverged memory is
  loud — never state the identity without it.
- **Input basis** = PCA fitted on the frozen host's own layer-24 activations →
  branch drops to **0.0092 % of the host** (344 835 trainable params). It moves the
  divergence wall (~256 → ~328 writes) but does not remove it; the magnitude rule
  does.
- **Read cadence decoupled from write cadence** (`write_stride`): multiply reach at
  zero VRAM cost, paying temporal resolution, never context. So the GPU ask is "at
  23 GB we get X minutes **at full temporal resolution**".
- **Cost** (`graft/check_qwen.py`): inference +0.39 GB at 676 tokens; **VRAM is read
  in writes, not tokens** (one memory state per write kept for the backward).
  `GraftedLayer` is not `torch.utils.checkpoint`-safe as written (its forward mutates
  carry/offset) — checkpoint `self.layer` only.

---

## 7. How memory is measured, and the write-size rule

Three instruments, in decreasing order of trust.

**Echo benchmark — `echo.py` (headline).** Next-step L1 is nearly blind to
long-term memory. Instead train on `[V;V]` (a video concatenated with itself, twice
what attention can see) predicting `K` steps ahead, then score copy 2 under two
prefixes on the **identical frames of V1**:

```
memory_gain = second_cross − second_repeat     (larger = better)
```

`mem_effect = second_memoff − second` (memory reset per segment) is the "does state
help at all" floor. **Train with full BPTT**: a read at `T+j` and its write at `j`
are exactly `T` apart, so any `--bptt-window ≤ T` severs every read/write pair and
the benchmark measures nothing (measured: `memory_gain` +0.215 → +0.032). The lever
against VRAM is `T` (`crop_cache.py`), never the window.

**Memory health — `harness.memory_health`.** `pr` (participation ratio of `w1`
singular values; 1.0 = rank-1 collapse), `norm`, `cos_ret`. Probed by running the
block's own segment loop on real latents. Health has repeatedly **failed to predict
task performance** — kept as a diagnostic of *what broke*, not a target.

**Inner-loop stability — `stability.py`.** Samples **per chunk** (the divergence
happens inside a single forward, so per-step sampling misses it). `trace_forward`
reports both candidate curvatures (`X=K` ATLAS's Gram, `X=H₀` the deep one), `‖h₀‖`,
norms, realised gates, `κ`, `ϱ(B)` — every raw quantity, so a spectral radius can be
recomputed offline. `ϱ` is **meaningless on a bounded-step path** (read the norms).
`graft/trace_stability.py` drives it on the real host; use `--videos` (continuous
clips), never `--repeat` (tiling makes the probe *easier*).

**Retention — `retention.py`.** Sweeps the echo over distance
(`[V;D₁..Dₙ;V]` vs `[U;D₁..Dₙ;V]`, distractors are other clips). **Retention and
`memory_gain` disagree** — a wider memory scores better at distance 0 *and* forgets
faster; rank on the curve, not the point. Absolute collapse rule: flag any artefact
whose state `pr < 0.10 × width` (tags 5/5 collapsed `unit` seeds at gap 32 vs 0/5
`usq`). Score `diverged` **and** `acc_mean` jointly — a finite near-rank-1 state is
just as dead.

**The write-size rule (`unit_squash` / ψ_c, the object the paper rests on).**
`newton_schulz5` restored `‖G‖_F`, so orthogonalisation was purely *directional* and
bounded nothing — "we use Muon so the step is bounded" was false here. ATLAS Eq 32
does **not** rescale: the lr multiplies an already-orthogonalised matrix (σ≈1), so
`‖ΔM‖ ≈ η_t` independently of the gradient — the only hard bound either paper has on
the inner loop. `orthogonalize_momentum` now takes five conventions:

| | `‖ΔM‖` | surprise |
|---|---|---|
| `rescale` (=`True`, bit-preserving) | `θ‖∇ℓ‖` — scales with state | unbounded |
| `unit` | `≈ θ` | learned gate only |
| `clip=c` | `min(‖S‖, c)` | flat above `c` |
| `squash=c` | `(2c/π)·arctan(π‖S‖/2c)` | monotone |
| **`unit_squash=c`** | `≈ θ·ψ_c(‖S‖)`, ψ_c∈[0,1) | **monotone** |

**`unit_squash` (Josep, 2026-08-18) is the `usq` family the paper is built on.** It
multiplies un-rescaled NS by the dimensionless `ψ_c(‖S‖) = (2/π)·arctan(π‖S‖/2c)`,
so the bound is exactly `unit`'s while amplitude stays monotone in the *accumulated*
surprise. `S` accumulates the **raw** gradient (θ never enters the momentum
recurrence). `unit` is the `c→0` limit (coupling ≡ 0), Titans-literal the `c→∞` end
— the family is a homotopy with the **coupling** as the knob. State `c` in effective
units.

## 8. Tests

```bash
pip install -e '.[test]'
pytest
```

Beyond shapes/trainability the suite guards the properties that were easy to break:
the two memory paths agree in float64 across chunk **and** window sizes; MAC
causality across segment boundaries; `grad-of-sum == sum-of-grads` for the Omega
window; **every opt-in flag actually changes the computation** (several were once
silently inert); the echo probe's defining property (identical scored halves, a
memoryless predictor scores exactly 0); PCA fitted on train only, no learned I/O
survives. That last class — "does the flag do anything?" — is the one worth
extending when adding a flag.

## 9. Key results

The full experiment log (with pre-registered predictions, refutations and two
retractions) is `predictions_collapse_2026-08-28.md`; the numbered blow-by-blow of
each study is in this README's git history. The durable conclusions:

**Stability / divergence.**
- The divergence is the deep memory's **own curvature**: `W₀` grows → `h₀=σ(W₀k+b₀)`
  grows → `λ≈‖h₀‖²` grows → effective step `θκλ` grows → `W₀` grows more. `normalize_qk`
  pins `‖k‖=1` but nothing bounds the hidden layer. Predicted from theory (`‖h₀‖≈2`
  threshold), then **measured directly on Qwen2.5-VL** — `‖h₀‖` turns the trajectory
  over right where `θΛ` crosses 2.
- **Bounding the update is the fix, not bounding a curvature.** On a continuous
  stream (104 800 fresh writes), `unit` grows `‖W₁‖` 6.5→42.7 (×1.00001, bounded)
  while `rescale` explodes at 80 writes and `rescale+normalize_hidden` only delays to
  4 328 (still exponential). On the world model, `unit` trains **5/5 seeds healthy
  where `rescale` diverged 0/5**, at −59 % `memory_gain` — the cost the `(c,α)` family
  was built to price.
- Bounding each step does **not** bound the state; the state bound is the pair
  `(c, α)` (`‖M∞‖ ≤ c/α`).

**Rank collapse (the paper's C3, campaign 2026-08-22→09-01).**
- **Learned, not dynamical.** `unit` state pr falls 69 → 2.9 between 2k and 10k outer
  steps; by 30k it is an attractor. At 10k the seed lottery *is* the learned write
  diversity.
- **The carrier is the η head.** Chimera transplants into a collapsed `unit` at
  replay: replay rule, k/v/q, θ, α, learned init — none rescue; **the η head alone
  gives full rescue** (write `r_eff` 2-4 → 38-68, state pr → 39-59, 3/3 seed pairs).
  In-replay η median ~1e-6 on `unit` vs 0.94-0.99 on `usq`. Forcing constant η at
  replay reproduces the whole dose curve (`r_eff` 1.3 at η=0 → 52 at η=0.98). Momentum
  is what feeds NS rank.
- **No replay-time cure.** Restoring η repairs write geometry but destroys function
  (gap-0 `mem_effect` +0.221 → −0.72): the read path is co-adapted to momentum-less
  writes. Stability against collapse is a **training-time design condition**.
- **ψ_c is the operative ingredient**; *why* it keeps η alive is the one open
  mechanism link, stated as open in the paper. `rawdir_usq` (raw direction + ψ_c
  magnitude) trains healthy 2/2 — stability and rank are carried by the magnitude
  rule; NS adds ~+25 % quality.
- **State-level law.** Two channels: early drop by correlated rank-1 fluctuations,
  then lock-in by coherent drift. The DC law
  `pr(L) = pr(dec^L·W₀ + (Σw)·m̄)` predicts the α-override dose within 2-24 % with no
  free parameter. α is **not** a mitigation lever.
- **Isolated recall vs load** (`scripts/diag_echo_recall_probe.py`). Write `N` real
  val latents into a *fresh* memory (memory alone — no attention, no readout — so no
  co-adaptation confound), query by **key** (`k_proj`, the memory's own
  reconstruction objective; `q_proj` washes out), score top-1 against the value bank.
  Rank behaves as **capacity**: below the collapsed memory's reduced capacity
  collapse costs nothing (`unit` is even *sharper*); past it (`N ≳ 256`) the diverse
  memory keeps absorbing while the collapsed one saturates. Three-way at `N=1024`
  (36 blocks): no-memory 0.004, collapsed (`pr<35`) 0.25, diverse (`pr≥35`) 0.68 —
  the geometric `pr` gap made functional. Scope, stated: high-load only,
  memory-isolated (not end-task), `pr` necessary-not-sufficient — a warning sign, not
  a guarantee.

**Gates, reach, SANE.**
- The **SANE** gate set (`alpha_init=0.002`, `normalize_qk="l2"`, `theta_scale=0.03`,
  `normalize_output=False`, Muon+bias freeze, `learned_qk_scale`) is what keeps the
  memory healthy — but **every ingredient is inert or harmful on its own**; the gains
  are the assembled result.
- **Reach is set by α alone**, in closed form: `α* = 1−√(d/(1+d)) ≈ 1/(2d)`,
  half-life `1.39·d`, `SNR* ≈ 0.61·√(P/d)`. Holding distance `d` at constant quality
  needs `P ∝ d` — the GPU ask in closed form.
- `eta_init` selects a **basin** (bimodal — η ends near 0 or near its start), not a
  value; this bimodality *is* the collapse mechanism.
- **Once SANE is in place the cascade adds nothing** on next-step L1 (`sane_ns` =
  `atlas_cascade_sane`). The cascade's earlier apparent win was compensating for
  broken gates. It dies in practice via a **conserved mode** (`B` eigenvalue exactly
  1, κ grows linearly), not a leak-rate problem; admissible regime `K ≥ log₄(T)`,
  `r_eff ≤ 0.1/T`.

**External reproduction (lucidrains `titans-pytorch`, enwik8).** Their trained
default collapses within a sequence (state pr 63.7 → 1-3, 12/12 streams) — through
the *forget* gate, not η (their architecture leaves a different gate open). ψ_c at
inference does not repair; ψ_c **retrained** (4000 steps, ~15 lines) revives the
writes and improves LM **6.42 → 3.25 bpc** global. The rule acts through the writes,
not the gates.

**Data/parameter ratio.** Our corpus is 15-130× outside ATLAS's 40-88 tokens/param
band, so **loss vs the paper is not interpretable; only internal A/B is**. But the
ratio is an LLM-scaling heuristic that assumes near-independent tokens, and video
latents are massively redundant — the honest indicator is the measured
generalisation gap, which is +0.9 % at 40 000 steps (no overfitting at 53 epochs).

## 10. Running experiments

```bash
# named world-model configs on a latent cache (PCA or raw, auto-detected)
python -m titans.worldmodel.atlas \
    --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
    --configs sane_ns,sane_ns_cascade --seeds 0,1,2 --epochs 10 --health-every 200

# the echo benchmark for one config
python -m titans.worldmodel.echo --label cas_r4 \
    --train ...pca128_train.pt --val ...pca128_val.pt \
    --horizon 24 --seed 0 --cascade --reservoirs 4 --leak none --out result.json

# --- the graft ---
python -m titans.graft.fit_basis --videos '<egolife>/**/*.mp4' --k 128 \
    --num-videos 24 --out cache/qwen_l24_pca128.pt
python -m titans.graft.check_qwen --basis cache/qwen_l24_pca128.pt \
    --video clip.mp4 --frames 16,32 --write-stride 1,8 --guard-gb 14
python -m titans.graft.trace_stability --videos '<egolife>/**/*.mp4' --num-videos 40 \
    --basis cache/qwen_l24_pca128.pt --frames 64 --chunk 8 \
    --override orthogonalize_momentum=unit --out trace.json
```

Key `echo.py` axes, all deltas on `sane_ns`: `--muon {rescale,unit,clip,squash}`
(magnitude convention), `--muon-scale c` (pick from `‖M‖ → c/α`), `--alpha-init a`
(reach knob, `a ≈ 1/(2d)` for recall distance `d` writes).

**On the shared box.** Size the VRAM guard on the **heaviest** arm
(`measure_vram.py`), never extrapolate from the lightest. `scripts/pool4.sh` is the
worker (free-VRAM guard with a per-card `flock` across read→pop→allocate, because the
guard is a check-then-act race; one requeue then drop; per-job timeout).
`scripts/keepalive.sh` + a cron entry restore dead workers; `scripts/tau_chain.sh` is
the worked multi-phase example (each phase gated on the previous jobs being *gone*
via `pgrep`, not on an empty queue). Arms must share the seed grid — the divergence
is deterministic per seed, so a failure rate only means something against a control
at the *same* seeds.

## 11. Not built, and retired

**Retired (2026-08-14): the synthetic benchmark** (`membench` MQAR/copy/needle) — it
found the SANE gate set and first characterised the in-forward divergence, but the
project is judged on real video latents; we do not build replacement benchmarks.
**Never built:** MAG/MAL variants (only MAC); per-channel gates (one scalar/token); a
true parallel scan; the tensor-sketch feature map; a language head (the graft is
built, its answer-token loss is not); a bound on the *first* layer's curvature
(largely moot — bounding the update bounds every block). Open theory items: per-token
`(H_i, v_i)` linearisation; a switched-system certificate for the data-dependent
gates; the admissible `(α, η)` region. Two are now closed — the threshold is measured,
and κ **saturates** at `m(m+1)/2` as η→1 (so `m`, not η, is the knob).
