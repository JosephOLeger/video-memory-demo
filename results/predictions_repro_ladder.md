# Predictions — reproduction ladder on lucidrains/titans-pytorch (2026-08-21)

Registered BEFORE launching `~/repro/repro_ladder.sh` (box, card 2, 8000 steps
per rung, enwik8, DEPTH=4, SEQ_LEN=512, BATCH_SIZE=1 — same deviations as R0).

## Why a ladder

R0 (their `train_mac.py` as shipped, 20 000 steps, finished 2026-08-21 05:05)
did **not** diverge. The audit explains why: in this codebase the surprise
gradient is evaluated at weights frozen over `NEURAL_MEM_BATCH_SIZE` = 128
tokens (32 chunks), the surprise magnitude is discarded by un-rescaled
Newton-Schulz (`SPEC_NORM_SURPRISES=True`), the memory MLP sits inside
`ResidualNorm` (LayerNorm makes the surprise decay as 1/||W||), θ is
sigmoid-bounded at 0.1, and the resting forget gate is ~0.5 per 4-token chunk.
The paper's feedback loop is structurally absent. Each rung restores one piece:

- **R1** = `NEURAL_MEM_BATCH_SIZE` 128 → 4 (= chunk size). The loop closes at
  the paper's granularity; every other damper stays.
- **R2** = R1 + `SPEC_NORM_SURPRISES` → False. The surprise magnitude re-enters
  the update (their pre-June-2025 behaviour).
- **R3** = R2 + `mem_model_norm_add_residual` → False. The memory is the
  paper's bare `M(k) = MLP(k)`; the loss can grow like ||W||² again.

Irreducible residual gap: θ stays `sigmoid(·)·0.1` in all rungs (structural in
this codebase since its first stability commit, `4c45653`).

## Predictions

- **R1: no divergence.** The three remaining dampers (unit-norm update, 0.5/chunk
  forget, ResidualNorm) each suffice on our theory. Loss quality may change.
  If R1 *does* diverge, staleness was the single load-bearing damper — a
  stronger result than expected.
- **R2: at risk but probably still finite.** ResidualNorm's 1/||W|| contraction
  plus the 0.5/chunk forget bound the state; our world-model experience says
  the magnitude path matters most, so this is the first rung where a NaN is
  plausible.
- **R3: divergence expected within a few thousand steps.** Closest to the
  paper's recurrence; the only remaining guard is the bounded θ. **If R3
  survives 8000 steps, the bounded step alone stabilises even the
  paper-faithful configuration in their stack** — which would promote the
  paper's central claim (the write-step bound is the load-bearing stabiliser)
  from our codebase to the reference one.

Either outcome of R3 feeds §audit: the NaN returning at a specific rung gives
the dose-response figure; R3 surviving gives the bounded-θ headline.

Context anchors: issues #22 (DEPTH=4, NaN after ~8k iters) and #29 (seq 4096;
`max_grad_norm=1.0` "prevents NaN but divergence remains" — note their
`softclamp_grad_norm` has a 0·inf NaN bug on exactly-zero gradients, so that
flag is itself suspect). 8000 steps per rung is below #22's horizon under
*their* semantics; rung-relative comparison is the point, not issue replication.

## Amendment 2026-08-21 12:06 (operational only — predictions above unchanged)

First launch (11:04) measured R1 at **68.65 s/it** → 8000 steps = 152 h/rung:
infeasible. Closing the loop makes every step ~47× slower than R0 (~1.45 s/it).
Per the contingency announced before launch:

- `NUM_BATCHES` 8000 → **1000** in all three `train_r*.py` (still 128 000
  closed-loop inner writes per run — far beyond the few-hundred-write horizon of
  our own systems' divergences; but below issue #22's ~8k *outer*-step horizon,
  so a null must be phrased "no divergence within 1000 outer steps").
- **Reordered and parallelised**: R3 (decisive rung) first on card 2
  (`repro_ladder_c2.sh`, PID 2270612, then R1 on the same card), R2 on card 1
  (`repro_ladder_c1.sh`, PID 2270613). Both started 12:06:10. Card 0 untouched;
  worker count 4 (2 ladder rungs + 2 retention evals) within the envelope.
- R1's 32-step probe log preserved as `~/repro/logs/repro_r1_probe68s.out`
  (the 68.65 s/it evidence).

Decision rule unchanged: compare rungs at equal horizon; NaN at a specific rung
= dose-response figure; all three surviving = bounded-θ headline with the
1000-step caveat stated.
