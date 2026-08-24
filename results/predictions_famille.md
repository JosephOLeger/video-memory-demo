# Predictions — magnitude family redo, matched budget + joint score (2026-08-21)

Registered BEFORE launching `jobs_fam.txt` (pool4 workers on card 3, then more
cards as the repro ladder frees them). This is the draft's "experience a
verrouiller avant le 10 septembre": Table 1's current reference number
(rescale 0.2058) comes from a model that goes NaN when replayed on validation,
so the table must be rebuilt with a joint score — diverged-at-replay AND
acc/memory_gain among survivors — at matched budget.

## What already exists (no retrain)

frozen_muon/ holds 4000-step × 5-seed artefacts for: rescale, rescale_nh,
clip(c=1.0, measured inert), unit, unit_nh, unit_sig, usq(c=350), usq_sig.
These get RE-SCORED jointly, not rerun. All arms: sane_ns + eta_init 0.995,
theta_scale 0.03, alpha_init 0.002, cache egolife_pca128 (T=64).

## New arms (25 runs)

Calibration first (`scripts/calib_snorm.py` on the 5 mu_rescale artefacts):
c is set at the measured median of the compressed quantity, per the draft's
own lesson (c=1 was inert here, c=0.25 saturated on the graft — both guessed).
`C_S` = median per-chunk ||S||_F; `C_TS` = median theta_t*||S|| (softclip's
quantity, per echo.py's --muon-scale help).

- `mu_clipc` ×5 : `--muon clip --muon-scale C_S`, 4000 steps.
- `mu_sq`    ×5 : `--muon squash --muon-scale C_S`, 4000 steps.
- `mu_scl`   ×5 : `--muon softclip --muon-scale C_TS`, 4000 steps.
- `mu_unit10k` ×5 : `--muon unit`, 10000 steps, `--frozen-every 2000`.
- `mu_usq10k`  ×5 : `--muon unit_squash --muon-scale 350`, 10000 steps,
  `--frozen-every 2000`.

## Predictions

1. **Joint scoring demotes rescale.** At replay, rescale (and clip c=1, its
   bit-identical twin) produce NaN on some seeds; their memory_gain rows are
   flagged invalid. The best *valid* unbounded-family number drops below the
   0.2058 headline. unit_squash keeps its place as best valid arm at 4000.
2. **Calibrated clip/squash/softclip are bounded AND competitive.** All three
   replay with 0 NaN. With c at the median, memory_gain lands between unit
   (~0.086) and rescale's level (~0.20) — squash ≥ clip (monotone tail kept),
   softclip ≈ squash. If instead they sit at unit's level, the magnitude
   hierarchy above the median carries no usable signal — itself reportable.
3. **The −58% is mostly budget (moderate confidence).** mu_unit10k closes a
   large part of the gap to the 4000-step unbounded reference: unit's real
   cost is a ~44× smaller step, i.e. slower learning, not a lost hierarchy.
   If the gap does NOT close by 10k, the cost is real and the paper says so.
   mu_usq10k stays at or above every 4000-step arm.
4. Snapshots every 2000 steps give the convergence trajectory
   (steps-2500-not-converged says memory_gain still doubles between 2.5k and
   10k — the snapshots either confirm the plateau or show it was not reached).

Scoring: after the queue drains, replay every arm (old + new) on validation
with NaN detection + the retention grid; report per arm
(n_diverged/5, memory_gain mean±sd over survivors). No survivor-only
averaging without printing n_diverged next to it.

## Calibration outcome (2026-08-21, before any famille run)

`results/calib_snorm.json`, 5 mu_rescale artefacts replayed on validation:
median per-chunk ||S||_F = 0.0701 (q25 0.011, q75 0.088, q90 0.120; per-artefact
medians 0.072/0.077/0.073/0.077/0.0042 — seed 4 is a low outlier, pooled median
robust). Median theta_t = 0.0064; median theta*||S|| = 4.48e-4 (q25 2.3e-6:
the bimodal gates are visible). Chosen: **C_S = 0.07**, **C_TS = 4.5e-4**.
So the old clip c=1.0 sat ~14x ABOVE the median — inert on this path for the
opposite reason it saturated on the graft (||S||~6-9e3 there). Same mistake,
both directions, one lesson: measure the median first.

## First read (2026-08-24, 25/25 done; PROVISIONAL until joint replay scoring)

- Prediction 2 EXCEEDED: clipc 0.3176±0.0214, sq 0.3016±0.0248 — not "between
  unit and rescale" but ABOVE rescale (0.206, invalid ref) at equal budget.
  If replay confirms 0 NaN, the stability↔quality trade-off is not a trade-off
  at calibrated c.
- Prediction 3 CONFIRMED strongly: unit10k 0.4263±0.0725 (5× its 4k score),
  usq10k 0.5332±0.0413. The −58% was budget.
- softclip c=4.5e-4 DEAD (0.0011): med(θ‖S‖) is dominated by the bimodal gate
  (q25=2.3e-6); the cap strangles exactly the high-θ writes that carry
  learning. Third instance of the c-trap — the "c = median" rule assumes a
  unimodal distribution. AMENDMENT: relaunched as mu_scl2, c=0.03 (~tail
  deposit scale, med‖S‖×θ_tail≈0.07×0.45), 5 seeds, 2026-08-24 10:06.
- Joint replay scoring launched 2026-08-24 10:06 (jobs_famscore.txt: 13 arms ×
  retention grid, 3 pool4 workers cards 1-3) → results/retention_fam_*.json.
