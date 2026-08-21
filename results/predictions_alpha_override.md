# Predictions — read-time alpha-override probe of the reach law (2026-08-21)

Registered BEFORE launching `scripts/run_alpha_override.sh` (9 artefacts from the
2026-08-20 alpha sweep × override alpha ∈ {7.8e-3, 2.0e-3, 4.9e-4, 1.2e-4},
retention gaps 0..32 blocks, n_eval 96).

## Why this probe replaces the alpha-init sweep

The 2026-08-20 sweep (3 alpha-inits × 3 seeds, `--muon unit`) cannot test
`alpha* ≈ 1/(2d)`: the realised forget gate after 4000 steps is ~1e-4 on every
arm (margin audit, `results/margin_alpha.json`) — the outer loop erases
`alpha_init` exactly as it erases `eta_init` (README §9, eta bimodality). The
law is a statement about a *constant test-time* alpha, so it is probed by
pinning alpha at read time on frozen artefacts (`--alpha-override`, commit
pending), which also separates the trace's physics from the trained reader's
behaviour.

Also observed in the same audit, worth its own follow-up: under `unit`, eta
collapses to ~0 on 9/9 artefacts despite `eta_init=0.995` (under `rescale` it
stayed at 0.98+). And `theta_t_max` reads at or below ~5e-5 on 5/9 artefacts
while the memories demonstrably work — either the compensation loop drives
theta to a floor, or `margin.py` mis-reads theta on `unit` artefacts. To be
resolved before any theta claim from this sweep.

## Predictions

- **P1 (the law holds on the trace):** the retention curves *cross* in
  override-alpha. ov=7.8e-3 is best at gaps +0/+64; ov=2.0e-3 best near +256;
  ov=4.9e-4 best at +1024/+2048. Peak location per override near d ≈ 1/(2·ov).
- **P2 (sharp null carried by the same instrument):** ov=7.8e-3 at +2048 has
  (1−alpha)^2048 ≈ e^−16 — the trace is erased, so retention_gain ≈ 0 ± noise
  there. If it stays clearly positive, the long-range signal is not carried by
  M's alpha-governed trace and the instrument does not measure the law at all.
- **P3 (readout confound):** if no crossing appears and the per-artefact
  ordering of the 2026-08-20 table (a20e3-trained best everywhere) persists
  under every override, then the trained read pathway dominates the measurement;
  §reach of the paper keeps the derivation + numerical verification and loses
  the "measured on the world model" figure, per the draft's own decision rule.

Baseline for comparison: `results/retention_alpha.json` (same artefacts, no
override — the learned ~1e-4 gate).
