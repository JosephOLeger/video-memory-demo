# Predictions — theta=0 read-time probe (2026-08-21)

Registered BEFORE launching. Instrument: `--theta-override 0.0` in retention.py
(mirror of `--alpha-override`, tests in `tests/test_theta_override.py`), on the
9 frozen artefacts of the 2026-08-20 alpha sweep, same grid as the alpha probes
(gaps 0..32 blocks, n_eval 96). Output `results/retention_theta_ov0.json`.

## The contradiction this discriminates

The margin audit (`results/margin_alpha.json`) reads `theta_t_max` at or below
~5e-5 on 5/9 of these artefacts — i.e. "the memory never writes anything" —
while the same artefacts show `mem_effect` ≈ +0.04..+0.05 and positive
`retention_gain` — i.e. "the memory demonstrably helps". Both cannot be true
of the write path. theta_override=0.0 removes test-time writing entirely, so:

- **P-A (expected): the effect comes from writes, the margin reading is wrong.**
  Under theta=0, `mem_effect` collapses to ≈0 and `retention_gain` falls to the
  memoff floor (`gain_memoff`) within noise, on ALL artefacts — including the
  five that margin.py claims never write. Consequence: margin.py mis-reads
  theta on `unit` artefacts (likely reading the gate at a site theta no longer
  passes through under `unit`, where theta is applied outside Newton-Schulz).
  **Fix margin.py before any theta claim in the paper**; no theta number from
  `margin_alpha.json` may be cited until then.
- **P-B: the effect survives without writes.** `mem_effect` stays clearly
  positive under theta=0. Then the measured benefit is carried by M_0 /
  architecture (a trained bias, not an episodic trace), the margin readings may
  be right, and every `mem_effect` interpretation in the project needs a
  theta=0 control column. This would be the more disruptive outcome.
- **Sharp internal check:** whatever the verdict, under theta=0 the
  `retention_gain` must equal `gain_memoff` within noise — if it does not, the
  probe itself is broken (something other than the write path crosses the gap)
  and neither conclusion may be drawn.

Baseline for comparison: `results/retention_alpha.json` (same artefacts, no
override).

## OUTCOME (2026-08-21 17:24, read 2026-08-24): P-A, clean

Under theta=0 on all 9 artefacts: retention_gain equals the memoff floor within
1.5e-4 at every gap (sharp internal check passes), and mem_effect collapses
from +0.015..+0.051 to |·|<0.003 on 9/9 — including the five artefacts whose
margin audit read theta~1e-6. All measured memory benefit comes from test-time
writes ⇒ margin.py mis-reads theta on `unit` artefacts. **No theta figure from
margin_alpha.json may be cited until the instrument is fixed** (suspects: b=0
single-sequence sampling; gate read site under `unit`). Result:
`results/retention_theta_ov0.json`.
