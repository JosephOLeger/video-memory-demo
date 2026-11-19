# Paper A: divergence and silent collapse

The bench is `titans.worldmodel` (echo task over EgoLife latents). The scripts here replay the frozen models,
measure the state, and draw the figures. Frozen models and `results/` live on the lab machines (`docs/FROZEN.md`,
`docs/DATA.md`).

## Replay probes (`scripts/diag_*.py`)

| script | question |
|---|---|
| `diag_pr_collapse_probe.py` | the main probe: replay a frozen model on long streams, record the participation ratio of the state and the gates |
| `diag_eta_probe.py`, `diag_swap_probe.py` | force eta at replay; swap gate heads between two models (chimeras) |
| `diag_cdose_probe.py` | dose of the squash scale c at replay |
| `diag_echo_recall_probe.py` | memory-isolated associative recall on frozen models |
| `diag_eta_trajectory.py` | which training mode switches eta off, and when |
| `diag_titans_train.py`, `diag_nons_train.py`, `diag_decouple_train.py` | the literal Titans rule, the missing cell of the direction x magnitude table, the decoupled update |
| `diag_graft_pr.py` | have the grafted memories collapsed? pr(W1) along a real stream |

`run_*.sh` are the campaigns that called them (card numbers and paths of the lab machines left as they ran).

## Figures

| figure | script | input |
|---|---|---|
| state law, pr against the gates | `fig_state_law.py` | `results/prlaw2`, `prlaw3`, `alpha_eff` |
| state norm | `fig_state_norm.py` | `results/prlaw3`, `d2interne`, `titans_lit` |
| interventions (chimeras, dose) | `fig_interventions.py` | `results/prlaw3`, `s5` |
| step x window | `fig_theta_eff.py` | `results/theff2` |
| mechanism, loss blind to the wall | `fig_mechanism.py`, `fig_loss_blind.py` | `results/d2interne` |
| retention protocol | `fig_protocol.py` | schematic |
| pr law, simulation and predictor | `sim_pr_law.py`, `fig_pr_law.py`, `pr_law_predictor.py`, `sim_predictor_check.py` | `results/sim_pr_law.json` |
| reach law | `fig_reach_law.py` | analytic |
| external 2x2 | `fig_external_2x2.py` | the lucidrains runs below |

`figstyle.py` holds the shared style. `analyze_pr_law.py` compares the law with the measured probes.

## External check (lucidrains titans-pytorch, enwik8)

`patch_psic.py` grafts the squash rule into the reference stack; `train_r_psic.py` retrains it; `probe_r_prlaw.py`,
`probe_r_psic.py`, `verify_r_collapse.py`, `measure_r_gates.py` and `eval_r_bpc.py` measure the same quantities
there. `build_text_cache.py`, `build_bpe_cache.py` and `build_stride_cache.py` prepare the text caches.

## Gaussian horizon (`horizon/horizon.sh`)

Two arms of `titans.graft.train_load` with a dense target whose horizon is drawn per sample from a gaussian: past
only (GP) or past and future mixed (GS), eight seeds each, 2500 steps. Run with the card numbers as arguments.
