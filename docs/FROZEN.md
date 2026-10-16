# Frozen models

Trained bench models, saved with their config (`titans.worldmodel.frozen`). They live on
the lab machine `.135` under `~/work/titans-mac/`, not in git. Every family below is the
`sane_ns` config plus the deltas listed. Unless noted: horizon 24, `eta_init` 0.995,
trained on `cache/egolife_pca128_train.pt`. `_sN` is the seed; `_atK` is a checkpoint
at step K (every 2000 steps).

## `frozen_muon/`

| family | seeds | write rule | what differs | used by |
|---|---|---|---|---|
| `mu_unit10k`, `mu_usq10k` | 0-7 | unit / usq c=350 | 10k steps, the paper's bench | Fig. state law, interventions, gate probes, rebuttal |
| `mu_unit`, `mu_usq` | 0-4 | unit / usq c=350 | memory hidden 64 | Fig. 3 recall probe, hetero probes |
| `mu_unit_sig`, `mu_usq_sig` | 0-4 / 0-2 | unit / usq | hidden 64, sigmoid theta gate | Fig. 3 recall probe |
| `mu_unit_nh` | 0-4 | unit | hidden 64, `normalize_hidden` 1.0 | normalize_hidden check |
| `mu_unit10k_dec`, `dec30k` | 0-1 | unit | `decouple_memory_update` (10k / 30k steps) | collapse mechanism, causal arm |
| `unit30k`, `usq30k` | 0-1 | unit / usq | 30k steps | long trajectories (prlaw3) |
| `rawdir_usq` | 0-1 | usq | raw direction instead of NS5 (`diag_nons_train.py`) | Fig. state norm, state law |
| `cdt_c35`, `cdt_c87`, `cdt_c1400`, `cdt_c3500` | 0-1 | usq | psi scale c swept | dose in c |
| `tl_*` | 0-1 | Titans literal (no NS) | eta_init `e*`, theta_scale `t*` | Titans literal divergence |
| `unit_str4`, `unit_ctrl64` | 0-1 | unit | caches `egolife_l256_str4`, `egolife_l256_ctrl64` | stride / length controls |
| `tx_unit`, `tx_usq`, `tx10k_unit` | 0-1 | unit / usq | text: `enwik8_emb128` | other modality |
| `wt_unit`, `wt_usq` | 0-1 | unit / usq | text: `wt103_qwen128` | other modality |

## `frozen_alpha/`

`al_*`: unit, hidden 64, `alpha_init` swept (`a20e3` = 2.0e-3, `a49e4`, `a78e3`); `al_T64_*`
is the same at horizon 64. Reach law (not in the submitted paper).

## `frozen_attic/` (archived 2026-10-16, to delete)

Everything trained before the bounded write rules: the `rescale` default, the metaplastic
cascade, and the abandoned `clip` / `squash` / `softclip` rules (`frozen_muon/mu_rescale*`,
`mu_clip*`, `mu_scl*`, `mu_sq`). No result from them is used. The cascade and those rules
were removed from the code; the last version that loads them is the git tag
`archive/cascade-2026-10-16`.
