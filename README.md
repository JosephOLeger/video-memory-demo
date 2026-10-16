# titans-mac

Deep test-time memories (Titans, ATLAS) in PyTorch, written from scratch, and the
experiments behind two papers:

* **Paper A** (theory, ICLR 2027 submission): why a deep test-time memory diverges, and
  why, once its write is bounded, it collapses silently. The outer loop learns to switch
  off the momentum gate, and the memory state concentrates on a few directions while
  the task loss does not notice.
* **Paper B** (in progress): a memory branch grafted onto a frozen Qwen2.5-VL-3B, for
  long video streams (fixed fps, sliding window, the memory keeps what leaves the
  window).

The memory is the research object. The latent world model (paper A) and the VLM graft
(paper B) are what stress it.

## Install

```bash
pip install -e '.[test]'          # torch only; add '.[worldmodel]' to encode video
PYTHONPATH=src python -m pytest   # ~275 tests, under a minute on CPU
```

Every command below assumes `PYTHONPATH=src` and runs from the repo root.

## The memory (`src/titans/memory.py`)

`NeuralMemory` is an MLP whose weights `M` are written at test time by gradient steps
on an inner key/value loss. Per token `t`, with gates `alpha_t` (forget), `eta_t`
(momentum) and `theta_t` (step):

```
S_t = eta_t S_{t-1} - grad_t                  (momentum buffer, raw gradients)
M_t = (1 - alpha_t) M_{t-1} + dM_t
```

The write `dM_t` is the object of paper A. Three rules (`titans.memory.WRITE_RULES`):

| rule | write | role |
|---|---|---|
| `False` | `dM = S_t` with `S_t = eta S - theta grad` | literal Titans; diverges |
| `"unit"` (`--muon unit`, the `sane_ns` default) | `theta_t NS5(S_t)` with `S_t = eta S - grad` | ATLAS Eq 32; bounded, collapses |
| `"unit_squash"` (`--muon unit_squash --muon-scale c`) | `theta_t psi_c(‖S_t‖) NS5(S_t)`, `psi_c(n) = (2/pi) arctan(pi n / 2c)` | "usq": same bound, amplitude grows with ‖S‖; does not collapse |

Writes happen at every token, with the gradient taken at the weights frozen at the
start of each chunk (`memory_chunk_size`). Also in this file:

* ATLAS additions: Omega sliding window (`omega_window`, `omega_stride`) and the
  polynomial feature map (`feature_degree`).
* `rank_cap`.
* Replay instruments `alpha_override`, `eta_override` and `theta_override`, used by
  the chimera and dose probes.

`mac.py` is Memory As Context, which reads the memory into the attention context.
`stability.py` holds the curvature measurements. The metaplastic momentum cascade and
the abandoned write rules (`rescale`, `clip`, `squash`, `softclip`) were removed on
2026-10-16; the last version with them is the git tag `archive/cascade-2026-10-16`.

## Paper A: the bench (`src/titans/worldmodel/`)

A two-block MAC world model over PCA-128 EgoLife latents, trained on the `[V;V]` echo
task: the second copy of a video can only be predicted well if the memory kept the
first.

```bash
# train one bench model (10k steps, checkpoints every 2k in frozen_muon/)
python -m titans.worldmodel.echo --label mu_unit10k_s0 \
  --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
  --horizon 24 --steps 10000 --seed 0 --eta-init 0.995 --theta-scale 0.03 \
  --alpha-init 0.002 --muon unit --frozen-every 2000 \
  --out mu_unit10k_s0.json --frozen-out frozen_muon/mu_unit10k_s0.pt
# usq: --muon unit_squash --muon-scale 350
```

* `make_pca_cache` builds the PCA cache, which `echo` requires.
* `frozen.py` saves and loads the trained models. Which frozen models exist, and what
  each family is: `docs/FROZEN.md`.
* `retention.py`, `margin.py` and `long_run.py` measure retention, margin and long
  streams.
* The replay probes live in `scripts/diag_*`. The main one is
  `diag_pr_collapse_probe.py`, which replays a frozen model on long streams and
  records the state participation ratio and the gates. `diag_eta_probe.py` and
  `diag_swap_probe.py` wrap it to force eta, or to swap gate heads between models.

### Reproducing the paper A figures

| figure | script | input |
|---|---|---|
| state law, pr vs gates | `scripts/fig_state_law.py` | `results/prlaw2`, `results/prlaw3`, `results/alpha_eff` |
| state norm | `scripts/fig_state_norm.py` | `results/prlaw3`, `results/d2interne`, `results/titans_lit` |
| interventions (chimeras, dose) | `scripts/fig_interventions.py` | `results/prlaw3`, `results/s5` |
| step x window | `scripts/fig_theta_eff.py` | `results/theff2` |
| mechanism, loss blind to the wall | `scripts/fig_mechanism.py`, `scripts/fig_loss_blind.py` | `results/d2interne` (trained under the removed `rescale` rule: the data is kept, the trainer is in the archive tag) |
| retention protocol | `scripts/fig_protocol.py` | (schematic) |
| pr law (simulation and predictor) | `scripts/sim_pr_law.py`, `scripts/fig_pr_law.py` | `results/sim_pr_law.json`, `results/prlaw` |

`scripts/figstyle.py` holds the shared style. The generators of the data behind
these figures:

* `run_prlaw2.sh`, which calls `diag_pr_collapse_probe.py`;
* `diag_titans_train.py`, which writes `titans_lit`;
* `diag_nons_train.py` and `diag_decouple_train.py`.

The external check on the lucidrains reference stack is reproduced by `patch_psic.py`,
`train_r_psic.py`, `probe_r_*.py`, `measure_r_gates.py` and `eval_r_bpc.py`.

Most of `results/` is not tracked by git (only about 8 MB of it is). The trained
models in `frozen_muon/` live on the lab machines.

## Paper B: the graft (`src/titans/graft/`)

A `MemoryBranch` sits at one high decoder layer of a frozen Qwen2.5-VL-3B. It is a
lateral branch: Qwen's attention is never replaced, and the branch is the identity at
init. It reads PCA-128 projections of the layer states (`fit_basis.py`, one token in
8) and writes them into a `NeuralMemory`. Main pieces:

* `branch.py`, `qwen.py`, `check_qwen.py`: the branch, the hook, and identity checks.
* `train_color.py`: the first functional result, recall of a past colour flash
  (`needle.py` builds the needle).
* `train_load.py`: the heavy-load associative task (K pairs, assembled-past bank).
  Paper A uses it to show that free gates kill the memory while blocked gates keep it,
  with no difference on the task.
* `stream_targets.py`, `stream_color_targets.py` and `train_stream.py`: the streaming
  regime. Questions are about shapes, colours or objects that left the window; the
  loss is CE plus an answer-latent distance to a teacher that sees the evidence.
  `eval_stream.py` scores it with the right memory, a wrong-video memory, the window
  alone, and the teacher.
* `demo_server.py` and `demo.html`: the interactive colour demo.

```bash
python -m titans.graft.stream_color_targets --reuse ~/work/iclr/stream/tg \
  --data data/finevideo --out ~/work/iclr/stream/tgc
python -m titans.graft.train_stream --targets '~/work/iclr/stream/tgc/*.pt' \
  --basis cache/qwen_l24_pca128.pt --kinds color --max-past 8 \
  --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --out run
```

The training data is FineVideo (`scripts/fetch_finevideo.py`, `audit_finevideo.py`).
It has no overlap with LongVideoBench.

## Where things live

* The lab journal is kept outside the repo. Long runs go on the lab machines, under
  `~/work/iclr/` on each.
* `scripts/nightly_report.sh`, `status_md.py` and `pull_report.sh` are called by a
  cron job that mirrors the box into `results/box`. Do not delete them while that
  job exists.
