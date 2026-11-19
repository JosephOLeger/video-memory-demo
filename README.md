# titans-mac

Deep test-time memories in PyTorch, written from scratch, and the experiments of the paper *Stability and collapse
in deep test-time-training memories* (ICLR 2027 submission): a deep test-time memory diverges, and once its write is
bounded it collapses silently. The outer loop learns to switch off the momentum gate and the state concentrates on a
few directions while the task loss does not notice.

The memory branch grafted on Qwen2.5-VL (the second part of the work) has its own repository, `memory-graft`. The
few graft modules kept here are those the paper's heavy-load task and gaussian horizon run on.

## Install

```bash
pip install -e '.[test]'          # torch only
PYTHONPATH=src python -m pytest   # 260 tests, under a minute on CPU
```

Every command runs from the repository root with `PYTHONPATH=src`. Caches, frozen models and most of `results/`
are not in git; `docs/DATA.md` says what each experiment needs.

## Layout

| path | what |
|---|---|
| `src/titans/memory.py` | `NeuralMemory`: the MLP written at test time, its gates and write rules |
| `src/titans/mac.py`, `stability.py` | memory as context; curvature and spectral-radius measurements |
| `src/titans/worldmodel/` | the bench: a two-block world model over EgoLife latents, echo task, retention probes |
| `src/titans/graft/` | the branch on a frozen Qwen2.5-VL and the heavy-load associative task (`train_load`) |
| `tests/` | unit tests |
| `paper_a/` | replay probes, figure scripts, external check on the lucidrains stack, gaussian horizon queue |
| `results/` | the 8 MB of probe outputs and pre-registered predictions the figures read |
| `docs/` | data layout, frozen models, notes on the removed cascade |

## The memory

`NeuralMemory` is an MLP whose weights `M` are written at test time by gradient steps on an inner key/value loss.
Per token `t`, with gates `alpha_t` (forget), `eta_t` (momentum) and `theta_t` (step):

```
S_t = eta_t S_{t-1} - grad_t
M_t = (1 - alpha_t) M_{t-1} + dM_t
```

| rule | write | behaviour |
|---|---|---|
| `unit` (default) | `theta_t NS5(S_t)` | ATLAS eq. 32; bounded, collapses |
| `unit_squash` (`--muon unit_squash --muon-scale c`) | `theta_t psi_c(‖S_t‖) NS5(S_t)` | same bound, amplitude grows with ‖S‖; does not collapse |

The literal Titans rule (`orthogonalize_momentum=False`) is kept for the divergence measurements. Writes happen at
every token with the gradient taken at the weights frozen at the start of each chunk. The file also holds the ATLAS
additions (Omega sliding window, polynomial feature map), `rank_cap`, and the replay instruments `alpha_override`,
`eta_override`, `theta_override`. The metaplastic cascade and the abandoned write rules were removed in October 2026
(last version with them: commit `6a044ea`); `docs/CASCADE_NOTES.md` says what was learned from them.

## The bench

A two-block world model over PCA-128 EgoLife latents is trained on the `[V;V]` echo task: the second copy of a video
is predictable only if the memory kept the first.

```bash
python -m titans.worldmodel.echo --label mu_unit10k_s0 \
  --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
  --horizon 24 --steps 10000 --seed 0 --eta-init 0.995 --theta-scale 0.03 \
  --alpha-init 0.002 --muon unit --frozen-every 2000 \
  --out mu_unit10k_s0.json --frozen-out frozen_muon/mu_unit10k_s0.pt
```

`make_pca_cache` builds the cache that `echo` requires; `frozen.py` saves and loads the trained models
(`docs/FROZEN.md` lists them); `retention.py`, `margin.py` and `long_run.py` measure retention, margin and long
streams. The replay probes, the figure table and the external check are in `paper_a/README.md`.
