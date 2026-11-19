# titans-mac

Deep test-time memories in PyTorch, written from scratch, and the experiments behind two papers.

* **Paper A** (theory, ICLR 2027 submission): a deep test-time memory diverges, and once its write is bounded it
  collapses silently. The outer loop learns to switch off the momentum gate and the state concentrates on a few
  directions while the task loss does not notice.
* **Paper B** (in progress): a small memory branch grafted onto a frozen Qwen2.5-VL-3B. It reads the layer-24
  states that leave a 256-token window, writes them into 16.6k weights by gradient steps, and is read back by the
  window. Content is measured against a memory written from another text or another video.

The memory is the research object. The latent world model (paper A) and the Qwen graft (paper B) are what stress it.

## Install

```bash
pip install -e '.[test]'          # torch only
pip install -e '.[graft]'         # adds transformers and the video decoders, for paper B
PYTHONPATH=src python -m pytest   # 275 tests, under a minute on CPU
```

Every command below runs from the repository root with `PYTHONPATH=src`. The data, checkpoints and run outputs are
not in git; `docs/DATA.md` says what each experiment needs and where the originals are.

## Layout

| path | what |
|---|---|
| `src/titans/memory.py` | `NeuralMemory`: the MLP written at test time, its gates and write rules |
| `src/titans/mac.py`, `stability.py` | memory as context; curvature and spectral-radius measurements |
| `src/titans/worldmodel/` | paper A bench: a two-block world model over EgoLife latents, echo task, retention probes |
| `src/titans/graft/` | paper B: the branch, the Qwen hook, the training objectives, the video tools |
| `tests/` | unit tests of all of the above |
| `paper_a/` | probes, figure scripts, external check on the lucidrains stack, gaussian horizon queue |
| `paper_b/text/` | the text track: BABILong-like data, thirty question types, trainer, scorers |
| `paper_b/video/` | the video track: FineVideo fetch and audit, amputated-past queue, CG-Bench tools |
| `paper_b/baselines/` | AHN (ByteDance) evaluated under our content control |
| `docs/` | data layout, frozen models, notes on the removed cascade and on ATLAS |

## The memory

`NeuralMemory` is an MLP whose weights `M` are written at test time by gradient steps on an inner key/value loss.
Per token `t`, with gates `alpha_t` (forget), `eta_t` (momentum) and `theta_t` (step):

```
S_t = eta_t S_{t-1} - grad_t
M_t = (1 - alpha_t) M_{t-1} + dM_t
```

Two write rules remain (`titans.memory.WRITE_RULES`):

| rule | write | behaviour |
|---|---|---|
| `unit` (default) | `theta_t NS5(S_t)` | ATLAS eq. 32; bounded, collapses |
| `unit_squash` (`--muon unit_squash --muon-scale c`) | `theta_t psi_c(‖S_t‖) NS5(S_t)` | same bound, amplitude grows with ‖S‖; does not collapse |

The literal Titans rule (`orthogonalize_momentum=False`) is kept for the divergence measurements. Writes happen at
every token with the gradient taken at the weights frozen at the start of each chunk. The file also holds the
ATLAS additions (Omega sliding window, polynomial feature map), `rank_cap`, and the replay instruments
`alpha_override`, `eta_override`, `theta_override`. The metaplastic cascade and the abandoned write rules were
removed in October 2026 (last version with them: commit `6a044ea`); `docs/CASCADE_NOTES.md` says what was learned from them.

## Paper A in short

A two-block world model over PCA-128 EgoLife latents is trained on the `[V;V]` echo task: the second copy of a video
is predictable only if the memory kept the first.

```bash
python -m titans.worldmodel.echo --label mu_unit10k_s0 \
  --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
  --horizon 24 --steps 10000 --seed 0 --eta-init 0.995 --theta-scale 0.03 \
  --alpha-init 0.002 --muon unit --frozen-every 2000 \
  --out mu_unit10k_s0.json --frozen-out frozen_muon/mu_unit10k_s0.pt
```

The replay probes, the figure scripts and the external check are in `paper_a/` (its README has the figure table).

## Paper B in short

`MemoryBranch` (`src/titans/graft/branch.py`) sits beside one high decoder layer of the frozen host. The host's
attention is never replaced and the branch is the identity at initialisation. It projects the layer states to 128
dimensions with a fixed PCA basis (`fit_basis.py`), writes them into a `NeuralMemory` (hidden 64, 16.6k weights,
65 KB of state whatever the length read), and adds its read back to the layer output. 0.33M parameters train:
the projections, the gate heads, the memory's initial weights.

Two tracks share this branch:

* **text** (`paper_b/text/`): the past of a story is written in chunks of 256 tokens, the question is hidden at
  write time. Training data are BABILong-like stories (bAbI facts hidden in PG19) and thirty classic memory question
  types in seven surface forms. Evaluation on the official BABILong test, LongMemEval, LoCoMo, RULER, LongBench,
  InfiniteBench En.Dia, plus the cost on the host (MMLU, TriviaQA, TextVQA, ...). Every table carries the window-only
  and full-context host, a memory written from another text, and two untrained memories (sliding notes, BM25).
* **video** (`paper_b/video/` and `titans.graft.train_ctx`): the memory replaces the clips that left the window,
  trained by distillation towards the same Qwen seeing the whole video. The control is a memory written from another
  video.

## Status of the results (November 2026)

* Text, BABILong 8k to 32k: the own memory beats the window, the sliding notes and AHN; a memory from another
  text is 15 to 25 points below, on three seeds. Full context and BM25 retrieval, which keeps the whole text or reads
  the question, stay above.
* Text, question types never trained: no gain.
* Video: the gain over the window is answer style, not content; a memory from another video scores the same.
* Host cost: 2 to 3 points on TriviaQA while a memory is carried.

The lab notebook with every run and decision is kept outside the repository.
