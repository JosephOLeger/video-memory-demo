# Data, caches and checkpoints

Nothing below is in git. Paths are relative to the repository root unless stated.

| what | where it comes from | used by |
|---|---|---|
| `cache/egolife_pca128_{train,val}.pt` | `titans.worldmodel.make_pca_cache` on the EgoLife day-1 latents (`encode_egolife`) | paper A bench, `echo` |
| `cache/enwik8_emb128`, WikiText-103 BPE cache | `paper_a/scripts/build_text_cache.py`, `build_bpe_cache.py` | the text arm of paper A and the external check |
| `frozen_muon/*.pt` | trained bench models, `docs/FROZEN.md` | every replay probe and figure of paper A |
| `results/` | outputs of the probes and campaigns | the figure scripts |
| `cache/qwen_l24_pca128.pt` | `titans.graft.fit_basis` on FineVideo, layer 24 | the heavy-load task and the gaussian horizon |
| `runs/load/bank.pt` | `titans.graft.train_load --build-bank` | heavy-load task, gaussian horizon |

## Where the originals are (November 2026)

Two lab machines hold the only copies of the heavy files: the tower (`~/work/iclr/`, bench outputs under
`bench30/runs/`, `ctx/`, `hg/`, `baselines_ext/`) and the second machine (`~/work/iclr/`, `~/work/titans-mac/frozen_muon`,
`~/work/titans-mac/results`, `~/work/titans-mac/cache`). Access ends on 28 November 2026; the small files (jsonl,
json, logs, final checkpoints of the graft) are copied out before that date. Videos and downloadable caches are not
copied.

## Checkpoint format

Graft checkpoints are `torch.save` dicts with `args` (the argparse namespace of the run) and the branch state.
`memory_kwargs_from_args` rebuilds the same memory from `args` at evaluation, so training and evaluation never
disagree on the inner loop.
