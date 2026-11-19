# Text track

Host Qwen2.5-VL-3B, text only, frozen. The branch sits at layer 24. The past of a story is written in sliding chunks
of 256 tokens without the question; the window of 256 tokens and the question are read with the memory.

## Files

| file | role |
|---|---|
| `gen_babilong.py` | BABILong-like training stories: bAbI facts (train split) hidden in PG19 (train split), four counterfactual variants per story; `gen_babilong_x.py` adds qa5 to qa10 |
| `gen_bench30.py`, `QUESTION_TYPES.md` | thirty memory question types in seven surface forms (22 trained, 8 held out), same API as `gen_babilong` |
| `gen_rich.py` | a bank of filler passages written by Qwen2.5-VL-7B in the seven forms (env `B30_BG`), to replace the PG19 filler |
| `build_eval_b30.py`, `build_ext.py`, `build_bl1120.py` | fixed evaluation files: dev sets of the thirty types, LoCoMo and LongMemEval oracle, reconstructed BABILong qa11 to qa20 |
| `prep_host_eval.py`, `build_preserve_host.py` | host benchmarks (MMLU, ARC, GSM8K, TriviaQA, PG19, MMBench, POPE, ChartQA, TextVQA) and the training-split pool for `--preserve-host` |
| `para_b30.py` | reworded copies of the dev questions (Qwen 7B) |
| `train_b30.py` | training and every evaluation mode (docstring) |
| `slot_memory.py` | the slot-memory baseline at the same byte budget (`--mem-type pool`) |
| `score_b30.py`, `score_std.py` | tables from the outputs: judged accuracy and token F1; official metrics for RULER, LongBench, En.Dia |
| `queue.sh` | the functions and the exact flags of every run |

## Data (not in git)

Under this directory, or under `B30_DATA`:

* `bl_data/babi_raw/tasks_1-20_v1-2/` (bAbI en and en-10k), `bl_data/pg19_raw/pg19_train.jsonl` (first 600 books
  of shard 0 of emozilla/pg19 longer than 20k characters) and `pg19_validation.jsonl`;
* `bl_data/babilong_test/<len>/qaN.json`, the official test files of RMT-team/babilong (`data/qaN/<len>.json` on
  the hub), 0k to 128k;
* `basis_l24_blf_pca{k}.pt`, the PCA basis of layer 24; `train_b30.py` fits it on 64 stories when the file is missing;
* `evalsets/`: built by the `build_*.py` scripts; `evalsets/std/` holds RULER 4k (simonjegou/ruler, 13 tasks x 50),
  LongBench v1 (7 subsets x 100) and InfiniteBench En.Dia (40), in the `--jsonl-eval` record format
  (`id, type, input, question, all_answers, metric`).

## Recipes

```bash
cd paper_b/text; export PY=python CARD=0; . ./queue.sh
```

Training, three stages, each continued from the previous checkpoint (`$A` holds the shared flags):

1. oracle write on the easiest setting, chunks of 1000 steps until the memory reads on narration
   (`sup - omem >= 10` on the narration dev set): `B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 14400 e_dp_c1
   "$A --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --steps 1000 --warmup 10 --eval-every 0 --out runs/e_dp_c1"`;
2. all seven forms, load 1 to 8, 2000 steps: `B30_NEP=1:0.7,2:0.3 run 18000 w_dp "$A --oracle-write
   --oracle-distract 1:8:1500 --preserve 1.0 --init runs/e_dp_cN.pt --steps 2000 --eval-every 0 --out runs/w_dp"`;
3. learned write, oracle mixed in at stream level over the first 800 steps, 2000 steps: `B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1
   run 18000 m_dp "$A --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/w_dp.pt --steps 2000
   --eval-every 0 --out runs/m_dp"`;
4. the models of the paper continue step 3 by 4000 steps: `trn m_dp_long runs/m_dp.pt 3 4000` (seeds 4 and 5 for
   `m_dp_s2_long`, `m_dp_s3_long`); the rich-filler model is the same with `B30_BG=rich/bank.jsonl`.

Evaluation of a checkpoint `runs/NAME.pt`: `ev NAME` (thirty types), `evx NAME` (BABILong official, LongMemEval,
LoCoMo), `std NAME` (RULER, LongBench, En.Dia), `host NAME` (cost on the host). References that do not depend on the
model: `ref fin2 $D2` gives window only, full context, sliding notes and BM25 on a record file.

Tables: `python score_b30.py runs/fin2_NAME_b30_dev_4k.jsonl`, `python score_std.py evalsets/std/ruler_4k.jsonl
runs/std_NAME_ruler_4k.jsonl`. Paired comparisons between two models use the `*_ok` fields of the same records
(McNemar).
