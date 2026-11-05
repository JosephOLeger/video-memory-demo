# bench30: memory on 30 question types (Paper B, text track)

Copied on 05/11/2026 from the two servers (`~/work/iclr/bench30`, .135 and the tower). Code only: runs, checkpoints
and eval sets stay on the servers. A script that differed between the two machines is kept twice, the tower copy
with the suffix `_tower`.

* `gen_bench30.py`: story generator (30 types, 7 surface forms, facts hidden in PG19), prompts and judge.
* `build_eval_b30.py`, `build_bl1120.py`, `build_ext.py`: fixed eval sets (dev sets, reconstructed BABILong
  qa11 to qa20, LoCoMo, LongMemEval).
* `train_b30.py`: training and every eval mode (`--jsonl-eval`, `--official`, `--host-eval`); `train_b30_sel.py`:
  write selection diagnostics (`--log-write-strength`, `--write-skip-theta`).
* `para_b30.py`: reworded copies of the dev sets; `prep_host_eval.py`: standard benchmarks for the effect of the
  graft outside its task.
* `q_*.sh`, `qt*.sh`: the queues actually run (one per experiment, dated in their header).

The `titans` package these runs import is `~/work/iclr/text/pylib/titans` on both servers. On 05/11 it was
checked file by file: it is identical to `src/titans` at this commit's parent (`af9f1b9`). The text helpers
(`gen_babilong.py`, `babilong_ext/`, ...) are in `experiments/text/`.

Run from the server, e.g.

    export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text TEXT_GEN=gen_bench30
    python train_b30.py <flags of the matching q_*.sh>

Results and decisions: `ASTAR/CARNET_PaperB.md`.
