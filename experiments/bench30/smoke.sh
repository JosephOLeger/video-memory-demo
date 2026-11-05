#!/bin/bash
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/work/titans-mac/.venv/bin/python
C="--arm B --past 2048 --past-choices 2048 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
CUDA_VISIBLE_DEVICES=3 TEXT_GEN=gen_bench30 nice -n 10 timeout 1800 $PY train_b30.py $C --init ~/work/iclr/text/runs_bl/offl_topk_pres.pt --jsonl-eval evalsets/b30_dev_4k.jsonl --jsonl-n 8 --jsonl-conds win,mem,omem,full --out runs/smoke_kv_ev > runs/smoke_kv_ev.log 2>&1
echo SMOKE-DONE
