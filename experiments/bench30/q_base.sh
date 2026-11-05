#!/bin/bash
# bench30 (agent bench30, 29/10): baseline official BABILong 4k of the best cache checkpoint BEFORE bench30 training.
cd ~/work/iclr/text
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-pasts 4096,8192 --eval-dists 300 --eval-groups 25 --final-eval-groups 50 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
echo "START $(date +%F_%T) base_offl_topk_pres" >> $L
CUDA_VISIBLE_DEVICES=3 TEXT_GEN=gen_babilong nice -n 10 timeout 21600 $PY train_bl.py $C --init runs_bl/offl_topk_pres.pt --official qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k --official-conds win,mem,omem,full --out ~/work/iclr/bench30/runs/base_offl_topk_pres > ~/work/iclr/bench30/runs/base_offl_topk_pres.log 2>&1
echo "END $(date +%F_%T) base_offl_topk_pres rc=$?" >> $L
