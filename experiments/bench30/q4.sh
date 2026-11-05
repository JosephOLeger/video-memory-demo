#!/bin/bash
# bench30 q4 (agent bench30, 30/10 10 h 30): write vs read diagnostic on the 1-episode dev set (20 per type).
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
T=~/work/iclr/text
KV="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
DP="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official"
run() { echo "START $(date +%F_%T) $2" >> $L; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D=evalsets/b30_dev_4k_1ep_sp.jsonl
run 14400 dg_kvL "$KV --init runs/b30_kvL.pt --jsonl-eval $D --jsonl-conds tea,sup,orc,mem,omem,topk --out runs/dg_kvL" &
sleep 90
run 14400 dg_dpL "$DP --init runs/b30_dpL.pt --jsonl-eval $D --jsonl-conds sup,orc,mem,omem --out runs/dg_dpL"
wait
run 14400 dg_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval $D --jsonl-conds sup,orc,mem,omem,topk --out runs/dg_base_kv" &
sleep 90
run 14400 dg_offs "$DP --init ckpt/offs_deep_pres.pt --jsonl-eval $D --jsonl-conds sup,orc,mem,omem --out runs/dg_offs"
wait
echo "Q4-FINI $(date +%F_%T)" >> $L
