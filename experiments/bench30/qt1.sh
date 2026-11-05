#!/bin/bash
# bench30 tower CARD 1 ONLY queue (agent bench30, 30/10). Lock: qt1.lock. Never card 0 / card 2.
cd ~/work/iclr/bench30
exec 9>qt1.lock; flock -n 9 || { echo "qt1 already running"; exit 1; }
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/miniforge3/envs/titans/bin/python
L=~/work/iclr/bench30/jobs_tower.log
T=~/work/iclr/text
DP="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official"
run() { echo "START $(date +%F_%T) card1 $2" >> $L; CUDA_VISIBLE_DEVICES=1 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D=evalsets/b30_dev_4k_1ep_sp.jsonl
run 3600 chk_heads_conv "$DP --init ckpt/heads_d_conv.pt --official qa1:4k --official-conds mem,omem --out runs/chk_heads_conv"
run 14400 dg_heads "$DP --init ckpt/heads_d_conv.pt --jsonl-eval $D --jsonl-conds tea,sup,orc,mem,omem --out runs/dg_heads"
echo "QT1-FINI $(date +%F_%T)" >> $L
