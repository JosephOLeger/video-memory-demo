#!/bin/bash
# q_sparse.sh (04/11): sparse write in training. The eval-only hard gate (q_gate) lifted m_dp_long fin1 from 11.8 to
# 13.0 (vs omem 10.5): the learned gate ranks facts above filler but writes all 512 vectors. Here the floor is in the
# training too (straight-through gate, budget 12.5 % of writes), from m_dp_long, 2000 steps. Control: the same 2000
# extra steps without the gate (ctl_long2), to separate the gate from more training. .135 card 3, after q_gate.
cd ~/work/iclr/bench30
true
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
SP="--sparse-write 0.005 --sparse-tau 0.5 --sparse-keep 0.125 --sparse-budget-w 1.0"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl; BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
echo "QSPARSE-START $(date +%F_%T)" >> $L
B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 1800 sp_smoke "$A $SP --preserve 1.0 --init runs/m_dp_long.pt --seed 5 --steps 20 --warmup 10 --eval-every 0 --out runs/sp_smoke"
if ! grep -q "TRAIN-TEXT-AB-FINI" runs/sp_smoke.log || ! grep -q "sparse write: " runs/sp_smoke.log || ! grep -q "keep_hard" runs/sp_smoke.log; then echo "QSPARSE-ABORT $(date +%F_%T) smoke failed" >> $L; exit 1; fi
M=sp_w
[ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 21600 $M "$A $SP --preserve 1.0 --init runs/m_dp_long.pt --seed 5 --steps 2000 --warmup 10 --eval-every 0 --out runs/$M"
run 7200 fin1_$M "$A $SP --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$M"
run 10800 fin2_$M "$A $SP --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$M"
M=ctl_long2
[ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 21600 $M "$A --preserve 1.0 --init runs/m_dp_long.pt --seed 5 --steps 2000 --warmup 10 --eval-every 0 --out runs/$M"
run 7200 fin1_$M "$A --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$M"
run 10800 fin2_$M "$A --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$M"
run 7200 fin1sp_$M "$A $SP --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds mem,omem --out runs/fin1sp_$M"
run 10800 finbl_sp_w "$A $SP --init runs/sp_w.pt --official $BL10 --official-conds mem,omem,orc --out runs/finbl_sp_w"
echo "QSPARSE-FINI $(date +%F_%T)" >> $L
