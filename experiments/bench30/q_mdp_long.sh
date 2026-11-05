#!/bin/bash
# q_mdp_long.sh (03/11): longer learned-write training. m_dp (2000 steps of learned write) continued 4000 more steps
# (6000 in all), same args as the m stage of q_weekend minus the oracle curriculum (already done). .135 card 3, after
# q_m512b (same lock). Then fin1, fin2, finbl, to compare with m_dp at 2000.
cd ~/work/iclr/bench30
exec 9>q135width.lock; flock 9
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log; M=m_dp_long
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl; BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
echo "QMDP-LONG-START $(date +%F_%T)" >> $L
[ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 36000 $M "$A --preserve 1.0 --init runs/m_dp.pt --seed 3 --steps 4000 --warmup 10 --eval-every 0 --out runs/$M"
run 7200 fin1_$M "$A --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$M"
run 10800 fin2_$M "$A --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$M"
run 10800 finbl_$M "$A --init runs/$M.pt --official $BL10 --official-conds mem,omem,orc --out runs/finbl_$M"
echo "QMDP-LONG-FINI $(date +%F_%T)" >> $L
