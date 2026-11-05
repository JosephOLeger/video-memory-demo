#!/bin/bash
# q_mdp_s2_long.sh (04/11): replicate m_dp_long (fin2 12.6 vs 9.7, p 0.002) on the seed-2 chain: m_dp_s2 continued
# 4000 more steps, same args as q_mdp_long.sh. Tower card 0 (Josep: free, use it). Then fin1, fin2, finbl.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=0; PY=~/miniforge3/envs/titans/bin/python; L=~/work/iclr/bench30/jobs_tower.log; M=m_dp_s2_long
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl; BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
echo "QMDP-S2-LONG-START $(date +%F_%T)" >> $L
[ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 36000 $M "$A --preserve 1.0 --init runs/m_dp_s2.pt --seed 4 --steps 4000 --warmup 10 --eval-every 0 --out runs/$M"
run 7200 fin1_$M "$A --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$M"
run 10800 fin2_$M "$A --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$M"
run 10800 finbl_$M "$A --init runs/$M.pt --official $BL10 --official-conds mem,omem,orc --out runs/finbl_$M"
echo "QMDP-S2-LONG-FINI $(date +%F_%T)" >> $L
