#!/bin/bash
# q_para.sh (05/11): are we prisoners of the question wording? bench30 dev sets with the question reworded by
# Qwen 7B (para_b30.py), same stories and answers. Meaning check: teacher (facts in clear, set 1) and full context
# (set 2), graft off, on the original and on the reworded question; analysis keeps records the check gets right both ways.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=1; PY=~/miniforge3/envs/titans/bin/python; L=~/work/iclr/bench30/jobs_tower.log
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
E=evalsets; P1=$E/b30_dev_4k_1ep_sp_para.jsonl; P2=$E/b30_dev_4k_para.jsonl
echo "QPARA-START $(date +%F_%T)" >> $L
run 7200 chk1_orig "$A --init runs/m_dp_long.pt --jsonl-eval $E/b30_dev_4k_1ep_sp.jsonl --jsonl-conds tea --out runs/chk1_orig"
run 7200 chk2_orig "$A --init runs/m_dp_long.pt --jsonl-eval $E/b30_dev_4k.jsonl --jsonl-conds full --out runs/chk2_orig"
for M in m_dp_long m_dp_s2_long; do
  run 7200 pfin1_$M "$A --init runs/$M.pt --jsonl-eval $P1 --jsonl-conds sup,tea,mem,omem --out runs/pfin1_$M"
  run 10800 pfin2_$M "$A --init runs/$M.pt --jsonl-eval $P2 --jsonl-conds full,mem,omem --out runs/pfin2_$M"
done
echo "QPARA-FINI $(date +%F_%T)" >> $L
