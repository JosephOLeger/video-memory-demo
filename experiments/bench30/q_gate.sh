#!/bin/bash
# q_gate.sh (04/11): hard write gate at eval on m_dp_long. Write strength showed the learned gate already ranks
# facts 12-25x above filler, but filler is 490 of 512 writes and so 2/3 of the written change. Question: if the
# vectors under a theta floor are not written at all, does mem rise toward orc (13.5) / sup (17.2)?
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30_sel.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
echo "QGATE-START $(date +%F_%T)" >> $L
for th in 0.02 0.005 0.05 0; do
  run 7200 gate${th}_m_dp_long "$A --init runs/m_dp_long.pt --jsonl-eval evalsets/b30_dev_4k_1ep_sp.jsonl --jsonl-n 600 --jsonl-conds mem,omem --log-write-strength --write-skip-theta $th --out runs/gate${th}_m_dp_long"
done
echo "QGATE-FINI $(date +%F_%T)" >> $L
