#!/bin/bash
# q_m512b.sh (03/11): redo the learned-write stage of the 512 run (m512s had 1324 OOM steps of 2000, OOM cascade
# fixed in train_b30.py on 03/11), same args as q_width.sh, from w512s.pt; then the same final evals. .135 card 3.
cd ~/work/iclr/bench30
exec 9>q135width.lock; flock -n 9 || { echo "width queue already running"; exit 1; }
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log; M=m512b
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_bld_pca{k}.pt --d-mem 512 --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl; BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
[ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 18000 $M "$A --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/w512s.pt --steps 2000 --eval-every 0 --out runs/$M"
echo "OOM-COUNT $(date +%F_%T) $M $(grep -c 'OOM at' runs/$M.log)" >> $L
run 7200 fin1_$M "$A --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$M"
run 10800 fin2_$M "$A --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$M"
run 10800 finbl_$M "$A --init runs/$M.pt --official $BL10 --official-conds mem,omem,orc --out runs/finbl_$M"
echo "QM512B-FINI $(date +%F_%T)" >> $L
