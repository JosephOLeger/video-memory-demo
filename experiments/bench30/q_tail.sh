#!/bin/bash
# bench30 tail queue (agent bench30): after q_main, reconstructed BABILong qa11-qa20 at 4k (50 per task).
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
T=~/work/iclr/text
KV="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
DP="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official"
run() { echo "START $(date +%F_%T) $2" >> $L; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
E="--jsonl-eval evalsets/bl1120_4k_reconstructed.jsonl --jsonl-n 500"
while kill -0 2463723 2>/dev/null; do sleep 30; done
run 14400 bl_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt $E --jsonl-conds win,full,mem,omem --out runs/bl_base_kv" &
sleep 90
run 14400 bl_base_dp "$DP --init ckpt/smixA_deep_ct_pres.pt $E --jsonl-conds mem,omem --out runs/bl_base_dp"
wait
[ -f runs/b30_kv.pt ] && run 14400 bl_b30_kv "$KV --init runs/b30_kv.pt $E --jsonl-conds mem,omem --out runs/bl_b30_kv" &
sleep 90
[ -f runs/b30_dp.pt ] && run 14400 bl_b30_dp "$DP --init runs/b30_dp.pt $E --jsonl-conds mem,omem --out runs/bl_b30_dp"
wait
echo "QTAIL-FINI $(date +%F_%T)" >> $L
