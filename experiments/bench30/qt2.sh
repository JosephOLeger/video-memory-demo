#!/bin/bash
# bench30 tower CARD 1 ONLY (agent bench30, 30/10 12 h 20). Branch 3 (oracle-write read does not read on bench30):
# P1 = train the READ with an oracle write (useful sentences + random other fact sentences, K 1 -> 12 over 1500 steps),
# low load; gate on dev 1 episode (sup - omem >= 10 points) -> P2 = oracle-mix curriculum to the learned write
# (800 steps, stream level), load 1 to 4; else P1 continued. Lock qt1.lock (same card).
cd ~/work/iclr/bench30
exec 9>qt1.lock; flock -n 9 || { echo "card 1 queue already running"; exit 1; }
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/miniforge3/envs/titans/bin/python
L=~/work/iclr/bench30/jobs_tower.log
T=~/work/iclr/text
DP="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official"
run() { echo "START $(date +%F_%T) card1 $2" >> $L; CUDA_VISIBLE_DEVICES=1 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
gap() { python3 -c "import json;d=json.load(open(\"runs/$1_b30_dev_4k_1ep_sp_summary.json\"))[\"_all\"];print(int(d[\"sup\"]-d[\"omem\"]>=10))"; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl; EXT=evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl; B11=evalsets/bl1120_4k_reconstructed.jsonl
BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
B30_NEP=1:0.7,2:0.3 run 36000 rd_dp "$DP --oracle-write --oracle-distract 1:12:1500 --preserve 1.0 --init ckpt/heads_d_conv.pt --steps 2000 --eval-every 0 --out runs/rd_dp"
run 14400 dg_rd_dp "$DP --init runs/rd_dp.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/dg_rd_dp"
if [ "$(gap dg_rd_dp)" = 1 ]; then
  echo "GATE $(date +%F_%T) rd_dp reads (sup - omem >= 10): P2 learned-write curriculum" >> $L
  B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 36000 wl_dp "$DP --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/rd_dp.pt --steps 2000 --eval-every 0 --out runs/wl_dp"
  M=wl_dp
else
  echo "GATE $(date +%F_%T) rd_dp does not read: P1 continued 2000 steps, K 12 -> 20" >> $L
  B30_NEP=1:0.7,2:0.3 run 36000 rd_dp2 "$DP --oracle-write --oracle-distract 12:20:1000 --preserve 1.0 --init runs/rd_dp.pt --steps 2000 --eval-every 0 --out runs/rd_dp2"
  M=rd_dp2
fi
run 14400 dg_$M "$DP --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/dg_$M"
run 14400 off_$M "$DP --init runs/$M.pt --official $BL10 --official-conds mem,omem,orc --out runs/off_$M"
run 14400 ev_$M "$DP --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/ev_$M"
run 14400 bl_$M "$DP --init runs/$M.pt --jsonl-eval $B11 --jsonl-n 500 --jsonl-conds mem,omem --out runs/bl_$M"
run 14400 ext_$M "$DP --init runs/$M.pt --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_$M"
echo "QT2-FINI $(date +%F_%T)" >> $L
