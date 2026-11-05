#!/bin/bash
# bench30 .135 CARD 3 ONLY (agent bench30, 30/10 12 h 30). Branch 3, CACHE arm (same design as tower qt2.sh, deep arm):
# P1 read trained with oracle write (useful sentence ends + random other fact ends, K 1 -> 8 over 1500 steps), low load;
# gate (dev 1 episode, sup - omem >= 10) -> P2 oracle-mix curriculum to the learned top-k write, load 1 to 4.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
T=~/work/iclr/text
KV="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
run() { echo "START $(date +%F_%T) $2" >> $L; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
gap() { python3 -c "import json;d=json.load(open(\"runs/$1_b30_dev_4k_1ep_sp_summary.json\"))[\"_all\"];print(int(d[\"sup\"]-d[\"omem\"]>=10))"; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl; EXT=evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl; B11=evalsets/bl1120_4k_reconstructed.jsonl
BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
while kill -0 1634238 2>/dev/null; do sleep 30; done
B30_NEP=1:0.7,2:0.3 run 36000 rd_kv "$KV --oracle-write --oracle-distract 1:8:1500 --preserve 1.0 --init $T/runs_bl/offl_topk_pres.pt --steps 2000 --eval-every 0 --out runs/rd_kv"
run 14400 dg_rd_kv "$KV --init runs/rd_kv.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem,topk --out runs/dg_rd_kv"
if [ "$(gap dg_rd_kv)" = 1 ]; then
  echo "GATE $(date +%F_%T) rd_kv reads (sup - omem >= 10): P2 learned-write curriculum" >> $L
  B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 36000 wl_kv "$KV --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/rd_kv.pt --steps 2000 --eval-every 0 --out runs/wl_kv"
  M=wl_kv
else
  echo "GATE $(date +%F_%T) rd_kv does not read: P1 continued 2000 steps at K 8" >> $L
  B30_NEP=1:0.7,2:0.3 run 36000 rd_kv2 "$KV --oracle-write --oracle-distract 8:8:1 --preserve 1.0 --init runs/rd_kv.pt --steps 2000 --eval-every 0 --out runs/rd_kv2"
  M=rd_kv2
fi
run 14400 dg_$M "$KV --init runs/$M.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem,topk --out runs/dg_$M"
run 14400 off_$M "$KV --init runs/$M.pt --official $BL10 --official-conds mem,omem,orc --out runs/off_$M" &
sleep 90
run 14400 ev_$M "$KV --init runs/$M.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/ev_$M"
wait
run 14400 bl_$M "$KV --init runs/$M.pt --jsonl-eval $B11 --jsonl-n 500 --jsonl-conds mem,omem --out runs/bl_$M" &
sleep 90
run 14400 ext_$M "$KV --init runs/$M.pt --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_$M"
wait
echo "Q5-FINI $(date +%F_%T)" >> $L
