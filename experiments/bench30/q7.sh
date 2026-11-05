#!/bin/bash
# bench30 .135 CARD 3 ONLY (agent bench30, 30/10 17 h 50). e_kv: sup 11.7 vs omem 8.7, training gap rising.
# Continue in 3 chunks of 1000 steps (same easiest setting), eval on the narration dev after each; widen when it reads.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
T=~/work/iclr/text
KV="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
run() { echo "START $(date +%F_%T) $2" >> $L; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
gap() { python3 -c "import json;d=json.load(open(\"runs/$1_summary.json\"))[\"_all\"];print(int(d[\"sup\"]-d[\"omem\"]>=10))"; }
DN=evalsets/b30_dev_4k_1ep_narr.jsonl; D1=evalsets/b30_dev_4k_1ep_sp.jsonl
prev=runs/e_kv.pt; ok=0
for c in 1 2 3; do
  B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 36000 e_kv_c$c "$KV --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --init $prev --seed $c --steps 1000 --warmup 10 --eval-every 0 --out runs/e_kv_c$c"
  run 14400 dge_kv_c$c "$KV --init runs/e_kv_c$c.pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem,topk --out runs/dge_kv_c$c"
  prev=runs/e_kv_c$c.pt
  if [ "$(gap dge_kv_c${c}_b30_dev_4k_1ep_narr)" = 1 ]; then ok=1; echo "GATE $(date +%F_%T) e_kv_c$c reads on narration" >> $L; break; fi
done
if [ $ok = 1 ]; then
  B30_NEP=1:0.7,2:0.3 run 36000 w_kv "$KV --oracle-write --oracle-distract 1:8:1500 --preserve 1.0 --init $prev --steps 2000 --eval-every 0 --out runs/w_kv"
  run 14400 dgw_kv "$KV --init runs/w_kv.pt --jsonl-eval $D1,$DN --jsonl-conds sup,orc,mem,omem,topk --out runs/dgw_kv"
else
  echo "GATE $(date +%F_%T) e_kv after 4000 steps still below the gate" >> $L
fi
echo "Q7-FINI $(date +%F_%T)" >> $L
