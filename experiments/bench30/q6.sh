#!/bin/bash
# bench30 .135 CARD 3 ONLY (agent bench30, 30/10 16 h 30). Step 2 for the CACHE (gate 1 failed): easiest setting.
# Oracle write of the USEFUL sentence ends only (K = support), one form (narration), one episode, no BABILong mixing.
# Gate on the narration dev set (sup - omem >= 10): widen to all forms (K 1 -> 8); else lr 3e-4, 3000 steps.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
T=~/work/iclr/text
KV="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
run() { echo "START $(date +%F_%T) $2" >> $L; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
gap() { python3 -c "import json;d=json.load(open(\"runs/$1_summary.json\"))[\"_all\"];print(int(d[\"sup\"]-d[\"omem\"]>=10))"; }
DN=evalsets/b30_dev_4k_1ep_narr.jsonl; D1=evalsets/b30_dev_4k_1ep_sp.jsonl
B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 36000 e_kv "$KV --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --init $T/runs_bl/offl_topk_pres.pt --steps 1500 --eval-every 0 --out runs/e_kv"
run 14400 dge_kv "$KV --init runs/e_kv.pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem,topk --out runs/dge_kv"
if [ "$(gap dge_kv_b30_dev_4k_1ep_narr)" = 1 ]; then
  echo "GATE $(date +%F_%T) e_kv reads on narration: widen to all forms" >> $L
  B30_NEP=1:0.7,2:0.3 run 36000 w_kv "$KV --oracle-write --oracle-distract 1:8:1500 --preserve 1.0 --init runs/e_kv.pt --steps 2000 --eval-every 0 --out runs/w_kv"
  run 14400 dgw_kv "$KV --init runs/w_kv.pt --jsonl-eval $D1,$DN --jsonl-conds sup,orc,mem,omem,topk --out runs/dgw_kv"
else
  echo "GATE $(date +%F_%T) e_kv does not read even in the easiest setting: lr 3e-4, 3000 steps" >> $L
  B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 36000 e_kv2 "$KV --lr 3e-4 --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --init $T/runs_bl/offl_topk_pres.pt --steps 3000 --eval-every 0 --out runs/e_kv2"
  run 14400 dge_kv2 "$KV --init runs/e_kv2.pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem --out runs/dge_kv2"
fi
echo "Q6-FINI $(date +%F_%T)" >> $L
