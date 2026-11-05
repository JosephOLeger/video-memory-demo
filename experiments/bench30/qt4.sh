#!/bin/bash
# bench30 tower CARD 1 ONLY (agent bench30, 30/10 16 h 45). e_dp (easiest setting) gave sup 14.0 vs omem 9.7 and a
# training gap still rising: continue it, 3 chunks of 1000 steps (same setting), eval on the narration dev after each.
# Stop early and widen to all forms as soon as sup - omem >= 10.
cd ~/work/iclr/bench30
exec 9>qt1.lock; flock -n 9 || { echo "card 1 queue already running"; exit 1; }
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/miniforge3/envs/titans/bin/python
L=~/work/iclr/bench30/jobs_tower.log
T=~/work/iclr/text
DP="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official"
run() { echo "START $(date +%F_%T) card1 $2" >> $L; CUDA_VISIBLE_DEVICES=1 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
gap() { python3 -c "import json;d=json.load(open(\"runs/$1_summary.json\"))[\"_all\"];print(int(d[\"sup\"]-d[\"omem\"]>=10))"; }
DN=evalsets/b30_dev_4k_1ep_narr.jsonl; D1=evalsets/b30_dev_4k_1ep_sp.jsonl
prev=runs/e_dp.pt; ok=0
for c in 1 2 3; do
  B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 36000 e_dp_c$c "$DP --oracle-write --oracle-distract 1:1:1 --b30-content --preserve 1.0 --init $prev --seed $c --steps 1000 --warmup 10 --eval-every 0 --out runs/e_dp_c$c"
  run 14400 dge_dp_c$c "$DP --b30-content --init runs/e_dp_c$c.pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem --out runs/dge_dp_c$c"
  prev=runs/e_dp_c$c.pt
  if [ "$(gap dge_dp_c${c}_b30_dev_4k_1ep_narr)" = 1 ]; then ok=1; echo "GATE $(date +%F_%T) e_dp_c$c reads on narration" >> $L; break; fi
done
if [ $ok = 1 ]; then
  B30_NEP=1:0.7,2:0.3 run 36000 w_dp "$DP --oracle-write --oracle-distract 1:8:1500 --b30-content --preserve 1.0 --init $prev --steps 2000 --eval-every 0 --out runs/w_dp"
  run 14400 dgw_dp "$DP --b30-content --init runs/w_dp.pt --jsonl-eval $D1,$DN --jsonl-conds sup,orc,mem,omem --out runs/dgw_dp"
else
  echo "GATE $(date +%F_%T) e_dp after 4000 steps still below the gate" >> $L
fi
echo "QT4-FINI $(date +%F_%T)" >> $L
