#!/bin/bash
# bench30 tower CARD 1 ONLY (agent bench30, 30/10 15 h 20). Step 2 after gate 1 failed (deep reads nothing even with
# oracle write): EASIEST setting first. Deep from heads_d_conv, oracle write of the useful sentences only (K = support),
# CONTENT tokens only (--b30-content), one surface form (narration), one episode, no BABILong mixing.
# Gate on the narration dev set (sup - omem >= 10): widen to all forms (content write, K 1 -> 8, 1 or 2 episodes);
# else: same easiest setting for 3000 steps at lr 1e-3 (slower, in case 3e-3 overshoots).
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
B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 36000 e_dp "$DP --oracle-write --oracle-distract 1:1:1 --b30-content --preserve 1.0 --init ckpt/heads_d_conv.pt --steps 1500 --eval-every 0 --out runs/e_dp"
run 14400 dge_dp "$DP --b30-content --init runs/e_dp.pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem --out runs/dge_dp"
if [ "$(gap dge_dp_b30_dev_4k_1ep_narr)" = 1 ]; then
  echo "GATE $(date +%F_%T) e_dp reads on narration: widen to all forms" >> $L
  B30_NEP=1:0.7,2:0.3 run 36000 w_dp "$DP --oracle-write --oracle-distract 1:8:1500 --b30-content --preserve 1.0 --init runs/e_dp.pt --steps 2000 --eval-every 0 --out runs/w_dp"
  run 14400 dgw_dp "$DP --b30-content --init runs/w_dp.pt --jsonl-eval $D1,$DN --jsonl-conds sup,orc,mem,omem --out runs/dgw_dp"
else
  echo "GATE $(date +%F_%T) e_dp does not read even in the easiest setting: lr 1e-3, 3000 steps" >> $L
  B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 36000 e_dp2 "$DP --lr 1e-3 --oracle-write --oracle-distract 1:1:1 --b30-content --preserve 1.0 --init ckpt/heads_d_conv.pt --steps 3000 --eval-every 0 --out runs/e_dp2"
  run 14400 dge_dp2 "$DP --b30-content --init runs/e_dp2.pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem --out runs/dge_dp2"
fi
echo "QT3-FINI $(date +%F_%T)" >> $L
