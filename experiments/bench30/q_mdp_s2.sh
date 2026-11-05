#!/bin/bash
# q_mdp_s2.sh (03/11): second seed of the m_dp chain (deep, 128, from the BABILong model heads_d_conv as m_dp),
# same stages and args as qt4 + q_weekend (tower card 1): e (1500 steps) -> 1000-step chunks until sup - omem >= 10
# on narration -> w (7 forms) -> m (learned write) -> fin1, fin2, finbl. Only the seeds change (2, chunks 200 + c).
cd ~/work/iclr/bench30
exec 9>qt1.lock; flock -n 9 || { echo "card 1 queue already running"; exit 1; }
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=1; PY=~/miniforge3/envs/titans/bin/python; L=~/work/iclr/bench30/jobs_tower.log
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
gap() { python3 -c "
import json
try:
    d=json.load(open('runs/$1_summary.json'))['_all']; print(int(d['sup']-d['omem']>=10))
except Exception: print(0)"; }
DN=evalsets/b30_dev_4k_1ep_narr.jsonl; D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl
BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
E=e_dp_s2; W=w_dp_s2; M=m_dp_s2
echo "QMDP-S2-START $(date +%F_%T)" >> $L
[ -f runs/$E.pt ] || B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 14400 $E "$A --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --init ckpt/heads_d_conv.pt --seed 2 --steps 1500 --eval-every 0 --out runs/$E"
prev=runs/$E.pt; last=$prev; ok=0
for c in $(seq 0 15); do
  [ $c -gt 0 ] && { pt=runs/${E}_c$c.pt
    [ -f $pt ] || B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 14400 ${E}_c$c "$A --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --init $prev --seed $((200 + c)) --steps 1000 --warmup 10 --eval-every 0 --out runs/${E}_c$c"
    [ -f $pt ] || continue; prev=$pt; last=$pt; }
  [ -f $last ] || break
  n=$(basename $last .pt)
  [ -f runs/dge_${n}_b30_dev_4k_1ep_narr_summary.json ] || run 7200 dge_$n "$A --init $last --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem --out runs/dge_$n"
  if [ "$(gap dge_${n}_b30_dev_4k_1ep_narr)" = 1 ]; then ok=1; echo "GATE $(date +%F_%T) $n reads on narration" >> $L; break; fi
done
FINAL=$last
if [ $ok = 1 ]; then
  [ -f runs/$W.pt ] || B30_NEP=1:0.7,2:0.3 run 18000 $W "$A --oracle-write --oracle-distract 1:8:1500 --preserve 1.0 --init $last --seed 2 --steps 2000 --eval-every 0 --out runs/$W"
  if [ -f runs/$W.pt ]; then
    FINAL=runs/$W.pt
    [ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 18000 $M "$A --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/$W.pt --seed 2 --steps 2000 --eval-every 0 --out runs/$M"
    [ -f runs/$M.pt ] && FINAL=runs/$M.pt
  fi
else
  echo "GATE $(date +%F_%T) $E never reached the gate; final evals on $last" >> $L
fi
N=$(basename $FINAL .pt); echo "FINAL-MODEL $(date +%F_%T) $FINAL" >> $L
run 7200 fin1_$N "$A --init $FINAL --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$N"
run 10800 fin2_$N "$A --init $FINAL --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$N"
run 10800 finbl_$N "$A --init $FINAL --official $BL10 --official-conds mem,omem,orc --out runs/finbl_$N"
echo "QMDP-S2-FINI $(date +%F_%T)" >> $L
