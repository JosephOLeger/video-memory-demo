#!/bin/bash
# bench30 WEEKEND queue (agent bench30, 30/10 evening). Runs alone until done (no intervention), resumable.
# Usage: q_weekend.sh tower|135
#   tower: deep arm, CARD 1 ONLY, chain e_dp_c*; 135: cache arm, CARD 3 ONLY, chain e_kv_c*.
# 1. waits for the train_b30.py job already running (a chunk of the old queue), then
# 2. easiest-setting chunks of 1000 steps (oracle write of the useful sentences, narration, 1 episode, no BABILong),
#    narration dev eval after each, up to chunk 15, stops as soon as sup - omem >= 10;
# 3. if the gate passed: widen to all 7 forms (oracle write, K 1 -> 8, 1 or 2 episodes, 2000 steps) + diagnostic,
#    then learned-write curriculum (oracle-mix at stream level over 800 steps, load 1 to 4, 2000 steps) + diagnostic;
# 4. final evals of the last model (or of the last chunk if the gate never passed): dev 1 episode (sup, orc, mem, omem),
#    dev 2 episodes, BABILong qa1-qa10 4k official (mem, omem, orc), qa11-qa20 reconstructed, LoCoMo + LongMemEval.
# Every job has a hard timeout; a failure never stops the queue; one checkpoint name per job; last line QT-WEEKEND-FINI.
cd ~/work/iclr/bench30
BOX=$1
LOGK=$([ "$BOX" = tower ] && echo jobs_tower.log || echo jobs.log)
if [ -z "$DRY" ]; then
  for q in qt4.sh q7.sh; do for p in $(pgrep -u "$USER" -x bash -a | awk -v q="$q" '$3==q {print $1}'); do
    kill $p && echo "KILLED $(date +%F_%T) old queue $q bash pid $p (its running job finishes, weekend queue resumes after it)" >> $LOGK; done; done
fi
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text
if [ "$BOX" = tower ]; then
  [ -n "$DRY" ] || { exec 9>qt1.lock; flock 9; }      # blocking: waits for the old queue job holding it
  CARD=1; PY=~/miniforge3/envs/titans/bin/python; L=~/work/iclr/bench30/jobs_tower.log; E=e_dp; W=w_dp; M=m_dp
  A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
  KW="1:8:1500"; XC=""
else
  [ -n "$DRY" ] || { exec 9>q135w.lock; flock -n 9 || { echo "card 3 weekend queue already running"; exit 1; }; }
  CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log; E=e_kv; W=w_kv; M=m_kv
  A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
  KW="1:8:1500"; XC=",topk"
fi
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
[ -n "$DRY" ] && { L=/dev/stdout; run() { echo "DRY run $2 :: ${3:0:60}..."; }; }
gap() { python3 -c "
import json,sys
try:
    d=json.load(open('runs/$1_summary.json'))['_all']; print(int(d['sup']-d['omem']>=10))
except Exception: print(0)"; }
DN=evalsets/b30_dev_4k_1ep_narr.jsonl; D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl
EXT=evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl; B11=evalsets/bl1120_4k_reconstructed.jsonl
BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
echo "QT-WEEKEND-START $(date +%F_%T) $BOX" >> $L
[ -n "$DRY" ] || while pgrep -u "$USER" -f "train_b30.py" > /dev/null; do sleep 60; done     # the chunk job of the old queue
prev=runs/$E.pt; ok=0; last=""
for c in $(seq 1 15); do
  pt=runs/${E}_c$c.pt
  if [ ! -f $pt ]; then
    B30_NEP=1:1 B30_FORMS=narration B30_BL_FRAC=0 run 14400 ${E}_c$c "$A --oracle-write --oracle-distract 1:1:1 --preserve 1.0 --init $prev --seed $c --steps 1000 --warmup 10 --eval-every 0 --out runs/${E}_c$c"
  fi
  [ -f $pt ] || continue
  prev=$pt; last=$pt
  [ -f runs/dge_${E#e_}_c${c}_b30_dev_4k_1ep_narr_summary.json ] || run 7200 dge_${E#e_}_c$c "$A --init $pt --jsonl-eval $DN --jsonl-conds sup,orc,mem,omem$XC --out runs/dge_${E#e_}_c$c"
  if [ "$(gap dge_${E#e_}_c${c}_b30_dev_4k_1ep_narr)" = 1 ]; then ok=1; echo "GATE $(date +%F_%T) ${E}_c$c reads on narration (sup - omem >= 10)" >> $L; break; fi
done
FINAL=$last
if [ $ok = 1 ]; then
  [ -f runs/$W.pt ] || B30_NEP=1:0.7,2:0.3 run 18000 $W "$A --oracle-write --oracle-distract $KW --preserve 1.0 --init $last --steps 2000 --eval-every 0 --out runs/$W"
  if [ -f runs/$W.pt ]; then
    FINAL=runs/$W.pt
    [ -f runs/dg_${W}_b30_dev_4k_1ep_sp_summary.json ] || run 7200 dg_$W "$A --init runs/$W.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem$XC --out runs/dg_$W"
    [ -f runs/$M.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 18000 $M "$A --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/$W.pt --steps 2000 --eval-every 0 --out runs/$M"
    [ -f runs/$M.pt ] && FINAL=runs/$M.pt
  fi
else
  echo "GATE $(date +%F_%T) $E never reached sup - omem >= 10 in 15 chunks; final evals on $last" >> $L
fi
if [ -n "$FINAL" ]; then
  N=$(basename $FINAL .pt)
  echo "FINAL-MODEL $(date +%F_%T) $FINAL" >> $L
  run 7200 fin1_$N "$A --init $FINAL --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem$XC --out runs/fin1_$N"
  run 10800 fin2_$N "$A --init $FINAL --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$N"
  run 10800 finbl_$N "$A --init $FINAL --official $BL10 --official-conds mem,omem,orc --out runs/finbl_$N"
  run 10800 fin11_$N "$A --init $FINAL --jsonl-eval $B11 --jsonl-n 500 --jsonl-conds mem,omem --out runs/fin11_$N"
  run 14400 finext_$N "$A --init $FINAL --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/finext_$N"
fi
echo "QT-WEEKEND-FINI $(date +%F_%T) $BOX" >> $L
