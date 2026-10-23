#!/bin/bash
# 23/10 : v3 = v2 avec le code fige d avant le correctif fps (src_old, fit_basis.py.bak_2210), recette identique a C1.
# on attend >= 10,5 Go libres avant de lancer (verrou + 4 min de chargement, contre la course
# verifier-puis-agir), et un run qui echoue (pas de TRAIN-STREAM-FINIE ou OOM au pas) est remis en file.
B=$HOME/work/iclr/baselines; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=$B/src_old:$B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2
CO=(--targets "$S/tgc/*.pt" --kinds color --basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
L=$1; WAITPID=$2
next() { ( flock 9; j=$(head -1 $B/jobs.txt); [ -n "$j" ] && sed -i 1d $B/jobs.txt; echo "$j" ) 9>$B/jobs.lock; }
requeue() { ( flock 9; echo "$1" >> $B/jobs.txt ) 9>$B/jobs.lock; }
free2() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2; }
[ -n "$WAITPID" ] && while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
while true; do
  j=$(next); [ -z "$j" ] && break; set -- $j; t=$1 s=$2
  exec 8>$B/gpu.lock; flock 8
  while [ "$(free2)" -lt 10500 ]; do sleep 60; done
  echo "$(date +%F_%T) lane $L start $t s$s (libre $(free2) Mo)" >> $B/queue.log
  [ -f $B/${t}_s$s.log ] && mv $B/${t}_s$s.log $B/${t}_s$s.fail$(date +%H%M).log
  $PY -m bl --mem-type $t "${CO[@]}" --seed $s --out $B/${t}_s$s > $B/${t}_s$s.log 2>&1 &
  pid=$!; sleep 240; flock -u 8
  wait $pid; c=$?
  if grep -q TRAIN-STREAM-FINIE $B/${t}_s$s.log && ! grep -q "OOM au pas" $B/${t}_s$s.log; then
    echo "$(date +%F_%T) lane $L fin $t s$s code $c" >> $B/queue.log
  else
    echo "$(date +%F_%T) lane $L ECHEC $t s$s code $c -> remis en file" >> $B/queue.log
    requeue "$t $s"; sleep 300
  fi
done
echo "$(date +%F_%T) lane $L file vide" >> $B/queue.log
