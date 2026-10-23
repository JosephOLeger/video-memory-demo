#!/bin/bash
CARD=$1
PY=~/miniforge3/envs/titans/bin/python; REPO=~/work/iclr/titans-mac; OUT=~/work/iclr
Q=$OUT/cap_queue_119.txt; LK=$OUT/cap_queue_119.lock
pop(){ exec 9>"$LK"; flock 9; local l=$(head -n1 "$Q" 2>/dev/null); if [ -n "$l" ]; then tail -n +2 "$Q" > "$Q.t" && mv "$Q.t" "$Q"; fi; flock -u 9; echo "$l"; }
while :; do
  cell=$(pop); [ -z "$cell" ] && { echo "[card $CARD] queue empty $(date +%T)"; break; }
  IFS=: read n rc s <<< "$cell"
  echo "[card $CARD] START n=$n rc=$rc s=$s $(date +%T)"
  cd "$REPO"; PYTHONPATH=src CUDA_VISIBLE_DEVICES=$CARD $PY $OUT/recall_load_one.py --n $n --rc $rc --seed $s --lr 5e-4 --maxsteps 30000 --patience 25 --tag cap
  echo "[card $CARD] END n=$n rc=$rc s=$s $(date +%T)"
done
