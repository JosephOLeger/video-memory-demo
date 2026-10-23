#!/bin/bash
# Launch each job line of $1 on the first card with >= 12.5 GB free; re-check after each launch.
PY=~/miniforge3/envs/titans/bin/python
cd ~/work/iclr/ladder
while read -r name args; do
  [ -z "$name" ] && continue
  while true; do
    card=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>12800{print \$1; exit}")
    [ -n "$card" ] && break
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=$card nohup $PY ladder.py $args > runs/$name.log 2>&1 &
  echo "$(date +%T) launched $name on card $card pid $!" >> runs/queue.log
  sleep 150
done < "$1"
echo "$(date +%T) queue empty" >> runs/queue.log
