#!/bin/bash
# Run each line of $1 ("name command...") on the first card with >= 12 GB free, one after the
# other (each job waits for the previous launch to settle 120 s). Logs in logs/<name>.log.
cd ~/work/iclr/graft_v3
export HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/work/titans-mac/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/titans/bin/python
while read -r name args; do
  [ -z "$name" ] && continue
  while true; do
    card=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>=12288 {print \$1; exit}")
    [ -n "$card" ] && break
    sleep 30
  done
  CUDA_VISIBLE_DEVICES=$card nohup $PY $args > logs/$name.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $name on card $card pid $!" >> logs/queue.log
  sleep 120
done < "$1"
echo "$(date +%T) queue empty" >> logs/queue.log
