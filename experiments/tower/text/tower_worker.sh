#!/bin/bash
# Tower worker for the text A/B hidden sweep. Uses a card only if it has >= 16 GB free, under the
# per-card lock /var/lock/cardN.lock held during check + launch + model load (3 min). Never touches
# other processes. Jobs: tower_jobs.txt (one command per line, {CARD} replaced by the card index).
cd ~/work/iclr/text
PY=~/miniforge3/envs/titans/bin/python
while true; do
  job=$(grep -v "^#" tower_jobs.txt | grep -v -F -x -f <(cat tower_jobs.started 2>/dev/null) | head -1)
  [ -z "$job" ] && break
  placed=0
  for c in 0 1 2; do
    exec 9</var/lock/card$c.lock
    if flock -n 9; then
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $c)
      if [ "$f" -ge 16000 ]; then
        echo "$job" >> tower_jobs.started
        cmd=${job//\{CARD\}/$c}
        echo "START $(date +%F_%T) card$c free=$f $cmd" >> tower_jobs.log
        setsid bash -c "$cmd && echo \"OK \$(date +%T) $cmd\" >> tower_jobs.log || echo \"FAIL \$(date +%T) $cmd\" >> tower_jobs.log" < /dev/null > /dev/null 2>&1 &
        sleep 180
        placed=1
      fi
      flock -u 9
    fi
    exec 9>&-
    [ $placed = 1 ] && break
  done
  [ $placed = 0 ] && sleep 120
done
echo "TOWER-WORKER-FINI $(date +%T)" >> tower_jobs.log
