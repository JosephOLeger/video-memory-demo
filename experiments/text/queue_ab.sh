#!/bin/bash
# card 3 ONLY. Runs jobs from queue_ab.txt one by one (first line not yet started), gate >= 20 GB free.
cd ~/work/iclr/text
PY=~/work/titans-mac/.venv/bin/python
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
while true; do
  job=$(grep -v "^#" queue_ab.txt | grep -v -F -x -f <(cat queue_ab.started 2>/dev/null) | head -1)
  [ -z "$job" ] && break
  echo "$job" >> queue_ab.started
  gate
  echo "START $(date +%F_%T) $job" >> queue_ab.log
  bash -c "$job" && echo "OK $(date +%T) $job" >> queue_ab.log || echo "FAIL $(date +%T) $job" >> queue_ab.log
done
echo "QUEUE-FINI $(date +%T)" >> queue_ab.log
