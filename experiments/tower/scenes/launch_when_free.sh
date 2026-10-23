#!/bin/bash
# usage: launch_when_free.sh LOG cmd...   waits until card 1 has >= 11 GB free, then runs cmd
LOG=$1; shift
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  [ "$free" -ge ${MINFREE:-11000} ] && break
  sleep 20
done
echo "launch at $(date) free=$free" > "$LOG"
CUDA_VISIBLE_DEVICES=1 exec "$@" >> "$LOG" 2>&1
