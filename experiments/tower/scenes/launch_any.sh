#!/bin/bash
# usage: launch_any.sh LOG cmd...   waits until SOME card in ${CARDS:-0 1 2} has >= MINFREE MiB free, then runs cmd on it.
# A lock serialises my own launches (hold 90 s after start so the model is resident before the next check).
LOG=$1; shift
exec 9>~/work/iclr/scenes/.launch.lock
flock 9
while true; do
  for c in ${CARDS:-0 1 2}; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $c)
    if [ "$free" -ge ${MINFREE:-12000} ]; then card=$c; break 2; fi
  done
  sleep 20
done
echo "launch at $(date) card=$card free=$free" > "$LOG"
CUDA_VISIBLE_DEVICES=$card "$@" >> "$LOG" 2>&1 9>&- &
pid=$!
sleep 90
flock -u 9
exec 9>&-
wait $pid
