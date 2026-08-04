#!/bin/bash
# Pool worker: memory guard + requeue-on-failure + per-job timeout.
#
# pool4 differs from pool3 in one place: the free-VRAM check is a check-then-act
# race, and pool3 lost it. At 14:46 on 2026-08-03 all eight workers polled a cold
# pool in the same second, every one of them read the card as fully free, and the
# two on gpu3 both launched cas_tag (11.6 + 10.6 GiB on a 22.18 GiB card -> OOM).
# The guard was sized correctly; it simply could not see an allocation that had
# not happened yet.
#
# Fix: hold a per-card lock across "read free -> pop job -> let it allocate", so
# the second worker on a card reads a post-allocation number. SETTLE covers job
# startup (import torch, move the 1.4 GB cache to the card, first backward pass);
# it costs at most SETTLE seconds per job start against jobs that run for hours.
#
#   usage: pool4.sh <jobs-file> <gpu> <worker-id> <need-MiB> <job-timeout-s>
JOBS=$1; GPU=$2; WID=$3; NEED=${4:-8000}; TMO=${5:-14400}
SETTLE=${SETTLE:-120}
LOCK="${JOBS}.lock"
cd ~/work/titans-mac || exit 1
LOG=~/work/titans-mac/logs/pool/w${GPU}_${WID}.log
GLOCK=~/work/titans-mac/logs/pool/gpu${GPU}.lock
mkdir -p "$(dirname "$LOG")"; : > "$GLOCK" 2>/dev/null
while true; do
  [ -s "$JOBS" ] || { echo "[w${GPU}_${WID}] pool vide $(date +%H:%M)" >> "$LOG"; break; }

  exec 9>"$GLOCK"; flock 9                 # serialise spin-up on this card
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "${free:-0}" -lt "$NEED" ]; then exec 9>&-; sleep 120; continue; fi
  job=$(flock "$LOCK" -c "head -1 '$JOBS'; sed -i '1d' '$JOBS'")
  if [ -z "$job" ]; then
    exec 9>&-; echo "[w${GPU}_${WID}] pool vide $(date +%H:%M)" >> "$LOG"; break
  fi
  retried=$(printf '%s' "$job" | grep -c "##RETRY##")
  job=${job/ \#\#RETRY\#\#/}
  echo "[w${GPU}_${WID}] START $(date +%H:%M) free=${free} :: $job" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU timeout "$TMO" bash -c "$job" >> "$LOG" 2>&1 &
  pid=$!
  sleep "$SETTLE"                          # let it allocate before anyone re-reads free
  exec 9>&-                                # release the card lock
  wait "$pid"; rc=$?

  echo "[w${GPU}_${WID}] END rc=$rc $(date +%H:%M)" >> "$LOG"
  if [ $rc -ne 0 ]; then
    if [ "$retried" = "0" ]; then
      flock "$LOCK" -c "echo '$job ##RETRY##' >> '$JOBS'"
      echo "[w${GPU}_${WID}] REQUEUE (rc=$rc)" >> "$LOG"
    else
      echo "[w${GPU}_${WID}] DROP after 2nd failure (rc=$rc)" >> "$LOG"
    fi
    sleep 30
  fi
done
