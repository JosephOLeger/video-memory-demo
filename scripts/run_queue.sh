#!/usr/bin/env bash
# A plain job-queue worker: one line of JOBS per run, least-loaded card, N workers.
#
# Replaces the chained `run_X_after.sh` pattern, which died silently once: the
# runner waited on another script's PIDs and went with the shell that spawned it,
# leaving 14 queued jobs that never ran and no error anywhere. This one is started
# under setsid, takes work by atomically popping a line, and each worker is
# independent -- if one dies the rest keep going.
set -u
cd "$(dirname "$0")/.."
JOBS=${JOBS:-jobs_alpha.txt}
WORKERS=${WORKERS:-2}
LOCK=$JOBS.lock
PY=$HOME/venvs/titans/bin/python

pop() {   # atomically take the first line
  ( flock 9
    line=$(head -1 "$JOBS" 2>/dev/null) || return 1
    [ -z "$line" ] && return 1
    tail -n +2 "$JOBS" > "$JOBS.tmp" && mv "$JOBS.tmp" "$JOBS"
    printf '%s' "$line"
  ) 9>"$LOCK"
}

worker() {
  local id=$1
  while :; do
    job=$(pop); [ -z "$job" ] && { echo "[w$id] file vide, fin"; return; }
    card=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -1 | cut -d, -f1)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$card")
    if [ "$free" -lt 4000 ]; then          # never crowd a card someone else is on
      echo "[w$id] carte $card n'a que ${free} Mio libres, j'attends"; sleep 300
      printf '%s\n' "$job" | cat - "$JOBS" > "$JOBS.tmp" && mv "$JOBS.tmp" "$JOBS"
      continue
    fi
    label=$(printf '%s' "$job" | sed 's/.*--label \([^ ]*\).*/\1/')
    echo "[w$id] carte $card <- $label  ($(date +%H:%M:%S))"
    CUDA_VISIBLE_DEVICES=$card timeout 36000 bash -c "$job" \
      >> "logs/queue_${label}.log" 2>&1
    echo "[w$id] $label fini rc=$? ($(date +%H:%M:%S))"
  done
}

for i in $(seq 1 "$WORKERS"); do worker "$i" & done
wait
echo "=== queue vide ==="
