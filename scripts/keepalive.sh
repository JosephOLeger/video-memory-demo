#!/bin/bash
# Keep a pool of workers alive on a shared multi-GPU box, from cron.
#
#   crontab:  */10 * * * * /bin/bash $HOME/work/titans-mac/scripts/keepalive.sh
#
# Why cron and not one long-lived launcher: workers die -- a dropped ssh session,
# an OOM-killed shell, a card taken by another user. Re-running this every ten
# minutes restores exactly the workers that are missing and is a no-op otherwise,
# so a sweep survives unattended nights. Each worker exits by itself when the queue
# empties, and this script then stops relaunching it (`[ -s $JOBS ]`).
#
# Configure per sweep:
JOBS=${JOBS:-jobs.txt}      # queue file, one shell command per line
GUARD=${GUARD:-13000}       # MiB free a card must have before a worker takes a job
TMO=${TMO:-21600}           # per-job timeout in seconds
PREFIX=${PREFIX:-W}         # worker-id prefix; CHANGE IT when replacing a running
                            # pool, or the pgrep below thinks the old ones are new
#
# **Set GUARD from titans.worldmodel.measure_vram, on the heaviest arm.** The arms
# differ by ~50% and a guard sized on the lightest one puts two heavy jobs on a card:
# that cost 74 jobs on 2026-07-29 and 22 on 07-31. Prefer a guard that also exceeds
# what remains free *after* one heavy job, so a card never takes two of them.

cd ~/work/titans-mac || exit 0
mkdir -p logs/pool ckpt
L=logs/pool/keepalive.log
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

launch() {  # $1 jobsfile  $2 gpu  $3 wid  $4 needMiB  $5 timeout
  if ! pgrep -f "[p]ool4.sh $1 $2 $3 " >/dev/null; then
    nohup bash "$HERE/pool4.sh" "$1" "$2" "$3" "$4" "$5" >>"$L" 2>&1 &
    echo "$(date '+%F %T') keepalive: launched $3 on gpu$2 ($1)" >>"$L"
  fi
}

if [ -s "$JOBS" ]; then
  # two workers per card: one job rarely saturates an A5000, and the guard plus
  # pool4's per-card lock decide whether the second one actually gets to start.
  gpus=(0 1 2 3 0 1 2 3)
  for j in 0 1 2 3 4 5 6 7; do
    launch "$JOBS" "${gpus[$j]}" "${PREFIX}${j}" "$GUARD" "$TMO"
  done
fi
