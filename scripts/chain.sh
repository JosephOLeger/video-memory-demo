#!/bin/bash
# Run a multi-phase study from a single cron entry, one phase at a time.
#
#   */10 * * * * /bin/bash $HOME/work/titans-mac/scripts/chain.sh
#
# Phases differ in what they need: echo cells are ~6.4 GiB and want two per card,
# the margin probe is a single fast pass, the length probe is light but long.
# Encoding that here rather than swapping the crontab between phases means an
# unattended handover cannot be missed, and the guard cannot be left at the previous
# phase's value -- the mistake that cost 74 jobs on 2026-07-29 and 22 on 07-31.
#
# **A phase starts only when the previous one's jobs are actually gone**, not when
# its queue empties. Gating on "queue empty" starts the next phase while the last
# cells still hold cards, which is what caused the in-flight OOMs on 07-29.
cd ~/work/titans-mac || exit 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=~/venvs/titans/bin/python
VAL=cache/egolife_pca128_val.pt
LOG=logs/chain.log
mkdir -p logs frozen_vv

# --- phase 1: the V;V sweep (heavy, two per card) ----------------------------
# Guard 11000: these cells peak at ~6.4 GiB, so a card takes two and refuses a
# third (free falls to ~6.6 GiB). It also keeps workers off a card another user
# is holding.
if [ -s jobs_vv.txt ]; then
  JOBS=jobs_vv.txt GUARD=11000 PREFIX=V SETTLE=120 TMO=21600 WORKERS=8 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[w]orldmodel.echo" >/dev/null; then
  exit 0                      # queue drained but cells still running
fi

# --- phase 2: stability margin on every artifact, and queue the length probe --
# One pass over all of them; seconds each, no training. Writing the marker file
# last makes this idempotent under a cron that fires every ten minutes.
if [ ! -f vv_margins.json ] && ls frozen_vv/*.pt >/dev/null 2>&1; then
  echo "$(date '+%F %T') chain: phase 2, marges" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 $P -m titans.worldmodel.margin \
      --frozen 'frozen_vv/*.pt' --val "$VAL" --out vv_margins.json >> "$LOG" 2>&1

  # One length probe per arm (not per seed): the drift is an architecture property
  # and 46 of them would cost more than the sweep that produced them.
  : > jobs_vvlong.txt
  for f in frozen_vv/*_s0.pt; do
    [ -e "$f" ] || continue
    lab=$(basename "$f" _s0.pt)
    echo "$P -m titans.worldmodel.long_run --frozen $f --val $VAL --reps 1000 --pool 4 --diag-every 10 --label $lab --out vvlong_${lab}.json" >> jobs_vvlong.txt
  done
  echo "$(date '+%F %T') chain: $(wc -l < jobs_vvlong.txt) sondes de longueur en file" >> "$LOG"
  exit 0
fi

# --- phase 3: length probe (light, forward-only) -----------------------------
if [ -s jobs_vvlong.txt ]; then
  JOBS=jobs_vvlong.txt GUARD=6000 PREFIX=L SETTLE=30 TMO=21600 WORKERS=4 \
    bash "$HERE/keepalive.sh"
fi
