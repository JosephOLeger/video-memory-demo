#!/bin/bash
# Run queues in order from a single cron entry, one phase at a time.
#
#   */10 * * * * /bin/bash $HOME/work/titans-mac/scripts/chain.sh
#
# Phases differ in what they need: MQAR cells are under a GiB and want many
# workers, echo runs are ~6.4 GiB and want two per card. Encoding that here rather
# than swapping the crontab between phases means an unattended handover cannot be
# missed, and the guard cannot be left at the previous phase's value -- which is
# the mistake that cost 74 jobs on 2026-07-29 and 22 on 07-31.
#
# **A phase starts only when the previous one's jobs are actually gone**, not when
# its queue empties. Gating on "queue empty" starts heavy jobs while light ones are
# still holding cards, which is what caused the in-flight OOMs on 07-29.
cd ~/work/titans-mac || exit 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- phase 1: MQAR cells (small, many workers) -------------------------------
if [ -s jobs_theta.txt ]; then
  JOBS=jobs_theta.txt GUARD=3000 PREFIX=T SETTLE=20 TMO=3600 WORKERS=12 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[n]an_sweep" >/dev/null; then
  exit 0                      # queue drained but cells still running
fi

# --- phase 2: echo runs (heavy, two per card) --------------------------------
# Guard 11000 from titans.worldmodel.measure_vram: these arms peak at 6.37 GiB, so
# a card takes two and refuses a third (free falls to ~7.6 GiB).
if [ -s jobs_echo.txt ]; then
  JOBS=jobs_echo.txt GUARD=11000 PREFIX=E SETTLE=120 TMO=21600 WORKERS=8 \
    bash "$HERE/keepalive.sh"
fi
