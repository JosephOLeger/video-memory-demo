#!/bin/bash
# Everything Josep should be able to read without a shell, regenerated on a timer.
#
#   */30 * * * * /bin/bash $HOME/projects/titans-mac/scripts/nightly_report.sh
#
# Three steps, each tolerant of the others failing: mirror the box, render the CSVs
# and the retention curve, render the trade-off figure. A study that has not finished
# drops out of the report rather than aborting it, so this is safe to run at any
# moment during a multi-hour queue.
#
# Figures land as PNGs **in the results folder itself**, not in a sibling: the folder
# is the deliverable, and a chart that lives one directory away from its numbers is a
# chart nobody opens.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
OUT=${OUT:-"/mnt/c/Users/josep/Desktop/Césure/ASTAR/Résultats"}
RAW=${RAW:-$REPO/results/box}
PY=${PY:-$HOME/miniforge3/envs/ai/bin/python}
LOG=$REPO/logs/nightly.log
mkdir -p "$OUT" "$REPO/logs"

{
  echo "===== $(date '+%F %T')"
  OUT="$OUT" RAW="$RAW" PY="$PY" bash "$HERE/pull_report.sh" 2>&1 | tail -12
  PYTHONPATH=$REPO/src $PY -m titans.worldmodel.plot_tradeoff \
      --root "$RAW" --out-dir "$OUT" 2>&1 | tail -3
  PYTHONPATH=$REPO/src $PY "$HERE/status_md.py" --root "$RAW" --out "$OUT/ETAT.md" 2>&1 | tail -3
  # Hand-written notes that belong next to the numbers they predict -- a prediction
  # registered before a launch is worthless if it lives where nobody reads it.
  cp -f "$REPO"/results/*.md "$OUT"/ 2>/dev/null
} >> "$LOG" 2>&1
tail -20 "$LOG"
