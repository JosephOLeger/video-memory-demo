#!/bin/bash
# Mirror the box's results locally, then regenerate the readable report folder.
#
#   bash scripts/pull_report.sh
#   */30 * * * * /bin/bash $HOME/projects/titans-mac/scripts/pull_report.sh
#
# Two things this fixes over the previous version, which lived in a session
# scratchpad under /tmp:
#
# * it survives the session that wrote it -- a report generator that disappears
#   when a conversation ends is a report generator that silently stops;
# * it pulls **every** result prefix, from one list shared with `report.py`.
#   The old copy globbed a hand-written set that predated the 10k rerun (`K_*`)
#   and the full-attention baseline (`F_*`), so those two studies finished and
#   stayed invisible in the report for a day.
#
# The mirror doubles as the offline copy of the ~1000 result files, so analysis
# does not need the box to be reachable.
set -u
BOX=josephleger@172.31.4.135:work/titans-mac
RAW=${RAW:-$HOME/projects/titans-mac/results/box}
OUT=${OUT:-"/mnt/c/Users/josep/Desktop/Césure/ASTAR/Résultats"}
PY=${PY:-$HOME/miniforge3/envs/ai/bin/python}
SRC=$HOME/projects/titans-mac/src
mkdir -p "$RAW" "$OUT"

# One rsync for every *.json under the repo root, rather than a list of globs that
# has to be edited whenever a study is added. --prune-empty-dirs keeps the mirror
# from filling with the box's empty directory tree.
rsync -az -e ssh --prune-empty-dirs \
  --include='*/' --include='*.json' --exclude='.venv/***' --exclude='*' \
  "$BOX/" "$RAW/" 2>/dev/null || true

# One list for the retention curves, which arrive as one file per arm.
$PY - "$RAW" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(root, "retention_out", "*.json"))):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    rows += r if isinstance(r, list) else [r]
if rows:
    json.dump(rows, open(os.path.join(root, "retention_all.json"), "w"))
    print("retention:", len(rows), "artefacts")
PYEOF

if [ -f "$RAW/retention_all.json" ]; then
  PYTHONPATH=$SRC $PY -m titans.worldmodel.plot_retention \
      --json "$RAW/retention_all.json" --out-dir "$OUT" 2>&1 | tail -20
fi
PYTHONPATH=$SRC $PY -m titans.worldmodel.report --root "$RAW" --out-dir "$OUT"
