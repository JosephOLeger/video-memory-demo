#!/usr/bin/env bash
# Retention avec la regle pr-absolu sur les bras 10k (jamais scores) —
# P-ABS-1 : les graines unit10k collapsees doivent etre taguees des le
# gap 0. Co-schedule sur carte 3 (nice 15) une fois usq30k en route.
set -u
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG=logs/ret10k.log
: > "$LOG"
until pgrep -f "label usq30k" > /dev/null; do sleep 120; done
sleep 300
for arm in mu_unit10k mu_usq10k mu_unit10k_dec; do
  echo "=== retention $arm $(date +%H:%M) ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=3 nice -n 15 timeout 14400 "$PY" -m titans.worldmodel.retention \
    --frozen "frozen_muon/${arm}_s[0-9].pt" --val cache/egolife_pca128_val.pt \
    --gaps 0,1,2,4,8,16,32 --n-eval 96 --batch 8 \
    --out "retention_${arm}_absrule.json" >> "$LOG" 2>&1 \
    || echo "$arm ECHEC" >> "$LOG"
done
echo "RET10K FINI $(date)" >> "$LOG"
