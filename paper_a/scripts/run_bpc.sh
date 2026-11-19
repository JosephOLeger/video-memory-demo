#!/usr/bin/env bash
# Ancrage bpc enwik8 : baseline vs reentraine psi_c. Attend la fin de
# la chaine psic2, puis evalue les deux checkpoints (carte 3).
set -u
cd "$HOME/repro/titans-pytorch" || exit 1
PY="$HOME/venvs/repro/bin/python"
OUT="$HOME/work/titans-mac/results/prlaw3"
LOG="$HOME/work/titans-mac/logs/bpc.log"
: > "$LOG"
while pgrep -f "run_psic2.sh" > /dev/null; do sleep 600; done
echo "--- bpc baseline $(date +%H:%M:%S) ---" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 7200 "$PY" eval_r_bpc.py \
  --ckpt ckpt_prlaw_final.pt --out "$OUT/bpc_baseline.json" \
  >> "$LOG" 2>&1 || echo "bpc baseline ECHEC" >> "$LOG"
echo "--- bpc psic2 $(date +%H:%M:%S) ---" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 7200 "$PY" eval_r_bpc.py \
  --ckpt ckpt_psic2_final.pt --psic --out "$OUT/bpc_psic2.json" \
  >> "$LOG" 2>&1 || echo "bpc psic2 ECHEC" >> "$LOG"
echo "BPC FINI $(date)" >> "$LOG"
