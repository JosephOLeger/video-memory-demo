#!/usr/bin/env bash
# Volet R2 : le levier de magnitude transplante dans lucidrains.
# R2a : sonde inference-only sur ckpt_prlaw_final (patch au rejeu).
# R2b : reentrainement 4000 pas avec la regle active, puis sonde.
# Attend la fin du runner c2 (paire stride) pour la carte 2.
# Predictions P-R2a / P-R2b (predictions_collapse_2026-08-28.md).
set -u
cd "$HOME/repro/titans-pytorch" || exit 1
PY="$HOME/venvs/repro/bin/python"
OUT="$HOME/work/titans-mac/results/prlaw3"
LOG="$HOME/work/titans-mac/logs/psic.log"
: > "$LOG"
echo "attente fin runner c2 $(date +%H:%M:%S)" >> "$LOG"

echo "carte 2 libre $(date +%H:%M:%S)" >> "$LOG"
echo "--- R2a sonde inference ---" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 nice -n 10 timeout 14400 "$PY" probe_r_psic.py \
  --ckpt ckpt_prlaw_final.pt --seqs 4 --seq-len 4096 --max-lag 24 \
  --diag-out "$OUT/R2a_psic_inf.jsonl" >> "$LOG" 2>&1 \
  && echo "R2a OK" >> "$LOG" || echo "R2a ECHEC" >> "$LOG"
echo "--- R2b entrainement ---" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 nice -n 10 timeout 172800 "$PY" train_r_psic.py \
  >> "$LOG" 2>&1 \
  && echo "R2b train OK" >> "$LOG" || echo "R2b train ECHEC" >> "$LOG"
echo "--- R2b sonde ---" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 nice -n 10 timeout 14400 "$PY" probe_r_psic.py \
  --ckpt ckpt_psic_final.pt --seqs 4 --seq-len 4096 --max-lag 24 \
  --diag-out "$OUT/R2b_psic_trained.jsonl" >> "$LOG" 2>&1 \
  && echo "R2b sonde OK" >> "$LOG" || echo "R2b sonde ECHEC" >> "$LOG"
echo "PSIC FINI $(date)" >> "$LOG"
