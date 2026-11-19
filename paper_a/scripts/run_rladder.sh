#!/usr/bin/env bash
# Baseline volet R : echelle de checkpoints lucidrains NON patchee
# (2000 / 3000 / final ; le 1000 existe : results_prlaw_r1000.jsonl).
# Reference obligatoire de R2a/R2b + completion P-R1 (T2 sur leur stack).
# Co-schedule nice 15 sur la carte 1 (dec30k y tourne, ~7 Go utilises).
set -u
cd "$HOME/repro/titans-pytorch" || exit 1
PY="$HOME/venvs/repro/bin/python"
OUT="$HOME/work/titans-mac/results/prlaw3"
LOG="$HOME/work/titans-mac/logs/rladder.log"
: > "$LOG"
for ck in 2000 3000 final; do
  echo "--- R_lad_$ck $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=1 nice -n 15 timeout 14400 "$PY" probe_r_prlaw.py \
    --ckpt "ckpt_prlaw_${ck}.pt" --seqs 4 --seq-len 4096 --max-lag 24 \
    --diag-out "$OUT/R_lad_${ck}.jsonl" >> "$LOG" 2>&1 \
    || echo "R_lad_$ck ECHEC" >> "$LOG"
done
echo "RLADDER FINI $(date)" >> "$LOG"
