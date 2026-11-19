#!/usr/bin/env bash
# Dose de c au rejeu sur les usq10k geles (P-CD-1) : c natif 350 deja
# sonde (M_usq10k.jsonl) ; on ajoute 35 / 87.5 / 1400 / 3500.
# Co-schedule nice sur la carte 3.
set -u
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG=logs/cdose.log
: > "$LOG"
echo "depart cdose $(date +%H:%M:%S)" >> "$LOG"
for c in 35 87.5 1400 3500; do
  echo "--- c=$c ---" >> "$LOG"
  CLIP_OVERRIDE=$c CUDA_VISIBLE_DEVICES=3 nice -n 15 timeout 10800 "$PY" \
    scripts/diag_cdose_probe.py --val cache/egolife_pca128_val.pt \
    --frozen "frozen_muon/mu_usq10k_s?.pt" --mode cycle --clips 100 \
    --diag-out "results/prlaw3/CD_usq_c${c}.jsonl" >> "$LOG" 2>&1 \
    || echo "c=$c ECHEC" >> "$LOG"
done
echo "CDOSE FINI $(date)" >> "$LOG"
