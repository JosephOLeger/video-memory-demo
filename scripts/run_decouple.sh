#!/usr/bin/env bash
# Bras causal decouple_memory_update (P-C1/P-C2) — cartes 2 et 3.
# Config EXACTE de mu_unit10k, une graine par carte, en parallele.
set -u
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG=logs/decouple.log
: > "$LOG"

train() {
  s=$1; card=$2
  echo "--- mu_unit10k_dec_s${s} carte ${card} $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 28800 "$PY" \
    scripts/diag_decouple_train.py --label "mu_unit10k_dec_s${s}" \
    --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
    --horizon 24 --steps 10000 --seed "$s" --eta-init 0.995 \
    --theta-scale 0.03 --alpha-init 0.002 --muon unit --frozen-every 2000 \
    --out "mu_unit10k_dec_s${s}.json" \
    --frozen-out "frozen_muon/mu_unit10k_dec_s${s}.pt" \
    > "logs/decouple_s${s}.log" 2>&1 \
    && echo "s${s} OK $(date +%H:%M:%S)" >> "$LOG" \
    || echo "s${s} ECHEC" >> "$LOG"
}

train 0 2 &
train 1 3 &
wait
# la sonde de loi sur les artefacts decouple, des qu'ils existent
CUDA_VISIBLE_DEVICES=2 nice -n 10 timeout 7200 "$PY" \
  scripts/diag_pr_collapse_probe.py --val cache/egolife_pca128_val.pt \
  --frozen "frozen_muon/mu_unit10k_dec_s[01].pt" --mode cycle --clips 100 \
  --diag-out results/prlaw/C_decouple.jsonl >> "$LOG" 2>&1 \
  || echo "sonde decouple ECHEC" >> "$LOG"
echo "DECOUPLE TOUT FINI $(date)" >> "$LOG"
