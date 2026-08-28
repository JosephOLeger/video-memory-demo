#!/usr/bin/env bash
# Sondes A (lecture au rejeu) et B (trace mecanisme), 2026-08-26.
# Predictions AVANT lancement : results/predictions_sondes.md.
# Sequentiel sur UNE carte (journee, colleagues sur la box) — les deux sont
# legeres (B : 3 runs courts ; A : no_grad, 5 artefacts x 4 gaps).
set -u
CARD=${1:-3}
cd "$HOME/work/titans-mac" || exit 1
PY=$HOME/venvs/titans/bin/python
mkdir -p results/d2interne results/readout logs

run_b() { # name theta seed steps
  local name=$1 th=$2 seed=$3 steps=$4
  local out=results/d2interne/${name}.jsonl
  if [ -s "$out" ] && grep -q '"summary"' "$out"; then echo "skip $name"; return 0; fi
  echo "--- sonde B $name $(date +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$CARD timeout 14400 nice -n 10 "$PY" scripts/diag_d2b_train.py \
    --label "$name" --train cache/egolife_pca128_train.pt \
    --val cache/egolife_pca128_val.pt \
    --horizon 24 --seg 8 --batch 8 --steps "$steps" --seed "$seed" \
    --muon rescale --theta-scale "$th" --alpha-init 0.002 --eta-init 0.995 \
    --diag-out "$out" > "logs/sondeB_${name}.log" 2>&1
  echo "rc=$? $(tail -n 1 "logs/sondeB_${name}.log")"
}

run_b b_t008_s0 0.08 0 300
run_b b_t008_s1 0.08 1 300
run_b b_t005_s0 0.05 0 300

echo "--- sonde A $(date +%H:%M:%S) ---"
CUDA_VISIBLE_DEVICES=$CARD timeout 14400 nice -n 10 "$PY" scripts/diag_readout_probe.py \
  --frozen "frozen_muon/mu_rescale_s[0-9].pt" \
  --val cache/egolife_pca128_val.pt \
  --horizon 24 --gaps 0,2,4,8 --n-eval 96 --batch 8 \
  --diag-out results/readout/readout_probe_rescale.jsonl \
  > logs/sondeA_rescale.log 2>&1
echo "rc=$? $(tail -n 1 logs/sondeA_rescale.log)"
echo "=== sondes finies $(date +%F' '%H:%M:%S) ==="
