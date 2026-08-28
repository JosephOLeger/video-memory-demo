#!/usr/bin/env bash
# Re-ancrage bases saines 27/08 : echelle Titans-litterale + sonde lecture
# sur bras bornes. Predictions AVANT : results/predictions_sane_reanchor.md.
set -u
CARD=${1:-3}
cd "$HOME/work/titans-mac" || exit 1
PY=$HOME/venvs/titans/bin/python
mkdir -p results/titans_lit results/readout logs

run_t() { # name theta seed
  local name=$1 th=$2 seed=$3
  local out=results/titans_lit/${name}.jsonl
  if [ -s "$out" ] && grep -q '"summary"' "$out"; then echo "skip $name"; return 0; fi
  echo "--- titans-lit $name $(date +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$CARD timeout 14400 nice -n 10 "$PY" scripts/diag_titans_train.py \
    --label "$name" --train cache/egolife_pca128_train.pt \
    --val cache/egolife_pca128_val.pt \
    --horizon 24 --seg 8 --batch 8 --steps 1500 --seed "$seed" \
    --theta-scale "$th" --alpha-init 0.002 --eta-init 0.995 \
    --diag-out "$out" > "logs/titanslit_${name}.log" 2>&1
  echo "rc=$? $(tail -n 1 "logs/titanslit_${name}.log")"
}

for th in 0.03 0.05 0.08 0.12; do
  for seed in 0 1; do
    run_t "tl_t$(echo $th | tr -d .)_s${seed}" "$th" "$seed"
  done
done

for arm in mu_unit mu_usq10k mu_scl; do
  out=results/readout/readout_probe_${arm}.jsonl
  if [ -s "$out" ] && grep -q '"done"' "$out"; then echo "skip sondeA $arm"; continue; fi
  echo "--- sonde A $arm $(date +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$CARD timeout 14400 nice -n 10 "$PY" scripts/diag_readout_probe.py \
    --frozen "frozen_muon/${arm}_s[0-9].pt" \
    --val cache/egolife_pca128_val.pt \
    --horizon 24 --gaps 0,2,4,8 --n-eval 96 --batch 8 \
    --diag-out "$out" > "logs/sondeA_${arm}.log" 2>&1
  echo "rc=$? $(tail -n 1 "logs/sondeA_${arm}.log")"
done
echo "=== reanchor fini $(date +%F' '%H:%M:%S) ==="
