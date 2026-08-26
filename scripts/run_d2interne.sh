#!/usr/bin/env bash
# Audit D2 interne : echelle theta_scale sur le config famille (sane_ns).
# Predictions AVANT lancement : results/predictions_d2interne.md.
# Carte en argument 1. Sequentiel, ~3h30 au pire (les morts ecourtent).
set -u
CARD=${1:-0}
cd "$HOME/work/titans-mac" || exit 1
PY=$HOME/venvs/titans/bin/python
mkdir -p results/d2interne logs

run_one() { # name muon theta seed
  local name=$1 muon=$2 th=$3 seed=$4
  local out=results/d2interne/${name}.jsonl
  if [ -s "$out" ] && grep -q '"summary"' "$out"; then echo "skip $name"; return 0; fi
  echo "--- $name $(date +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$CARD timeout 7200 nice -n 10 "$PY" scripts/diag_d2_train.py \
    --label "$name" --train cache/egolife_pca128_train.pt \
    --val cache/egolife_pca128_val.pt \
    --horizon 24 --seg 8 --batch 8 --steps 1500 --seed "$seed" \
    --muon "$muon" --theta-scale "$th" --alpha-init 0.002 --eta-init 0.995 \
    --diag-out "$out" > "logs/d2interne_${name}.log" 2>&1
  echo "rc=$? $(tail -n 1 "logs/d2interne_${name}.log")"
}

run_one ctrl_s0 rescale 0.03 0
run_one t03_s0  rescale 0.3  0
run_one t03_s1  rescale 0.3  1
run_one t10_s0  rescale 1.0  0
run_one t10_s1  rescale 1.0  1
run_one u10_s0  unit    1.0  0
echo "=== D2 interne fini $(date +%F' '%H:%M:%S) ==="
