#!/usr/bin/env bash
# Echelle 2 de l'audit D2 interne (predictions_d2interne.md, section
# "Echelle 2", enregistrees AVANT). Trois volets :
#   A. arbitre externe : echo.py NON MODIFIE, 200 pas, theta 0.3 et 0.03
#   B. echelle fine theta {0.05, 0.08, 0.12, 0.2} x graines {0,1}, 1500 pas
#   C. controles longs theta 0.03, graines 1-2, 4000 pas (budget famille)
# Attend le drainage de run_d2interne.sh (echelle 1) avant de toucher la carte.
set -u
CARD=${1:-2}
cd "$HOME/work/titans-mac" || exit 1
PY=$HOME/venvs/titans/bin/python
mkdir -p results/d2interne logs

while pgrep -f "run_d2interne\.sh" > /dev/null; do sleep 120; done
echo "=== echelle 1 drainee, depart echelle 2 $(date +%H:%M:%S) ==="

# --- Volet A : arbitre (harnais hors-jeu) -----------------------------------
arbitre() { # name theta
  local name=$1 th=$2
  local out=results/d2interne/arbitre_${name}.json
  [ -s "$out" ] && { echo "skip arbitre $name"; return 0; }
  echo "--- arbitre $name $(date +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$CARD timeout 3600 nice -n 10 "$PY" -m titans.worldmodel.echo \
    --label "arb_${name}" --train cache/egolife_pca128_train.pt \
    --val cache/egolife_pca128_val.pt \
    --horizon 24 --seg 8 --batch 8 --steps 200 --seed 0 \
    --muon rescale --theta-scale "$th" --alpha-init 0.002 --eta-init 0.995 \
    --out "$out" > "logs/d2interne_arb_${name}.log" 2>&1
  echo "rc=$? $(tail -n 2 "logs/d2interne_arb_${name}.log" | head -n 1)"
}
arbitre t03 0.3
arbitre ctrl 0.03

# --- Volets B et C : harnais forensique -------------------------------------
run_one() { # name theta seed steps
  local name=$1 th=$2 seed=$3 steps=$4
  local out=results/d2interne/${name}.jsonl
  if [ -s "$out" ] && grep -q '"summary"' "$out"; then echo "skip $name"; return 0; fi
  echo "--- $name $(date +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$CARD timeout 14400 nice -n 10 "$PY" scripts/diag_d2_train.py \
    --label "$name" --train cache/egolife_pca128_train.pt \
    --val cache/egolife_pca128_val.pt \
    --horizon 24 --seg 8 --batch 8 --steps "$steps" --seed "$seed" \
    --muon rescale --theta-scale "$th" --alpha-init 0.002 --eta-init 0.995 \
    --diag-out "$out" > "logs/d2interne_${name}.log" 2>&1
  echo "rc=$? $(tail -n 1 "logs/d2interne_${name}.log")"
}

for th in 0.05 0.08 0.12 0.2; do
  for seed in 0 1; do
    run_one "fine_t$(echo $th | tr -d .)_s${seed}" "$th" "$seed" 1500
  done
done
run_one long003_s1 0.03 1 4000
run_one long003_s2 0.03 2 4000
echo "=== echelle 2 finie $(date +%F' '%H:%M:%S) ==="
