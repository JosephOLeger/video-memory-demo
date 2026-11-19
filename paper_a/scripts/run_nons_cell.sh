#!/usr/bin/env bash
# Cellule manquante du 2x2 direction x magnitude (voir diag_nons_train.py) :
# direction BRUTE + magnitude usq apparie (psi_c, c=350). Si ce bras tient,
# la stabilite est portee par la regle de magnitude seule.
# Attend la fin d'unit30k (carte 0), puis 2 graines en sequence, puis sonde.
# Prediction P-2x2 (predictions_collapse_2026-08-28.md, enregistree avant).
set -u
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG=logs/nons_cell.log
: > "$LOG"
echo "attente unit30k $(date +%H:%M:%S)" >> "$LOG"
# motif ancre _s0 : le week-end lance unit30k_s1 sur la carte 1, il ne
# doit pas retenir cette chaine-ci.
until pgrep -f "label unit30k_s0" > /dev/null; do sleep 300; done
while pgrep -f "label unit30k_s0" > /dev/null; do sleep 600; done
echo "carte 0 libre $(date +%H:%M:%S)" >> "$LOG"
for s in 0 1; do
  echo "--- rawdir_usq_s${s} $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 43200 "$PY" \
    scripts/diag_nons_train.py --label "rawdir_usq_s${s}" \
    --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt \
    --horizon 24 --steps 10000 --seed "$s" --eta-init 0.995 \
    --theta-scale 0.03 --alpha-init 0.002 --muon unit_squash \
    --muon-scale 350 --frozen-every 2000 --out "rawdir_usq_s${s}.json" \
    --frozen-out "frozen_muon/rawdir_usq_s${s}.pt" \
    > "logs/rawdir_usq_s${s}.log" 2>&1 \
    && echo "rawdir_usq_s${s} OK" >> "$LOG" \
    || echo "rawdir_usq_s${s} ECHEC" >> "$LOG"
done
CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 7200 "$PY" \
  scripts/diag_pr_collapse_probe.py --val cache/egolife_pca128_val.pt \
  --frozen "frozen_muon/rawdir_usq_s*.pt" --mode cycle --clips 100 \
  --diag-out results/prlaw3/N_rawdir_usq.jsonl >> "$LOG" 2>&1 \
  || echo "sonde rawdir ECHEC" >> "$LOG"
echo "NONS-CELL FINI $(date)" >> "$LOG"
