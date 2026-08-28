#!/usr/bin/env bash
# Nuit du 28/08 — temps longs + leviers. Usage: run_collapse_night.sh {c3|c01|c2}
#  c3  (carte 3, immediat)   : sondes cos_m (volet M) puis usq30k (30k pas).
#  c01 (cartes 0-1, attend tx10k) : sonde tx10k puis unit30k + dec30k.
#  c2  (carte 2, attend lucidrains) : caches stride, puis unit_str4 et
#       unit_ctrl64 (10k pas chacun, sequentiel).
# Predictions AVANT lancement : results/predictions_collapse_2026-08-28.md
set -u
PART=$1
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG="logs/night_${PART}.log"
mkdir -p results/prlaw3
: > "$LOG"

probe() { # out card args...
  local out=$1 card=$2; shift 2
  echo "--- probe $out $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 7200 "$PY" \
    scripts/diag_pr_collapse_probe.py "$@" \
    --diag-out "results/prlaw3/${out}.jsonl" >> "$LOG" 2>&1 \
    || echo "ECHEC probe $out" >> "$LOG"
}
train() { # label card steps train val muonargs...
  local label=$1 card=$2 steps=$3 tr=$4 va=$5; shift 5
  echo "--- train $label carte $card $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 86400 "$PY" \
    -m titans.worldmodel.echo --label "$label" --train "$tr" --val "$va" \
    --horizon 24 --steps "$steps" --seed 0 --eta-init 0.995 \
    --theta-scale 0.03 --alpha-init 0.002 "$@" --frozen-every 2000 \
    --out "${label}.json" --frozen-out "frozen_muon/${label}.pt" \
    > "logs/${label}.log" 2>&1 \
    && echo "$label OK $(date +%H:%M:%S)" >> "$LOG" \
    || echo "$label ECHEC" >> "$LOG"
}
EV="cache/egolife_pca128_val.pt"
ET="cache/egolife_pca128_train.pt"

case "$PART" in
c3)
  probe M_unit10k 3 --val $EV --frozen "frozen_muon/mu_unit10k_s[0-9].pt" --mode cycle --clips 100
  probe M_usq10k  3 --val $EV --frozen "frozen_muon/mu_usq10k_s[0-9].pt"  --mode cycle --clips 100
  probe M_dec     3 --val $EV --frozen "frozen_muon/mu_unit10k_dec_s[01].pt" --mode cycle --clips 100
  probe M_text    3 --val cache/enwik8_emb128_val.pt --frozen "frozen_muon/tx_*_s[01].pt" --mode cycle --clips 100
  train usq30k_s0 3 30000 $ET $EV --muon unit_squash --muon-scale 350
  probe P30_usq 3 --val $EV --frozen "frozen_muon/usq30k_s0*.pt" --mode cycle --clips 100
  ;;
c01)
  while pgrep -f "label tx10k_unit" > /dev/null; do sleep 300; done
  probe X10k_text 0 --val cache/enwik8_emb128_val.pt --frozen "frozen_muon/tx10k_unit_s[01]*.pt" --mode cycle --clips 100
  train unit30k_s0 0 30000 $ET $EV --muon unit &
  (
    echo "--- train dec30k_s0 carte 1 (wrapper decouple) $(date +%H:%M:%S)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=1 nice -n 10 timeout 86400 "$PY" \
      scripts/diag_decouple_train.py --label dec30k_s0 \
      --train $ET --val $EV --horizon 24 --steps 30000 --seed 0 \
      --eta-init 0.995 --theta-scale 0.03 --alpha-init 0.002 --muon unit \
      --frozen-every 2000 --out dec30k_s0.json \
      --frozen-out frozen_muon/dec30k_s0.pt > logs/dec30k_s0.log 2>&1 \
      && echo "dec30k_s0 OK $(date +%H:%M:%S)" >> "$LOG" \
      || echo "dec30k_s0 ECHEC" >> "$LOG"
  ) &
  wait
  probe P30_unit 0 --val $EV --frozen "frozen_muon/unit30k_s0*.pt" --mode cycle --clips 100
  probe P30_dec  1 --val $EV --frozen "frozen_muon/dec30k_s0*.pt" --mode cycle --clips 100
  ;;
c2)
  "$PY" scripts/build_stride_cache.py --src-train cache/egolife_long256_train.pt \
    --src-val cache/egolife_long256_val.pt --out-prefix cache/egolife_l256 >> "$LOG" 2>&1
  while pgrep -f "train_r_prlaw.py" > /dev/null; do sleep 300; done
  train unit_str4_s0   2 10000 cache/egolife_l256_str4_train.pt   cache/egolife_l256_str4_val.pt   --muon unit
  train unit_ctrl64_s0 2 10000 cache/egolife_l256_ctrl64_train.pt cache/egolife_l256_ctrl64_val.pt --muon unit
  probe S4_str4   2 --val cache/egolife_l256_str4_val.pt   --frozen "frozen_muon/unit_str4_s0*.pt"   --mode cycle --clips 100
  probe S4_ctrl64 2 --val cache/egolife_l256_ctrl64_val.pt --frozen "frozen_muon/unit_ctrl64_s0*.pt" --mode cycle --clips 100
  ;;
esac
echo "NIGHT-${PART} FINI $(date)" >> "$LOG"
