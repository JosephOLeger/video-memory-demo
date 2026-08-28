#!/usr/bin/env bash
# Campagne week-end 28-31/08 — predictions P-CDT / P-T30-s1 / P-S4-s1 /
# P-RLAD enregistrees AVANT lancement (predictions_collapse_2026-08-28.md).
# Configs strictement identiques a la famille 10k/30k (memes defauts CLI :
# batch 8, T=64, seg 8, dim 128, qk l2 ; seule la graine et --muon-scale
# varient la ou c'est le point mesure). Sondes/retentions : load_frozen =
# eval + requires_grad False, aucun pas d'outer loop.
#  w3 (carte 3, attend usq30k_s0)  : usq30k_s1 puis cdt c35/c1400 s0
#  w1 (carte 1, attend dec30k_s0)  : unit30k_s1, cdt c87/c3500 s0, dec30k_s1
#  w0 (carte 0, attend nons_cell)  : paire stride s1, cdt c35/c87 s1
#  w2 (carte 2, attend run_psic)   : cdt c1400/c3500 s1, retentions absrule
set -u
PART=${1:?w0, w1, w2 ou w3}
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG="logs/weekend_${PART}.log"
: > "$LOG"
EV="cache/egolife_pca128_val.pt"
ET="cache/egolife_pca128_train.pt"

vram_wait() { # carte — garde collegue : attendre <8 Go utilises
  local card=$1 used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card")
    [ "${used:-99999}" -lt 8000 ] && break
    echo "carte $card occupee (${used} MiB), attente $(date +%H:%M)" >> "$LOG"
    sleep 300
  done
}
train() { # label carte graine pas train val muonargs...
  local label=$1 card=$2 seed=$3 steps=$4 tr=$5 va=$6; shift 6
  vram_wait "$card"
  echo "--- train $label carte $card graine $seed $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 86400 "$PY" \
    -m titans.worldmodel.echo --label "$label" --train "$tr" --val "$va" \
    --horizon 24 --steps "$steps" --seed "$seed" --eta-init 0.995 \
    --theta-scale 0.03 --alpha-init 0.002 "$@" --frozen-every 2000 \
    --out "${label}.json" --frozen-out "frozen_muon/${label}.pt" \
    > "logs/${label}.log" 2>&1 \
    && echo "$label OK $(date +%H:%M:%S)" >> "$LOG" \
    || echo "$label ECHEC" >> "$LOG"
}
train_dec() { # comme train mais via le wrapper decouple
  local label=$1 card=$2 seed=$3 steps=$4 tr=$5 va=$6; shift 6
  vram_wait "$card"
  echo "--- train_dec $label carte $card graine $seed $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 86400 "$PY" \
    scripts/diag_decouple_train.py --label "$label" --train "$tr" --val "$va" \
    --horizon 24 --steps "$steps" --seed "$seed" --eta-init 0.995 \
    --theta-scale 0.03 --alpha-init 0.002 "$@" --frozen-every 2000 \
    --out "${label}.json" --frozen-out "frozen_muon/${label}.pt" \
    > "logs/${label}.log" 2>&1 \
    && echo "$label OK $(date +%H:%M:%S)" >> "$LOG" \
    || echo "$label ECHEC" >> "$LOG"
}
probe() { # out carte args...
  local out=$1 card=$2; shift 2
  echo "--- probe $out $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 7200 "$PY" \
    scripts/diag_pr_collapse_probe.py "$@" \
    --diag-out "results/prlaw3/${out}.jsonl" >> "$LOG" 2>&1 \
    || echo "ECHEC probe $out" >> "$LOG"
}
ret() { # out glob carte [val]
  local out=$1 glob=$2 card=$3 val=${4:-$EV}
  echo "--- retention $out $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$card nice -n 10 timeout 14400 "$PY" \
    -m titans.worldmodel.retention --frozen "$glob" --val "$val" \
    --gaps 0,1,2,4,8,16,32 --n-eval 96 --batch 8 \
    --out "${out}.json" >> "$LOG" 2>&1 || echo "ECHEC ret $out" >> "$LOG"
}

case "$PART" in
w3)
  while pgrep -f "label usq30k_s0" > /dev/null; do sleep 600; done
  train usq30k_s1 3 1 30000 $ET $EV --muon unit_squash --muon-scale 350
  probe P30_usq_s1 3 --val $EV --frozen "frozen_muon/usq30k_s1*.pt" --mode cycle --clips 100
  train cdt_c35_s0   3 0 10000 $ET $EV --muon unit_squash --muon-scale 35
  train cdt_c1400_s0 3 0 10000 $ET $EV --muon unit_squash --muon-scale 1400
  probe CDT_c35_s0   3 --val $EV --frozen "frozen_muon/cdt_c35_s0*.pt"   --mode cycle --clips 100
  probe CDT_c1400_s0 3 --val $EV --frozen "frozen_muon/cdt_c1400_s0*.pt" --mode cycle --clips 100
  ;;
w1)
  while pgrep -f "label dec30k_s0" > /dev/null; do sleep 600; done
  train unit30k_s1 1 1 30000 $ET $EV --muon unit
  probe P30_unit_s1 1 --val $EV --frozen "frozen_muon/unit30k_s1*.pt" --mode cycle --clips 100
  train cdt_c87_s0   1 0 10000 $ET $EV --muon unit_squash --muon-scale 87.5
  train cdt_c3500_s0 1 0 10000 $ET $EV --muon unit_squash --muon-scale 3500
  probe CDT_c87_s0   1 --val $EV --frozen "frozen_muon/cdt_c87_s0*.pt"   --mode cycle --clips 100
  probe CDT_c3500_s0 1 --val $EV --frozen "frozen_muon/cdt_c3500_s0*.pt" --mode cycle --clips 100
  train_dec dec30k_s1 1 1 30000 $ET $EV --muon unit
  probe P30_dec_s1 1 --val $EV --frozen "frozen_muon/dec30k_s1*.pt" --mode cycle --clips 100
  ;;
w0)
  while pgrep -f "run_nons_cell.sh" > /dev/null; do sleep 600; done
  train unit_str4_s1   0 1 10000 cache/egolife_l256_str4_train.pt   cache/egolife_l256_str4_val.pt   --muon unit
  train unit_ctrl64_s1 0 1 10000 cache/egolife_l256_ctrl64_train.pt cache/egolife_l256_ctrl64_val.pt --muon unit
  probe S4_str4_s1   0 --val cache/egolife_l256_str4_val.pt   --frozen "frozen_muon/unit_str4_s1*.pt"   --mode cycle --clips 100
  probe S4_ctrl64_s1 0 --val cache/egolife_l256_ctrl64_val.pt --frozen "frozen_muon/unit_ctrl64_s1*.pt" --mode cycle --clips 100
  train cdt_c35_s1 0 1 10000 $ET $EV --muon unit_squash --muon-scale 35
  train cdt_c87_s1 0 1 10000 $ET $EV --muon unit_squash --muon-scale 87.5
  probe CDT_c35_s1 0 --val $EV --frozen "frozen_muon/cdt_c35_s1*.pt" --mode cycle --clips 100
  probe CDT_c87_s1 0 --val $EV --frozen "frozen_muon/cdt_c87_s1*.pt" --mode cycle --clips 100
  ;;
w2)
  while pgrep -f "run_psic.sh" > /dev/null; do sleep 900; done
  train cdt_c1400_s1 2 1 10000 $ET $EV --muon unit_squash --muon-scale 1400
  train cdt_c3500_s1 2 1 10000 $ET $EV --muon unit_squash --muon-scale 3500
  probe CDT_c1400_s1 2 --val $EV --frozen "frozen_muon/cdt_c1400_s1*.pt" --mode cycle --clips 100
  probe CDT_c3500_s1 2 --val $EV --frozen "frozen_muon/cdt_c3500_s1*.pt" --mode cycle --clips 100
  for fam in c35 c87 c1400 c3500; do
    until [ -f "frozen_muon/cdt_${fam}_s0.pt" ] && [ -f "frozen_muon/cdt_${fam}_s1.pt" ]; do sleep 600; done
    ret "retention_cdt_${fam}_absrule" "frozen_muon/cdt_${fam}_s[0-9].pt" 2
  done
  until [ -f "frozen_muon/rawdir_usq_s1.pt" ]; do sleep 600; done
  ret retention_rawdir_usq_absrule "frozen_muon/rawdir_usq_s[0-9].pt" 2
  ret retention_usq30k_absrule  "frozen_muon/usq30k_s[0-9].pt"  2
  ret retention_unit30k_absrule "frozen_muon/unit30k_s[0-9].pt" 2
  # dec30k : s1 finit lundi matin ; cette passe couvre ce qui existe.
  ret retention_dec30k_absrule  "frozen_muon/dec30k_s[0-9].pt"  2
  ret retention_tx10k_absrule   "frozen_muon/tx10k_unit_s[0-9].pt" 2 cache/enwik8_emb128_val.pt
  ;;
esac
echo "WEEKEND_${PART} FINI $(date)" >> "$LOG"
