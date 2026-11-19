#!/usr/bin/env bash
# Campagne 2 (loi DC incluse dans la sonde) — usage: run_prlaw2.sh {a|b|c} CARTE
# a: volet L (4 bras) ; b: T2 checkpoints (+dec) ; c: S + D + C + X.
set -u
PART=$1; CARD=$2
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG="logs/prlaw2_${PART}.log"
mkdir -p results/prlaw2
: > "$LOG"

run() {
  local out=$1; shift
  echo "--- $out $* $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout 7200 "$PY" \
    scripts/diag_pr_collapse_probe.py "$@" \
    --diag-out "results/prlaw2/${out}.jsonl" >> "$LOG" 2>&1 \
    || echo "ECHEC: $out" >> "$LOG"
}
V="--val cache/egolife_pca128_val.pt"
VT="--val cache/enwik8_emb128_val.pt"

case "$PART" in
a)
  for arm in mu_unit mu_unit10k mu_usq10k mu_scl; do
    run "L_${arm}" $V --frozen "frozen_muon/${arm}_s[0-9].pt" --mode cycle --clips 100
  done
  ;;
b)
  for s in 0 1; do
    run "T2_unit10k_s${s}" $V --frozen "frozen_muon/mu_unit10k_s${s}_at*.pt" --mode cycle --clips 60
    run "T2_usq10k_s${s}"  $V --frozen "frozen_muon/mu_usq10k_s${s}_at*.pt"  --mode cycle --clips 60
    run "T2_dec_s${s}"     $V --frozen "frozen_muon/mu_unit10k_dec_s${s}_at*.pt" --mode cycle --clips 60
  done
  ;;
c)
  run "C_decouple" $V --frozen "frozen_muon/mu_unit10k_dec_s[01].pt" --mode cycle --clips 100
  run "X_text" $VT --frozen "frozen_muon/tx_*_s[01].pt" --mode cycle --clips 100
  for m in loop shuffle; do
    run "S_${m}_unit10k" $V --frozen "frozen_muon/mu_unit10k_s[01].pt" --mode "$m" --clips 100
    run "S_${m}_usq10k"  $V --frozen "frozen_muon/mu_usq10k_s0.pt"    --mode "$m" --clips 100
  done
  for al in 0.003 0.01 0.03; do
    run "D_a${al}_unit10k" $V --frozen "frozen_muon/mu_unit10k_s[01].pt" \
        --mode cycle --clips 100 --alpha-override "$al" --tokens-per-call 8
  done
  ;;
esac
echo "PRLAW2-${PART} FINI $(date)" >> "$LOG"
