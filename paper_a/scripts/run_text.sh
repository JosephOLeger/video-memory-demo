#!/usr/bin/env bash
# Volet X — le texte sur notre banc (P-X1..X3), carte 0.
# Cache enwik8 -> embedding octet fixe 128-d, puis 4 bras (unit/usq x s0/s1)
# a la config mu_unit10k (4000 pas), puis la sonde de loi sur chacun.
set -u
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG=logs/text_arms.log
: > "$LOG"

"$PY" scripts/build_text_cache.py \
  --enwik "$HOME/repro/titans-pytorch/data/enwik8.gz" \
  --out-train cache/enwik8_emb128_train.pt \
  --out-val cache/enwik8_emb128_val.pt >> "$LOG" 2>&1 \
  || { echo "CACHE ECHEC" >> "$LOG"; exit 1; }

for s in 0 1; do
  for arm in unit usq; do
    if [ "$arm" = unit ]; then extra="--muon unit";
    else extra="--muon unit_squash --muon-scale 350"; fi
    echo "--- tx_${arm}_s${s} $(date +%H:%M:%S) ---" >> "$LOG"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 21600 "$PY" \
      -m titans.worldmodel.echo --label "tx_${arm}_s${s}" \
      --train cache/enwik8_emb128_train.pt --val cache/enwik8_emb128_val.pt \
      --horizon 24 --steps 4000 --seed "$s" --eta-init 0.995 \
      --theta-scale 0.03 --alpha-init 0.002 $extra \
      --out "tx_${arm}_s${s}.json" \
      --frozen-out "frozen_muon/tx_${arm}_s${s}.pt" \
      > "logs/tx_${arm}_s${s}.log" 2>&1 \
      && echo "tx_${arm}_s${s} OK" >> "$LOG" \
      || echo "tx_${arm}_s${s} ECHEC" >> "$LOG"
  done
done

CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 7200 "$PY" \
  scripts/diag_pr_collapse_probe.py --val cache/enwik8_emb128_val.pt \
  --frozen "frozen_muon/tx_*_s[01].pt" --mode cycle --clips 100 \
  --diag-out results/prlaw/X_text.jsonl >> "$LOG" 2>&1 \
  || echo "sonde texte ECHEC" >> "$LOG"
echo "TEXTE TOUT FINI $(date)" >> "$LOG"
