#!/usr/bin/env bash
# Death-step arm (titans-literal, diag_titans_train.py), theta 0.03, seeds 2,3,4.
# Sequential on card 2 so card 2 never exceeds 2 concurrent jobs.
set -u
cd "$HOME/work/iclr/titans-mac" || exit 1
PY="$HOME/miniforge3/envs/titans/bin/python"
TR="$HOME/work/iclr/titans-mac/cache/egolife_pca128_train.pt"
VA="$HOME/work/iclr/titans-mac/cache/egolife_pca128_val.pt"
for S in 2 3 4; do
  echo "=== tl_t003_s$S start $(date) ==="
  env PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 "$PY" scripts/diag_titans_train.py \
    --label tl_t003_s$S --train "$TR" --val "$VA" \
    --horizon 24 --seg 8 --batch 8 --steps 1500 --seed "$S" \
    --theta-scale 0.03 --alpha-init 0.002 --eta-init 0.995 \
    --diag-out "$HOME/work/iclr/out_theory/tl_t003_s$S.jsonl" \
    > "$HOME/work/iclr/logs_theory/tl_t003_s$S.log" 2>&1
  echo "=== tl_t003_s$S rc=$? $(date) ==="
done
echo "TL_CHAIN DONE $(date)"
