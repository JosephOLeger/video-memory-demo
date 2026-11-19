#!/bin/bash
# Amputated-past track (paper B, video). Three steps on FineVideo (data/finevideo, see fetch_finevideo.py):
#   1. targets: teacher and window-only distributions at four cut points per video, in N shards (one GPU each);
#   2. a four-step smoke test of the trainer;
#   3. the arms: unweighted KL (the formulation), weighted KL, negatives p 0.3, memory width 256.
# Usage: run_ctx.sh targets SHARD NSHARDS CARD | smoke CARD | arm NAME CARD [train_ctx flags]
set -u
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; D=${D:-$ROOT/runs/ctx}; PY=${PY:-python}; mkdir -p $D
cd $ROOT; export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=cache/qwen_l24_pca128.pt
case $1 in
  targets) CUDA_VISIBLE_DEVICES=$4 $PY -m titans.graft.ctx_targets --basis $B --shard $2 --nshards $3 --out $D/targets > $D/targets_$2.log 2>&1 ;;
  smoke)   CUDA_VISIBLE_DEVICES=$2 $PY -m titans.graft.train_ctx --targets "$D/targets_0.pt" --basis $B --steps 4 --eval-every 4 --eval-n 16 --batch 4 --neg-p 0.5 --weight min --out $D/smoke > $D/smoke.log 2>&1
           grep -q TRAIN-CTX-FINIE $D/smoke.log && echo "smoke ok" || { echo "smoke failed, see $D/smoke.log"; exit 1; } ;;
  arm)     name=$2; card=$3; shift 3
           CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_ctx --targets "$D/targets_*.pt" --basis $B --steps 8000 --eval-every 1000 --eval-n 480 --save-every 2000 "$@" --out $D/$name > $D/$name.log 2>&1 ;;
  *) echo "usage: run_ctx.sh targets SHARD NSHARDS CARD | smoke CARD | arm NAME CARD [flags]"; exit 1 ;;
esac
# the four arms of November 2026:
#   run_ctx.sh arm ctx_u_s0 0 --weight none            run_ctx.sh arm ctx_w_s0 1 --weight short
#   run_ctx.sh arm ctx_neg_s0 2 --weight none --neg-p 0.3   run_ctx.sh arm ctx_wide_s0 3 --weight none --d-mem 256
