#!/bin/bash
# Gaussian horizon of the dense target (paper A, November 2026). Two arms of train_load, 8 seeds each:
#   GP  delta ~ N(-16, 8^2) truncated to [-48, -1]    recall of the past at several scales
#   GS  delta ~ N(0, 16^2)  truncated to [-48, 48], 0 excluded   past and future mixed
# Same configuration as the October seed study: lambda 10, 2500 steps. Usage: horizon.sh CARD [CARD ...]
# Needs cache/qwen_l24_pca128.pt and the assembled-past bank (train_load --build-bank) at $BANK.
set -u
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; D=${D:-$ROOT/runs/horizon}; BANK=${BANK:-$ROOT/runs/load/bank.pt}
PY=${PY:-python}; Q=$D/queue.txt; mkdir -p $D
cd $ROOT; export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UC="--basis cache/qwen_l24_pca128.pt --bank $BANK --muon unit --steps 2500 --train-k 1,2,4 --fill-choices 8 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500 --dense-lambda 10"
if [ ! -s $Q ]; then
  for s in 0 1 2 3 4 5 6 7; do
    echo "GP_s$s|--seed $s --dense-delta -16 --dense-delta-sigma 8 --dense-delta-lo -48 --dense-delta-hi -1" >> $Q
    echo "GS_s$s|--seed $s --dense-delta 0 --dense-delta-sigma 16 --dense-delta-lo -48 --dense-delta-hi 48" >> $Q
  done
fi
log() { echo "$(date +%F_%T) $*" >> $D/horizon.log; }
pop() { ( flock 9; l=$(head -1 $Q); [ -n "$l" ] && sed -i 1d $Q; echo "$l" ) 9>$D/queue.lock; }
worker() { local card=$1 l name args
  while true; do l=$(pop); [ -z "$l" ] && break
    name=${l%%|*}; args=${l#*|}; log "card $card: $name ($args)"
    CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $UC $args --out $D/$name > $D/$name.log 2>&1
    log "card $card: $name done ($(grep -c TRAIN-LOAD-FINIE $D/$name.log) finished, $(grep -c OOM $D/$name.log) OOM)"
  done; log "card $card: queue empty"; }
for c in "$@"; do worker $c & done; wait; log "all done"
