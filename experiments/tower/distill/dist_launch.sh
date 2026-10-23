#!/bin/bash
# Distillation (13/10, demande de Joseph) : remplace la fin d'orch5 (glob des shards non quote = bug).
# Tour seulement. Carte 0 tout de suite ; cartes 1 et 2 des que les runs H_d32 qui les occupent finissent.
D=$HOME/work/iclr/distill
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
C="--basis cache/qwen_l24_pca128.pt --steps 8000"
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
run() { # carte muon seed
  CUDA_VISIBLE_DEVICES=$1 $PY -m titans.graft.train_distill --targets "$D/targets_[0-3].pt" $C \
    --muon $2 --seed $3 --out $D/dist_$2_s$3 > $D/dist_$2_s$3.log 2>&1
  log "distill $2 s$3 fini (code $?)"
}
log "dist_launch : unit s0 carte 0"
run 0 unit 0 &
( while kill -0 3023225 2>/dev/null; do sleep 60; done; log "dist_launch : unit_squash s0 carte 1"; run 1 unit_squash 0 ) &
( while kill -0 3024148 2>/dev/null; do sleep 60; done; log "dist_launch : unit s1 carte 2"; run 2 unit 1 ) &
wait
