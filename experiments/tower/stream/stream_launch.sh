#!/bin/bash
# 15/10 : precalcul des cibles FLUX (stream_targets), 3 eclats. Eclat 0 carte 1 tout de suite ;
# eclats 1 et 2 sur les cartes 0 et 2 quand leurs ablations ATLAS sont finies.
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
run() { # eclat carte
  log "stream : eclat $1 carte $2"
  CUDA_VISIBLE_DEVICES=$2 $PY -m titans.graft.stream_targets --out $S/tg --shard $1 --nshards 3 > $S/tg_$1.log 2>&1
  log "stream : eclat $1 fini (code $?)"
}
run 0 1 &
( until grep -q "atlas AT_A_om_s1 fini" $D/orch.log; do sleep 120; done; run 1 0 ) &
( until grep -q "atlas AT_A_phi_s1 fini" $D/orch.log; do sleep 120; done; run 2 2 ) &
wait
log "stream : cibles finies"
