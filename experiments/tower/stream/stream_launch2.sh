#!/bin/bash
# v2 (15/10 12:00) : equilibrage. Carte 1 enchaine l'eclat 1 apres l'eclat 0 ; carte 0 fait
# l'eclat 2 apres AT_A_om_s1 ; carte 2 fait les eclats 1 puis 2 A L'ENVERS apres AT_A_phi_s1
# (les fichiers existants sont sautes : deux cartes sur un eclat se rejoignent au milieu).
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
run() { # eclat carte [--reverse]
  log "stream : eclat $1 carte $2 $3"
  CUDA_VISIBLE_DEVICES=$2 $PY -m titans.graft.stream_targets --out $S/tg --shard $1 --nshards 3 $3 > $S/tg_$1_c$2.log 2>&1
  log "stream : eclat $1 carte $2 fini (code $?)"
}
( until grep -q "stream : eclat 0 fini" $D/orch.log; do sleep 60; done; run 1 1 ) &
( until grep -q "atlas AT_A_om_s1 fini" $D/orch.log; do sleep 60; done; run 2 0 ) &
( until grep -q "atlas AT_A_phi_s1 fini" $D/orch.log; do sleep 60; done; run 1 2 --reverse; run 2 2 --reverse ) &
wait
log "stream : cibles finies"
