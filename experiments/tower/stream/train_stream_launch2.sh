#!/bin/bash
# 15/10 14:00 v2 : carte 1 aide l'eclat 2 (sens direct) pendant que la carte 2 le fait a l'envers ;
# entrainements apres les DEUX : R1 complet carte 1, R2 CE seule carte 2. Carte 0 = papier A (Omega seul).
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
log "stream : eclat 2 carte 1"
CUDA_VISIBLE_DEVICES=1 $PY -m titans.graft.stream_targets --out $S/tg --shard 2 --nshards 3 > $S/tg_2_c1.log 2>&1
log "stream : eclat 2 carte 1 fini (code $?)"
until grep -q "stream : eclat 2 carte 2 fini" $D/orch.log; do sleep 60; done
log "stream : cibles finies ($(ls $S/tg | wc -l) videos)"
C=(--targets "$S/tg/*.pt" --basis cache/qwen_l24_pca128.pt --steps 2000 --batch 4 --eval-every 250 --max-eval 300)
go() { # nom carte args...
  local n=$1 c=$2; shift 2
  log "stream : $n carte $c"
  CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream "${C[@]}" "$@" --out $S/$n > $S/$n.log 2>&1
  log "stream : $n fini (code $?)"
}
go R1_full_s0 1 &
go R2_ce_s0 2 --lam-lat 0 --lam-con 0 &
wait
