#!/bin/bash
# v2 (14/10 14:20) : comme atlas.sh, mais la carte 0 fait d'abord le bras LongVideoBench
# "16 dernieres images" (regime de la distillation, demande de Joseph), puis rejoint la file ATLAS.
D=$HOME/work/iclr/distill; U=$HOME/work/iclr/atlas; Q=$U/queue.txt; L=$HOME/work/iclr/lvb
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
AT="--omega-window 8 --omega-stride 2 --feature-degree 2"
BASE="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500 $AT"
PN="--dense-lambda 10 --dense-delta -32 --dense-noise 4.9 --fill-choices 8"
lvb_last() {
  until grep -q "lvb : full256 fini" $D/orch.log; do sleep 120; done
  local C="--data $L/data --ckpt $L/ckpt/dist_unit_s0_at4000.pt --basis cache/qwen_l24_pca128.pt"
  CUDA_VISIBLE_DEVICES=0 $PY $L/eval_lvb_last.py $C --limit 4 --out $L/last_smoke.json > $L/last_smoke.log 2>&1
  if [ $? -ne 0 ] || ! grep -q "json:" $L/last_smoke.log || grep -qE "saut|Traceback" $L/last_smoke.log; then
    log "lvb : last16 smoke ECHEC"; return; fi
  log "lvb : last16 (smoke ok) carte 0"
  CUDA_VISIBLE_DEVICES=0 $PY $L/eval_lvb_last.py $C --out $L/last16_s0_4000.json > $L/last16_s0_4000.log 2>&1
  log "lvb : last16 fini (code $?)"
}
worker() { # carte, motif de fin du job qui l'occupe
  local card=$1 pat=$2
  until grep -q "$pat" $D/orch.log; do sleep 120; done
  ( flock 9
    if [ ! -f $U/smoke.ok ] && [ ! -f $U/smoke.fail ]; then
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $BASE $PN \
        --steps 6 --eval-every 6 --eval-n 8 --reach-n 0 --batch 2 --out $U/smoke > $U/smoke.log 2>&1
      if [ $? -eq 0 ] && grep -q TRAIN-LOAD-FINIE $U/smoke.log && ! grep -qwiE "nan|Traceback|OOM" $U/smoke.log
      then touch $U/smoke.ok; log "atlas : smoke ok (carte $card)"; else touch $U/smoke.fail; log "atlas : smoke ECHEC"; fi
    fi ) 9>$U/smoke.lock
  [ -f $U/smoke.ok ] || return
  while true; do
    local line
    line=$(flock $U/q.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
    [ -z "$line" ] && break
    local name=${line%%|*} args=${line#*|}
    log "atlas $name carte $card"
    CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $BASE $args --out $U/$name > $U/$name.log 2>&1
    log "atlas $name fini (code $?)"
  done
}
worker 2 "lvb : mem_s0_4000 fini" &
worker 1 "lvb : mis_s0_4000 fini" &
( lvb_last; worker 0 "lvb : last16" ) &
wait
log "atlas : file videe"
