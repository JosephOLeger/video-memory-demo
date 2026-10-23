#!/bin/bash
# Controles de previsibilite (revues du 13/10), TOUR, apres la distillation (ne la touche pas).
# Chaque carte attend la fin de SON run de distillation, puis prend des jobs dans la file.
# PN = cible passee z_{t-32} bruitee (R^2 atteignable ~0,04, comme z_{t+1}) ; P1 = cible passee proche z_{t-1} ;
# H s2 = 3e seed de la cible future lointaine z_{t+32}. lambda 10, config U2.
D=$HOME/work/iclr/distill; U=$HOME/work/iclr/u2; Q=$U/ctrl_queue.txt
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
UC="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500 --dense-lambda 10 --fill-choices 8"
printf "%s\n" \
 "PN_D32_l10_s0|--seed 0 --dense-delta -32 --dense-noise 4.9" "P1_l10_s0|--seed 0 --dense-delta -1" \
 "PN_D32_l10_s1|--seed 1 --dense-delta -32 --dense-noise 4.9" "P1_l10_s1|--seed 1 --dense-delta -1" \
 "PN_D32_l10_s2|--seed 2 --dense-delta -32 --dense-noise 4.9" "P1_l10_s2|--seed 2 --dense-delta -1" \
 "H_d32_l10_s2|--seed 2 --dense-delta 32" > $Q
worker() { # carte, motif de fin de distillation
  local card=$1 pat=$2
  until grep -q "$pat" $D/orch.log; do sleep 120; done
  # smoke une seule fois (premiere carte libre), les autres attendent son verdict
  ( flock 9
    if [ ! -f $U/ctrl_smoke.ok ] && [ ! -f $U/ctrl_smoke.fail ]; then
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $UC --dense-delta -32 --dense-noise 4.9 \
        --steps 6 --eval-every 6 --eval-n 8 --reach-n 0 --batch 2 --out $U/ctrl_smoke > $U/ctrl_smoke.log 2>&1
      if [ $? -eq 0 ] && grep -q TRAIN-LOAD-FINIE $U/ctrl_smoke.log && ! grep -qwiE "nan|Traceback|OOM" $U/ctrl_smoke.log
      then touch $U/ctrl_smoke.ok; log "ctrl : smoke ok (carte $card)"; else touch $U/ctrl_smoke.fail; log "ctrl : smoke ECHEC"; fi
    fi ) 9>$U/ctrl_smoke.lock
  [ -f $U/ctrl_smoke.ok ] || return
  while true; do
    local line
    line=$(flock $U/ctrl.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
    [ -z "$line" ] && break
    local name=${line%%|*} args=${line#*|}
    log "ctrl $name carte $card"
    CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $UC $args --out $U/$name > $U/$name.log 2>&1
    log "ctrl $name fini (code $?)"
  done
}
worker 0 "distill unit s0 fini" &
worker 1 "distill unit_squash s0 fini" &
worker 2 "distill unit s1 fini" &
wait
log "ctrl : file videe"
