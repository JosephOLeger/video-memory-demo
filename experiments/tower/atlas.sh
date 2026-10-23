#!/bin/bash
# Generalite a une memoire de type ATLAS (papier A, Joseph 14/10) : Omega c=8 s=2 (vraie fenetre),
# feature map polynomiale de degre 2, ecriture unit (NS). Bras du par. 5, configs identiques aux bras Titans.
# Tour UNIQUEMENT, chaque carte apres son bras LongVideoBench.
D=$HOME/work/iclr/distill; U=$HOME/work/iclr/atlas; Q=$U/queue.txt
mkdir -p $U; cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
AT="--omega-window 8 --omega-stride 2 --feature-degree 2"
BASE="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500 $AT"
A="--dense-lambda 0 --fill-choices 4,8,16"                       # = U2 A (rappel seul)
E3="--dense-lambda 10 --dense-delta 1 --fill-choices 8"           # = E3 (cible z_{t+1})
PN="--dense-lambda 10 --dense-delta -32 --dense-noise 4.9 --fill-choices 8"   # = PN (cible imprevisible)
printf "%s\n" "AT_E3_s0|$E3 --seed 0" "AT_PN_s0|$PN --seed 0" "AT_A_s0|$A --seed 0" \
              "AT_E3_s1|$E3 --seed 1" "AT_PN_s1|$PN --seed 1" "AT_A_s1|$A --seed 1" > $Q
worker() { # carte, motif de fin du bras LVB qui l'occupe
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
worker 0 "lvb : full256 fini" &
wait
log "atlas : file videe"
