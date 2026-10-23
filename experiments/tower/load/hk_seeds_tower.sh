#!/bin/bash
# 15/10 : graines 3 a 7 de la FORTE CHARGE sur la tour. Une file partagee, une carte rejoint quand
# son travail courant est fini : carte 1 apres R1, carte 2 apres R2, carte 0 apres la file Omega.
D=$HOME/work/iclr/distill; L=$HOME/work/iclr/load; Q=$L/hkq.txt
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
C="--basis cache/qwen_l24_pca128.pt --bank $L/bank512.pt --muon unit --dense-lambda 0 --fill-choices 8 --max-writes 4000"
F="--clamp-eta 0.995 --clamp-alpha 0.000127"
R="--steps 30000 --stop-after 8000 --train-k 1,2,4,8,16,32,64,128,256 --advance-acc 0.35 --eval-every 1000 --eval-n 96 --eval-k 1,4,16,64,128,256,512 --save-every 2000"
worker() { # carte motif
  local card=$1 pat=$2
  until grep -q "$pat" $D/orch.log; do sleep 120; done
  ( flock 9
    if [ ! -f $L/hkt_smoke.ok ] && [ ! -f $L/hkt_smoke.fail ]; then
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $C $F --steps 30000 --stop-after 3 --train-k 1,2 --eval-every 3 --eval-n 4 --eval-k 1 --save-every 3 --reach-n 0 --seed 9 --out $L/hkt_smoke > $L/hkt_smoke.log 2>&1
      if grep -q TRAIN-LOAD-FINIE $L/hkt_smoke.log && [ -f $L/hkt_smoke_at3.pt ] && [ ! -f $L/hkt_smoke_at6.pt ]
      then touch $L/hkt_smoke.ok; log "graines : smoke ok (carte $card)"; else touch $L/hkt_smoke.fail; log "graines : smoke ECHEC"; fi
    fi ) 9>$L/hkt_smoke.lock
  [ -f $L/hkt_smoke.ok ] || return
  while true; do
    local line
    line=$(flock $L/hkq.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
    [ -z "$line" ] && break
    local arm=${line% *} s=${line#* } X=""
    [ $arm = hkf ] && X="$F"
    log "graines : ${arm}_unit_s$s carte $card"
    CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $C $X $R --seed $s --out $L/${arm}_unit_s$s > $L/${arm}_unit_s$s.log 2>&1
    log "graines : ${arm}_unit_s$s fini (code $?)"
  done
}
worker 1 "stream : R1_full_s0 fini" &
worker 2 "stream : R2_ce_s0 fini" &
worker 0 "atlas AT_PN_om_s1 fini" &
wait
log "graines : file tour videe"
