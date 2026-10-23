#!/bin/bash
# 15/10 : 8 seeds par bras pour le rebuttal A (demande de Joseph), APRES la file forte charge.
# Seeds 3 a 7 des bras de la Fig. 5 / Tableau 2 (A, E3, FAR, puis P1, H, PN) et de la Fig. 4b (recent, far).
# Une carte rejoint quand la file hkq.txt est videe ET qu'aucun de mes python n'occupe plus la carte.
D=$HOME/work/iclr/distill; L=$HOME/work/iclr/load; U=$HOME/work/iclr/u2; H=$HOME/work/iclr/horizon
Q=$HOME/work/iclr/seeds8q.txt
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) seeds8 : $*" >> $D/orch.log; }
UC="--basis cache/qwen_l24_pca128.pt --bank $L/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --fill-choices 4,8,16 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500"
COL="--dataset finevideo --data data/finevideo --dur-min 400 --dur-max 800 --basis cache/qwen_l24_pca128.pt --eval-videos 12 --max-train-videos 60 --write-stride 8 --accum 8 --lr 3e-3 --muon unit --steps 4000 --eval-every 800 --save-every 800 --guard-gb 11"
if [ ! -f $Q ]; then
  {
  for s in 3 4 5 6 7; do
    echo "load|A_s$s|--seed $s --dense-lambda 0"
    echo "load|E3_d1_l10_s$s|--seed $s --dense-lambda 10 --dense-delta 1 --fill-choices 8"
    echo "load|FAR_D32_l10_s$s|--seed $s --dense-lambda 10 --dense-delta -32 --fill-choices 8"
  done
  for s in 3 4 5 6 7; do
    echo "color|recent_s$s|--seed $s --depth-min 0.9 --depth-max 1.0"
    echo "color|far_s$s|--seed $s --depth-min 0.0 --depth-max 0.3"
  done
  for s in 3 4 5 6 7; do
    echo "load|P1_l10_s$s|--seed $s --dense-lambda 10 --dense-delta -1 --fill-choices 8"
    echo "load|H_d32_l10_s$s|--seed $s --dense-lambda 10 --dense-delta 32 --fill-choices 8"
    echo "load|PN_D32_l10_s$s|--seed $s --dense-lambda 10 --dense-delta -32 --dense-noise 4.9 --fill-choices 8"
  done
  } > $Q
fi
card_busy() { # un de MES python sur la carte ?
  local p
  for p in $(pgrep -u "$USER" -f python); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0
  done
  return 1
}
worker() { # carte motif : la file precedente de la carte doit etre finie
  local card=$1 pat=$2
  until grep -q "$pat" $D/orch.log; do sleep 300; done
  while [ -s $L/hkq.txt ]; do sleep 300; done
  while card_busy $card || { sleep 90; card_busy $card; }; do sleep 300; done
  ( flock 9
    if [ ! -f $U/s8_smoke.ok ] && [ ! -f $U/s8_smoke.fail ]; then
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $UC --dense-lambda 10 --dense-delta 1 --fill-choices 8 --stop-after 3 --eval-every 3 --eval-n 4 --eval-k 1 --save-every 3 --reach-n 0 --seed 9 --out $U/s8_smoke > $U/s8_smoke.log 2>&1
      a=$?
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_color $COL --steps 2 --eval-every 2 --save-every 2 --eval-videos 1 --max-train-videos 2 --seed 9 --out $H/s8_smoke > $H/s8_smoke.log 2>&1
      b=$?
      if [ $a = 0 ] && [ $b = 0 ] && grep -q TRAIN-LOAD-FINIE $U/s8_smoke.log && grep -q TRAIN-COLOR-FINIE $H/s8_smoke.log
      then touch $U/s8_smoke.ok; log "smoke ok (carte $card)"; else touch $U/s8_smoke.fail; log "smoke ECHEC ($a $b)"; fi
    fi ) 9>$U/s8_smoke.lock
  [ -f $U/s8_smoke.ok ] || return
  log "carte $card rejoint"
  while true; do
    local line
    line=$(flock $Q.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
    [ -z "$line" ] && break
    local kind=${line%%|*} rest=${line#*|}
    local name=${rest%%|*} args=${rest#*|}
    log "$name carte $card"
    if [ $kind = load ]; then
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $UC $args --out $U/$name > $U/$name.log 2>&1
    else
      CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_color $COL $args --out $H/$name > $H/$name.log 2>&1
    fi
    log "$name fini (code $?)"
  done
}
worker 0 "atlas AT_PN_om_s1 fini" & worker 1 "stream : R1_full_s0 fini" & worker 2 "stream : R2_ce_s0 fini" &
wait
if [ -f $U/s8_smoke.ok ]; then
  CK=$(ls $H/{recent,far}_s[3-7]_at*.pt 2>/dev/null | sort -V | paste -sd,)
  CUDA_VISIBLE_DEVICES=0 $PY scripts/diag_graft_pr.py --data data/finevideo --basis cache/qwen_l24_pca128.pt --ckpts untrained,$CK --num-videos 6 --out $H/gates_seeds8.json > $H/gates_seeds8.log 2>&1
  log "portes horizon mesurees (code $?)"
fi
log "file videe"
