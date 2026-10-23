#!/bin/bash
# 20/10 (Josep : « lance un essai court sur la cascade ») : carte 0 de la tour pour la cascade DIRECTE.
# 1. retire la voie « carte 0 » de tower_2010b.sh (couleur + banc sur carte 0) avant qu'elle ne demarre ;
# 2. apres « seeds8 : file videe » et carte 0 libre : smoke 20 pas, puis les bras en file, un a la fois.
C=$HOME/work/iclr/cascade; O=$HOME/work/iclr/distill/orch.log; MAIN=2891448
log() { echo "$(date +%F_%T) cascade : $*" >> $O; }
busy0() { local p; for p in $(pgrep -u "$USER" -f python); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=0" && return 0; done; return 1; }
# 1. la voie carte 0 = sous-shell de MAIN dont l'enfant direct est « sleep 300 »
until grep -q "couleur v : .* videos de variantes" $O; do sleep 60; done
for i in $(seq 30); do
  for p in $(pgrep -P $MAIN -x bash); do
    if pgrep -P $p -x sleep >/dev/null; then kill $(pgrep -P $p -x sleep) 2>/dev/null; kill $p; log "voie carte 0 de tower_2010b retiree (PID $p)"; break 2; fi
  done; sleep 30
done
# 2.
until grep -q "seeds8 : file videe" $O; do sleep 300; done
while busy0; do sleep 120; done
cd $C/tm
PY=$HOME/miniforge3/envs/titans/bin/python
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $C/runs
COM="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --train-k 1,2,4 --fill-choices 4,8,16 --steps 2500 --batch 8 --eval-every 1250 --eval-k 1,2,4,8 --eval-n 96 --reach-n 48 --reach-fill 90 --reach-after 0,2,8,22,45,90 --save-every 500 --lr 3e-3 --seed 0"
CAS="--cascade --cascade-mode direct --cascade-leak none --cascade-areas 2 --cascade-g 2 --cascade-beta1 0.5 --cascade-terminal-leak"
$PY -m titans.graft.train_load $COM $CAS --reservoirs 4 --cascade-terminal-tau 16000 --cascade-readout shallow --stop-after 20 --eval-every 1000 --save-every 1000 --reach-n 0 --out $C/runs/smoke > $C/runs/smoke.log 2>&1
if ! grep -q TRAIN-LOAD-FINIE $C/runs/smoke.log || grep -qi nan $C/runs/smoke.log; then log "smoke ECHEC"; exit 1; fi
log "smoke ok"
run() { local n=$1; shift; log "$n carte 0"; $PY -m titans.graft.train_load $COM "$@" --out $C/runs/$n > $C/runs/$n.log 2>&1; log "$n fini (code $?)"; }
run d_K4_t16k_u1 $CAS --reservoirs 4 --cascade-terminal-tau 16000 --cascade-readout shallow
run ctl
run exp_T --clamp-eta 0 --clamp-alpha 2.44e-4
run d_K6_u1 $CAS --reservoirs 6 --cascade-readout shallow
run d_K4_t16k_geo $CAS --reservoirs 4 --cascade-terminal-tau 16000 --cascade-readout geo
log "FINI"
