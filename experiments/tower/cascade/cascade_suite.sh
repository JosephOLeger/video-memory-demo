#!/bin/bash
# 20/10 soir : la simulation (regime stationnaire, plasticite egale) classe la lecture MOYENNE
# (uniform = "mean" dans le code) devant u1/shallow et geo a K=4. Bras ajoute apres la file cascade_card0.sh.
C=$HOME/work/iclr/cascade; O=$HOME/work/iclr/distill/orch.log
log() { echo "$(date +%F_%T) cascade : $*" >> $O; }
until grep -q "cascade : FINI\|cascade : smoke ECHEC" $O; do sleep 300; done
grep -q "cascade : smoke ECHEC" $O && exit 1
cd $C/tm
PY=$HOME/miniforge3/envs/titans/bin/python
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COM="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --train-k 1,2,4 --fill-choices 4,8,16 --steps 2500 --batch 8 --eval-every 1250 --eval-k 1,2,4,8 --eval-n 96 --reach-n 48 --reach-fill 90 --reach-after 0,2,8,22,45,90 --save-every 500 --lr 3e-3 --seed 0"
CAS="--cascade --cascade-mode direct --cascade-leak none --cascade-areas 2 --cascade-g 2 --cascade-beta1 0.5 --cascade-terminal-leak"
n=d_K4_t16k_mean; log "$n carte 0"
$PY -m titans.graft.train_load $COM $CAS --reservoirs 4 --cascade-terminal-tau 16000 --cascade-readout mean --out $C/runs/$n > $C/runs/$n.log 2>&1
log "$n fini (code $?)"
