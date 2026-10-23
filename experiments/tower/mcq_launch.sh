#!/bin/bash
# QCM sur elements precis (Joseph 14/10). Tour, CARTE 0, apres le bras LVB last16 et quand
# aucun autre job n'occupe la carte (la file ATLAS peut la prendre si elle n'est pas vide).
D=$HOME/work/iclr/distill; M=$HOME/work/iclr/mcq; mkdir -p $M
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
until grep -q "lvb : last16 fini" $D/orch.log; do sleep 120; done
free0() { [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)")" ]; }
until free0 && sleep 60 && free0; do sleep 120; done
C="--mcq $HOME/work/iclr/mcq3.jsonl --basis cache/qwen_l24_pca128.pt"
$PY -m titans.graft.train_mcq --targets "$D/targets_*.pt" $C --init $D/dist_unit_s0_at2000.pt --steps 3 --eval-every 3 --max-eval 16 \
   --save-every 0 --out $M/smoke > $M/smoke.log 2>&1
if [ $? -ne 0 ] || ! grep -q TRAIN-MCQ-FINIE $M/smoke.log || grep -qwiE "nan|Traceback" $M/smoke.log; then
  log "mcq : smoke ECHEC"; exit 1; fi
log "mcq : smoke ok ; mcq_dist2000_s0 carte 0"
$PY -m titans.graft.train_mcq --targets "$D/targets_*.pt" $C --init $D/dist_unit_s0_at2000.pt --steps 1500 --out $M/mcq_dist2000_s0 > $M/mcq_dist2000_s0.log 2>&1
log "mcq : mcq_dist2000_s0 fini (code $?) ; mcq_scratch_s0 carte 0"
$PY -m titans.graft.train_mcq --targets "$D/targets_*.pt" $C --steps 1500 --out $M/mcq_scratch_s0 > $M/mcq_scratch_s0.log 2>&1
log "mcq : mcq_scratch_s0 fini (code $?)"
