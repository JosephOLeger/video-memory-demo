#!/bin/bash
# QCM v2 (15/10 00:10) : relance apres le crash des tailles de clips (videos non tronquees). Tour, carte 0.
D=$HOME/work/iclr/distill; M=$HOME/work/iclr/mcq
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
C=(--targets "$D/targets_*.pt" --mcq $HOME/work/iclr/mcq3.jsonl --basis cache/qwen_l24_pca128.pt)

log "mcq : relance mcq_dist2000_s0 carte 0"
$PY -m titans.graft.train_mcq "${C[@]}" --init $D/dist_unit_s0_at2000.pt --steps 1500 --out $M/mcq_dist2000_s0 > $M/mcq_dist2000_s0.log 2>&1
rc=$?; log "mcq : mcq_dist2000_s0 fini (code $rc)"
[ $rc -eq 0 ] || exit 1
log "mcq : mcq_scratch_s0 carte 0"
$PY -m titans.graft.train_mcq "${C[@]}" --steps 1500 --out $M/mcq_scratch_s0 > $M/mcq_scratch_s0.log 2>&1
log "mcq : mcq_scratch_s0 fini (code $?)"
