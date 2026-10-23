#!/bin/bash
# 20/10 soir (Josep : « lance le diagnostic, retire s4 a s7, le cas echeant lr plus agressif »).
# Carte 1 : apres C1v_color_s0 (PID 3199514) -> diag_read sur 4 checkpoints -> C1v couleur s2, lr lu dans lr_s2.txt.
S=$HOME/work/iclr/stream; O=$HOME/work/iclr/distill/orch.log; PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
log() { echo "$(date +%F_%T) $*" >> $O; }
while kill -0 3199514 2>/dev/null; do sleep 60; done; sleep 30
cd $HOME/work/titans-mac
log "diag_read couleur carte 1"
$PY $S/diag_read.py --ckpts $S/C1_color_s0_at500.pt $S/C1_color_s0_at4000.pt $S/C1_color_s1_at4000.pt $S/C1v_color_s3_at4000.pt \
  --targets "$S/tgc/*.pt" --variants "$S/tgv/*.pt" --basis cache/qwen_l24_pca128.pt --n 120 --out $S/diag_read.jsonl > $S/diag_read.log 2>&1
log "diag_read fini (code $?)"
LR=$(cat $S/lr_s2.txt); N=C1v_color_s2_lr$LR
log "couleur v : $N carte 1"
$PY -m titans.graft.train_stream --targets "$S/tgc/*.pt" --variants "$S/tgv/*.pt" --kinds color --basis cache/qwen_l24_pca128.pt \
  --max-past 8 --lam-lat 0 --lam-con 0 --lr $LR --steps 4000 --batch 4 --eval-every 500 --max-eval 300 --seed 2 --out $S/$N > $S/$N.log 2>&1
log "couleur v : $N fini (code $?)"
