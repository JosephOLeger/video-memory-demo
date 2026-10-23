#!/bin/bash
# 21/10 matin : C1v s1 a eu son declic au pas ~3300 (26 -> 58 -> 70 %). Le levier semble etre le nombre de pas.
# On reprend les deux graines C1v qui ont echoue en 4000 pas, pour 4000 pas de plus (nouveau warmup + cosinus, lr 3e-3).
#   carte 1 : C1vc_color_s0 depuis C1v_color_s0_at4000 ; carte 2 : C1vc_color_s3 depuis C1v_color_s3_at4000
S=$HOME/work/iclr/stream; O=$HOME/work/iclr/distill/orch.log
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $O; }
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
run() { local c=$1 s=$2 n=C1vc_color_s$2
  log "couleur v : $n carte $c"
  ( cd $HOME/work/titans-mac && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream --targets "$S/tgc/*.pt" --variants "$S/tgv/*.pt" --kinds color "${CO[@]}" --seed $s --init $S/C1v_color_s${s}_at4000.pt --out $S/$n > $S/$n.log 2>&1 )
  log "couleur v : $n fini (code $?)"; }
run 1 0 & run 2 3 & wait
