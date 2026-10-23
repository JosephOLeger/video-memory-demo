#!/bin/bash
# 16/10 (accord Josep) : greffe en flux COULEUR SEULE puis FORMES, carte 2 de la tour, des que
# hkf_unit_s3 (PID 767017) a fini ; puis la carte 2 reprend la file des graines forte charge.
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream; L=$HOME/work/iclr/load; Q=$L/hkq.txt
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
while kill -0 767017 2>/dev/null; do sleep 60; done
log "graines : hkf_unit_s3 fini (carte 2, fin constatee par color_card2)"
log "couleur : precalcul des cibles carte 2 (2 processus)"
for sh in 0 1; do
  $PY -m titans.graft.stream_color_targets --reuse $S/tg --data data/finevideo --out $S/tgc --n-color 4 --shard $sh --nshards 2 > $S/tgc_$sh.log 2>&1 &
done
wait
log "couleur : cibles finies ($(ls $S/tgc | wc -l) videos)"
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
log "couleur : C1_color_s0 carte 2"
$PY -m titans.graft.train_stream --targets "$S/tgc/*.pt" --kinds color "${CO[@]}" --out $S/C1_color_s0 > $S/C1_color_s0.log 2>&1
log "couleur : C1_color_s0 fini (code $?)"
log "couleur : C2_shape_s0 carte 2 (depart de C1)"
$PY -m titans.graft.train_stream --targets "$S/tg/*.pt" --kinds shape "${CO[@]}" --init $S/C1_color_s0_at4000.pt --out $S/C2_shape_s0 > $S/C2_shape_s0.log 2>&1
log "couleur : C2_shape_s0 fini (code $?)"
# la carte 2 reprend la file des graines forte charge (meme commande que hk_seeds_tower.sh)
C="--basis cache/qwen_l24_pca128.pt --bank $L/bank512.pt --muon unit --dense-lambda 0 --fill-choices 8 --max-writes 4000"
F="--clamp-eta 0.995 --clamp-alpha 0.000127"
R="--steps 30000 --stop-after 8000 --train-k 1,2,4,8,16,32,64,128,256 --advance-acc 0.35 --eval-every 1000 --eval-n 96 --eval-k 1,4,16,64,128,256,512 --save-every 2000"
while true; do
  line=$(flock $L/hkq.lock sh -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\"")
  [ -z "$line" ] && break
  arm=${line% *}; s=${line#* }; X=""
  [ $arm = hkf ] && X="$F"
  log "graines : ${arm}_unit_s$s carte 2"
  $PY -m titans.graft.train_load $C $X $R --seed $s --out $L/${arm}_unit_s$s > $L/${arm}_unit_s$s.log 2>&1
  log "graines : ${arm}_unit_s$s fini (code $?)"
done
