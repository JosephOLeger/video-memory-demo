#!/bin/bash
# 21/10 : test court de la greffe en flux a 4 tailles d etat (64 Ko, 1 Mo, 4 Mo, 32 Mo), carte 0 apres la cascade.
# Mesure le temps par pas et le pic de memoire GPU. Donnees : couleur variantes (seulement pour le chrono).
S=$HOME/work/iclr/stream; B=$HOME/work/iclr/bigmem; O=$HOME/work/iclr/distill/orch.log
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) bigmem : $*" >> $O; }
until grep -q "cascade : d_K4_t16k_mean fini" $O; do sleep 120; done
log "smoke tailles carte 0"
for cfg in "128 64" "512 256" "1024 512" "2048 2048"; do set -- $cfg; n=smoke_d$1_h$2
  ( cd $HOME/work/titans-mac && PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $PY -m titans.graft.train_stream --targets "$S/tgc/*.pt" --variants "$S/tgv/*.pt" --kinds color \
     --basis cache/qwen_l24_stream_pca$1.pt --d-mem $1 --memory-hidden $2 --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 30 --batch 4 --eval-every 100000 --max-eval 20 \
     --seed 0 --out $B/$n > $B/$n.log 2>&1 ) & P=$!
  pk=0; while kill -0 $P 2>/dev/null; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); [ $m -gt $pk ] && pk=$m; sleep 5; done
  wait $P; c=$?
  echo "$n code $c pic_MiB $pk $(grep "s/pas" $B/$n.log | tail -1 | grep -o "[0-9.]* s/pas")" >> $B/smoke_sizes.txt
  log "$n code $c pic $pk MiB"
done
log "smoke tailles fini"
