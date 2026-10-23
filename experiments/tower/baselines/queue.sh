#!/bin/bash
# 22/10 : temoins a budget egal (bl.py), recette exacte de C1_color_s0, carte 2 seulement, 2 couloirs.
B=$HOME/work/iclr/baselines; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/work/titans-mac/src:$B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2
CO=(--targets "$S/tgc/*.pt" --kinds color --basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
next() { ( flock 9; j=$(head -1 $B/jobs.txt); [ -n "$j" ] && sed -i 1d $B/jobs.txt; echo "$j" ) 9>$B/jobs.lock; }
lane() { while true; do j=$(next); [ -z "$j" ] && break; set -- $j; t=$1 s=$2
  echo "$(date +%F_%T) lane $L start $t s$s" >> $B/queue.log
  $PY -m bl --mem-type $t "${CO[@]}" --seed $s --out $B/${t}_s$s > $B/${t}_s$s.log 2>&1
  echo "$(date +%F_%T) lane $L fin $t s$s code $?" >> $B/queue.log; done; }
L=$1 lane
