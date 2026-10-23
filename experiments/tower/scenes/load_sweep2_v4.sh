#!/bin/bash
P=~/miniforge3/envs/titans/bin/python
for o in first last; do
  MINFREE=12000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./launch_any.sh results/loadsweep2_v4_banner_$o.log \
     $P ceiling_v4.py --cue banner --count-even --families count,colour --gstart 100 --groups 10 --check-at 99 \
     --conds "oracle_all|L2,oracle_all|L5" --load-order $o --load-src variants --out results/loadsweep2_v4_banner_$o.json
done
echo SWEEP2-FINI
