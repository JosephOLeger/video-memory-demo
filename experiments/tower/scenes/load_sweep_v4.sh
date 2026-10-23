#!/bin/bash
# design sweep of the load cue (design groups 100-109, never used for certification); each config waits for a card
P=~/miniforge3/envs/titans/bin/python
for cfg in "first variants" "last variants" "first groups" "last groups"; do
  set -- $cfg
  MINFREE=12000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./launch_any.sh results/loadsweep_v4_$1_$2.log \
     $P ceiling_v4.py --count-even --families count,colour --gstart 100 --groups 10 --check-at 99 \
     --conds "oracle_all|L2,oracle_all|L5" --load-order $1 --load-src $2 --out results/loadsweep_v4_$1_$2.json
done
echo SWEEP-FINI
