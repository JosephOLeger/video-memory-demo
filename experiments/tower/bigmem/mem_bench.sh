#!/bin/bash
O=$HOME/work/iclr/distill/orch.log
until grep -q "cgb : smoke cibles fini" $O; do sleep 60; done
cd $HOME/work/titans-mac
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $HOME/miniforge3/envs/titans/bin/python $HOME/work/iclr/bigmem/mem_bench.py 32 > $HOME/work/iclr/bigmem/mem_bench.log 2>&1
echo "$(date +%F_%T) bigmem : mem_bench fini" >> $O
