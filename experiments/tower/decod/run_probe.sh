#!/bin/bash
D=$HOME/work/iclr/decod
until grep -q "DECOD-EXTRACT-FINI" $D/extract.log; do sleep 60; done
cd $HOME/work/titans-mac
echo "$(date +%F_%T) decod : extraction finie, sonde carte 1" >> $HOME/work/iclr/distill/orch.log
CUDA_VISIBLE_DEVICES=1 $HOME/miniforge3/envs/titans/bin/python $D/decod_probe.py --feat $D/feat --basis cache/qwen_l24_pca128.pt --out $D/decod.json > $D/probe.log 2>&1
echo "$(date +%F_%T) decod : sonde finie (code $?)" >> $HOME/work/iclr/distill/orch.log
