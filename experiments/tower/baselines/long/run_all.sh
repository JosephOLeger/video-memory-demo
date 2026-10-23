#!/bin/bash
# 23/10 : prep puis eval complete, carte 1 seulement (partagee : on attend 10 Go libres).
cd ~/work/iclr/baselines/long
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/titans/bin/python
wait_free() {
  until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 10500 ]; do sleep 60; done
}
wait_free
PYTHONPATH=$HOME/work/titans-mac/src $PY prep.py --out data > prep.log 2>&1 || exit 1
wait_free
PYTHONPATH=$HOME/work/titans-mac/src:$HOME/work/iclr/baselines $PY eval_long.py --data data \
    --conds orig,L8,D2,D3,L32,L80 --out res > eval.log 2>&1
$PY table_long.py res > tables.md
