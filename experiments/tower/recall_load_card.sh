#!/bin/bash
CARD=$1; shift
cd ~/work/iclr/titans-mac; export PYTHONPATH=src
P=~/miniforge3/envs/titans/bin/python
for cell in "$@"; do
  n=${cell%%:*}; rc=${cell##*:}
  echo "[card $CARD] START n=$n rc=$rc $(date +%T)"
  CUDA_VISIBLE_DEVICES=$CARD $P ~/work/iclr/recall_load_one.py --n $n --rc $rc --seed 0 --tag load
  echo "[card $CARD] END   n=$n rc=$rc $(date +%T)"
done
echo "[card $CARD] ALL DONE $(date +%T)"
