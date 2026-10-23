#!/bin/bash
# wait for a card with >= 12 GB free, then run the probe on it
cd ~/work/iclr/audit
while true; do
  card=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=12288{print $1; exit}')
  [ -n "$card" ] && break
  sleep 60
done
echo "card $card $(date)" > probe.card
CUDA_VISIBLE_DEVICES=$card PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/titans/bin/python ceiling_probe.py init_count,first_left,sign_number,hat_color,final_count 8 100 > probe.log 2>&1
echo "EXIT $?" >> probe.log
