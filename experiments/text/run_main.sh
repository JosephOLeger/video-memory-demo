#!/bin/bash
# card 3 ONLY; waits for >= 20 GB free before each launch
cd ~/work/iclr/text
PY=~/work/titans-mac/.venv/bin/python
FAMS=count_start,count_left,presence,place,order,main_object2,colour,number,object_name
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; echo "gate ok $(date +%T)"; }
gate
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=~/work/titans-mac/src nice -n 10 timeout 36000 $PY ladder_text.py --families $FAMS --groups 150 --gstart 10000 --out runs/main.jsonl > runs/main.log 2>&1
gate
CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 18000 $PY summ_text.py --families $FAMS --groups 10 --gstart 10000 --out runs/summ.jsonl > runs/summ.log 2>&1
echo "RUN-MAIN-FINI $(date +%T)"
