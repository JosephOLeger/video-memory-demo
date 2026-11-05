#!/bin/bash
# projres s7, full chain (pre-registered confirmation). 25/10.
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3"
PJ="--proj-hidden 1024 --proj-residual --down-lr-mult 0.01"
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm B --seed 7 --write-stride 1 --oracle-write $PJ --init runs_ab/oraclewB_s7.pt --steps 1000 --out runs_ab/oraclewB_projres_s7 > runs_ab/oraclewB_projres_s7.log 2>&1 || exit 1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --seed 7 --steps 1500 --write-attnpool 8 --oracle-mix-steps 800 --oracle-mix-stream $PJ --init runs_ab/oraclewB_projres_s7.pt --out runs_ab/smixA_projres_s7 > runs_ab/smixA_projres_s7.log 2>&1
echo DIAG17-FINI
