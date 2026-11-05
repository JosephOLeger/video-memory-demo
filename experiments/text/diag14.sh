#!/bin/bash
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py --past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3 --arm A --steps 1500 --write-attnpool 8 --oracle-mix-steps 800 --oracle-mix-stream --proj-hidden 1024 --proj-residual --down-lr-mult 0.01 --init runs_ab/oraclewB_projres_dn001.pt --out runs_ab/smixA_projres > runs_ab/smixA_projres.log 2>&1
echo DIAG14-FINI
