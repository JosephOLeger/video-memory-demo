#!/bin/bash
# deep (unit) smixA seed 3, full chain, for the reliability count.
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3"
SM="--write-attnpool 8 --oracle-mix-steps 800 --oracle-mix-stream"
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm B --seed 3 --steps 500 --write-stride 1 --oracle-write --out runs_ab/oraclewB_s3 > runs_ab/oraclewB_s3.log 2>&1 || exit 1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --seed 3 --steps 1500 $SM --init runs_ab/oraclewB_s3.pt --out runs_ab/smixA_s3 > runs_ab/smixA_s3.log 2>&1
echo DIAG12-FINI
