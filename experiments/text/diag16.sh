#!/bin/bash
# deep smixA seeds 8-9 (pre-registered confirmation), full chains (reliability 8 vs 8). Pre-registered 25/10.
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3"
SM="--write-attnpool 8 --oracle-mix-steps 800 --oracle-mix-stream"
for s in 8 9; do
  gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm B --seed $s --steps 500 --write-stride 1 --oracle-write --out runs_ab/oraclewB_s$s > runs_ab/oraclewB_s$s.log 2>&1 || continue
  gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --seed $s --steps 1500 $SM --init runs_ab/oraclewB_s$s.pt --out runs_ab/smixA_s$s > runs_ab/smixA_s$s.log 2>&1
done
echo DIAG16-FINI
