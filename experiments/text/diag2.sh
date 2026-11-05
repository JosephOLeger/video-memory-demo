#!/bin/bash
cd ~/work/iclr/text
while pgrep -u josephleger -f "diag_s1_lr3e-3" >/dev/null; do sleep 30; done
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python probe_pca.py 300 > runs_ab/probe_pca.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --steps 150 --eval-every 150 --eval-groups 10 --final-eval-groups 10 --eval-dists 300 --write-stride 1 --lr 3e-3 --oracle-write --out runs_ab/diag_oraclew > runs_ab/diag_oraclew.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --steps 150 --eval-every 150 --eval-groups 10 --final-eval-groups 10 --eval-dists 300 --write-stride 1 --lr 3e-3 --mem-type pool --pool 1 --slots 2048 --out runs_ab/diag_poolall > runs_ab/diag_poolall.log 2>&1
echo DIAG2-FINI
