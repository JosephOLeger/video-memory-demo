#!/bin/bash
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --steps 1500 --write-attnpool 8 --oracle-mix-steps 500 --lr 3e-3 --init runs_ab/diag_oraclew_at500.pt --out runs_ab/diag_wap8_mix > runs_ab/diag_wap8_mix.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --steps 1000 --write-stride 1 --oracle-write --lr 3e-3 --init runs_ab/diag_oraclew_at500.pt --out runs_ab/diag_oraclew_cont > runs_ab/diag_oraclew_cont.log 2>&1
echo DIAG8-FINI
