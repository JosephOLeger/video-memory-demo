#!/bin/bash
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --write-stride 1 --oracle-write --init runs_ab/diag_oraclew_at500.pt --eval-only --out runs_ab/evalonly_oraclew500 > runs_ab/evalonly_oraclew500.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python probe_pca.py 300 > runs_ab/probe_pca2.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --steps 1500 --write-pool 8 --lr 3e-3 --out runs_ab/diag_wp8_lr3e-3 > runs_ab/diag_wp8_lr3e-3.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 /home/josephleger/work/titans-mac/.venv/bin/python train_text_ab.py --arm B --past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --steps 1000 --write-stride 1 --lr 3e-3 --mem-type pool --pool 1 --slots 2048 --out runs_ab/diag_poolall > runs_ab/diag_poolall.log 2>&1
echo DIAG5-FINI
