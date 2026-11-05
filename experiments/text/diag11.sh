#!/bin/bash
# kvfull seed 1, full chain (oracle value tokens s1, then smix A s1), to replicate deep-vs-cache.
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3 --mem-type kvfull --head-lr-mult 10"
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm B --seed 1 --oracle-write --steps 500 --out runs_ab/kvfull_oraclev_B_s1 > runs_ab/kvfull_oraclev_B_s1.log 2>&1 || exit 1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --seed 1 --steps 1500 --oracle-mix-steps 800 --oracle-mix-stream --init runs_ab/kvfull_oraclev_B_s1.pt --out runs_ab/smixA_kvfull_s1 > runs_ab/smixA_kvfull_s1.log 2>&1
echo DIAG11-FINI
