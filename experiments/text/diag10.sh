#!/bin/bash
# smixA seeds 1 and 2, full chain (oracle reader arm B sN, then smix A sN), as smixA_s0. Waits for diag9.
cd ~/work/iclr/text
while pgrep -f "bash diag9.sh" >/dev/null; do sleep 60; done
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--past 2048 --dmax 600 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3"
SM="--write-attnpool 8 --oracle-mix-steps 800 --oracle-mix-stream"
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --steps 12 --eval-every 0 $SM --init runs_ab/oraclewB_s1.pt --out runs_ab/smoke_gradlog > runs_ab/smoke_gradlog.log 2>&1 || { echo SMOKE-FAIL; exit 1; }
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --seed 1 --steps 1500 $SM --init runs_ab/oraclewB_s1.pt --out runs_ab/smixA_s1 > runs_ab/smixA_s1.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm B --seed 2 --steps 500 --write-stride 1 --oracle-write --out runs_ab/oraclewB_s2 > runs_ab/oraclewB_s2.log 2>&1
gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C --arm A --seed 2 --steps 1500 $SM --init runs_ab/oraclewB_s2.pt --out runs_ab/smixA_s2 > runs_ab/smixA_s2.log 2>&1
echo DIAG10-FINI
