#!/bin/bash
# smixA on the 1500-step v3 deep oracle (does the learned write keep majority?). 26/10.
cd ~/work/iclr/text
while ! grep -q DIAG18-FINI runs_ab/diag18.out 2>/dev/null; do sleep 60; done
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--past 4096 --past-choices 4096,8192 --dmax 1500 --eval-pasts 4096,8192 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3 --past-chunk 256 --past-mode slide"
gate; CUDA_VISIBLE_DEVICES=3 TEXT_GEN=gen_story3 nice -n 10 $PY train_text_ab.py $C --arm A --steps 1500 --write-attnpool 8 --oracle-mix-steps 800 --oracle-mix-stream --init runs3/v3_oraclewB_1500.pt --out runs3/v3_smixA_1500 > runs3/v3_smixA_1500.log 2>&1
echo DIAG19-FINI
