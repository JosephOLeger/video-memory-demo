#!/bin/bash
# v3 deep oracle on all fact tokens, 1500 steps (balanced stories, slide 256, curriculum). 25/10.
cd ~/work/iclr/text
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
mkdir -p runs3
C="--past 4096 --past-choices 4096,8192 --dmax 1500 --eval-pasts 4096,8192 --eval-every 250 --eval-groups 10 --final-eval-groups 20 --eval-dists 300 --lr 3e-3 --past-chunk 256 --past-mode slide"
gate; CUDA_VISIBLE_DEVICES=3 TEXT_GEN=gen_story3 nice -n 10 $PY train_text_ab.py $C --arm B --write-stride 1 --oracle-write --steps 1500 --out runs3/v3_oraclewB_1500 > runs3/v3_oraclewB_1500.log 2>&1
echo DIAG18-FINI
