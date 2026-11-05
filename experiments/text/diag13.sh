#!/bin/bash
# key statistics on existing oracle checkpoints (no training): 9 vs ~340 writes, h64 vs h512.
cd ~/work/iclr/text
while pgrep -f "diag12.sh" >/dev/null; do sleep 60; done
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; }
PY=/home/josephleger/work/titans-mac/.venv/bin/python
C="--arm B --past 2048 --window 256 --write-stride 1 --oracle-write --eval-only --key-stats 100"
run() { gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY train_text_ab.py $C "$@" 2>&1 | grep -E "KEY-STATS|Error|identity" ; }
echo "== h64 trained on all fact tokens (diag_oraclew_cont), keys at all fact tokens"; run --init runs_ab/diag_oraclew_cont.pt --out runs_ab/ks1
echo "== same checkpoint, keys at the 9 value tokens"; run --init runs_ab/diag_oraclew_cont.pt --oracle-values --out runs_ab/ks2
echo "== h64 trained on value tokens (deep_oraclev1500_B), keys at the 9 value tokens"; run --init runs_ab/deep_oraclev1500_B.pt --oracle-values --out runs_ab/ks3
echo "== same checkpoint, keys at all fact tokens"; run --init runs_ab/deep_oraclev1500_B.pt --out runs_ab/ks4
echo "== h512 trained on all fact tokens, keys at all fact tokens"; run --init runs_ab/oraclewB_h512.pt --memory-hidden 512 --out runs_ab/ks5
echo DIAG13-FINI
