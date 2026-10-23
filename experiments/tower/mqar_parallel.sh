#!/bin/bash
# Kill the sequential MQAR diagnostic and fan the 5 remaining cells across
# the 3 free cards. Completed cells (seeds 0,1 both + seed2/None) stay in
# mqar_diag.json; the 5 relaunched cells write mqar_diag_s*_rc*.json.
PY=~/miniforge3/envs/titans/bin/python
LOG=~/work/iclr/logs_theory
pgrep -f "mqar_diag.py" | grep -vw $$ | xargs -r kill -9      # not this script, not _one runners
cd ~/work/iclr/titans-mac || exit 1
export PYTHONPATH=src
# card 0: (seed2,rc1) then (seed4,rc1)
CUDA_VISIBLE_DEVICES=0 nohup bash -c "$PY ~/work/iclr/mqar_diag_one.py --seed 2 --rc 1 && $PY ~/work/iclr/mqar_diag_one.py --seed 4 --rc 1" </dev/null >$LOG/mqar_card0.log 2>&1 &
# card 1: (seed3,None) then (seed4,None)
CUDA_VISIBLE_DEVICES=1 nohup bash -c "$PY ~/work/iclr/mqar_diag_one.py --seed 3 --rc None && $PY ~/work/iclr/mqar_diag_one.py --seed 4 --rc None" </dev/null >$LOG/mqar_card1.log 2>&1 &
# card 2: (seed3,rc1)
CUDA_VISIBLE_DEVICES=2 nohup $PY ~/work/iclr/mqar_diag_one.py --seed 3 --rc 1 </dev/null >$LOG/mqar_card2.log 2>&1 &
sleep 3
echo "=== running mqar_diag_one processes ==="
pgrep -af "mqar_diag_one.py" | grep -v "bash -c"
echo "=== cards ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
