export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$HOME/work/titans-mac/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/titans/bin/python
C="--data data_ab_smoke --steps 30 --eval-every 0 --final-eval-groups 3 --final-eval-dists 0,120 --save-every 0"
$PY train_ab.py --arm A $C --out runs_ab/smoke_A > logs_ab/smoke_A.log 2>&1
$PY train_ab.py --arm B $C --out runs_ab/smoke_B > logs_ab/smoke_B.log 2>&1
$PY train_ab.py --arm A --mem-type pool $C --out runs_ab/smoke_poolA > logs_ab/smoke_poolA.log 2>&1
echo SMOKE-TRAIN-DONE
