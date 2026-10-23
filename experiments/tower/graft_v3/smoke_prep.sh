export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$HOME/work/titans-mac/src
PY=~/miniforge3/envs/titans/bin/python
$PY prep_ab.py fill --nfill 4 --out data_ab_smoke && $PY prep_ab.py train --nfill 4 --groups 40 --out data_ab_smoke && $PY prep_ab.py eval --groups 3 --out data_ab_smoke && echo SMOKE-PREP-DONE
