#!/bin/bash
cd ~/work/iclr/ptest
~/miniforge3/envs/titans/bin/python -I scripts/punzip.py && echo UNZIP_OK $(ls data/videos | wc -l)
CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/titans/bin/python -I scripts/eval_ptest.py --questions questions_sub3000.jsonl --out res_sub3000.jsonl
