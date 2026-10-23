#!/bin/bash
cd ~/work/iclr/ptest
while kill -0 2356928 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/titans/bin/python -I scripts/eval_ptest.py --questions questions_rest.jsonl --out res_rest.jsonl
