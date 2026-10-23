#!/bin/bash
# card 3 ONLY; waits for >= 20 GB free before each launch. usage: run_v2.sh <stage>
cd ~/work/iclr/text
PY=~/work/titans-mac/.venv/bin/python
gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)" -lt 20000 ]; do sleep 60; done; echo "gate ok $(date +%T)"; }
case $1 in
cert) gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 7200 $PY ladder_text2.py --groups 15 --gstart 20000 --out runs2/cert.jsonl > runs2/cert.log 2>&1 ;;
main) gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 36000 $PY ladder_text2.py --groups 150 --gstart 10000 --out runs2/main.jsonl > runs2/main.log 2>&1 ;;
summ) gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 18000 $PY summ_text2.py --groups 20 --gstart 10000 --out runs2/summ.jsonl > runs2/summ.log 2>&1 ;;
babi) gate; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout 7200 $PY babilong.py --out runs2/babilong.jsonl > runs2/babilong.log 2>&1 ;;
esac
echo "STAGE-$1-FINI $(date +%T)"
