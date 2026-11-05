#!/bin/bash
# q_host.sh (05/11): effect of the graft on the host outside its task (train_b30.py --host-eval). Standard short
# prompts (MMLU, ARC, GSM8K, TriviaQA, MMBench, POPE, ChartQA, TextVQA) and PG19 test continuation, graft off / on with
# a fresh memory / on with the memory of an unrelated story. Scored by score_host.py.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log; M=m_dp_long
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
H=evalsets/host; F=$H/mmlu.jsonl,$H/arc.jsonl,$H/triviaqa.jsonl,$H/pg19test.jsonl,$H/mmbench.jsonl,$H/pope.jsonl,$H/chartqa.jsonl,$H/textvqa.jsonl,$H/gsm8k.jsonl
echo "START $(date +%F_%T) card$CARD host_$M" >> $L
CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout 43200 $PY train_b30.py $A --init runs/$M.pt --host-eval $F --host-conds off,init,carry,mem,omem --out runs/host_$M > runs/host_$M.log 2>&1
echo "END $(date +%F_%T) host_$M rc=$?" >> $L
