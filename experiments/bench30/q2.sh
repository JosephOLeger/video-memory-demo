#!/bin/bash
# bench30 queue 2 (agent bench30, 29/10 17 h). Re-prioritised after b30_kv (mem = omem on the dev set):
# LOW-LOAD training (B30_NEP 1 episode 70 %, 2 episodes 30 %) for cache and deep, evals, then references.
cd ~/work/iclr/bench30
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
PY=~/work/titans-mac/.venv/bin/python
L=~/work/iclr/bench30/jobs.log
T=~/work/iclr/text
KV="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --lr 1e-3 --past-chunk 256 --past-mode slide --long-q 1 --mem-type kvfull --head-lr-mult 1 --layout official --slot-pos --oracle-sent-ends 1000 --write-topk"
DP="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official"
run() { echo "START $(date +%F_%T) $2" >> $L; CUDA_VISIBLE_DEVICES=3 nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
ok() { grep -q TRAIN-TEXT-AB-FINI runs/$1.log && ! grep -q NON-FINITE runs/$1.log; }
BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
D2=evalsets/b30_dev_4k.jsonl; D1=evalsets/b30_dev_4k_1ep.jsonl; EXT=evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl
B11=evalsets/bl1120_4k_reconstructed.jsonl
while kill -0 2795033 2>/dev/null; do sleep 30; done
B30_NEP=1:0.7,2:0.3 run 36000 b30_kvL "$KV --preserve 1.0 --init $T/runs_bl/offl_topk_pres.pt --steps 1500 --eval-every 0 --out runs/b30_kvL"
run 14400 ev1_b30_kvL "$KV --init runs/b30_kvL.pt --jsonl-eval $D1 --jsonl-conds mem,omem --out runs/ev1_b30_kvL" &
sleep 90
run 14400 ev1_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval $D1 --jsonl-conds win,full,mem,omem --out runs/ev1_base_kv"
wait
run 2400 smoke_dp "$DP --preserve 1.0 --init ckpt/smixA_deep_ct_pres.pt --past-choices 2048,7100 --steps 20 --eval-every 0 --out runs/smoke_dp"
ok smoke_dp && B30_NEP=1:0.7,2:0.3 run 36000 b30_dpL "$DP --preserve 1.0 --init ckpt/smixA_deep_ct_pres.pt --steps 1500 --eval-every 0 --out runs/b30_dpL"
[ -f runs/b30_dpL.pt ] && run 14400 ev1_b30_dpL "$DP --init runs/b30_dpL.pt --jsonl-eval $D1 --jsonl-conds mem,omem --out runs/ev1_b30_dpL" &
sleep 90
run 14400 off_b30_kvL "$KV --init runs/b30_kvL.pt --official $BL10 --official-conds mem,omem --out runs/off_b30_kvL"
wait
[ -f runs/b30_dpL.pt ] && run 14400 off_b30_dpL "$DP --init runs/b30_dpL.pt --official $BL10 --official-conds mem,omem --out runs/off_b30_dpL" &
sleep 90
run 14400 ev_b30_kvL "$KV --init runs/b30_kvL.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/ev_b30_kvL"
wait
[ -f runs/b30_dpL.pt ] && run 14400 ev_b30_dpL "$DP --init runs/b30_dpL.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/ev_b30_dpL" &
sleep 90
run 14400 ev_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval $D2 --jsonl-conds win,full,mem,omem --out runs/ev_base_kv"
wait
run 14400 ev1_base_dp "$DP --init ckpt/smixA_deep_ct_pres.pt --jsonl-eval $D1 --jsonl-conds mem,omem --out runs/ev1_base_dp" &
sleep 90
run 14400 ev1_b30_kv "$KV --init runs/b30_kv.pt --jsonl-eval $D1 --jsonl-conds mem,omem --out runs/ev1_b30_kv"
wait
run 18000 ext_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval $EXT --jsonl-conds win,full,mem,omem --jsonl-gen 24 --out runs/ext_base_kv"
run 14400 ext_b30_kvL "$KV --init runs/b30_kvL.pt --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_b30_kvL" &
sleep 90
[ -f runs/b30_dpL.pt ] && run 14400 ext_b30_dpL "$DP --init runs/b30_dpL.pt --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_b30_dpL"
wait
run 14400 bl_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval $B11 --jsonl-n 500 --jsonl-conds win,full,mem,omem --out runs/bl_base_kv" &
sleep 90
run 14400 bl_b30_kvL "$KV --init runs/b30_kvL.pt --jsonl-eval $B11 --jsonl-n 500 --jsonl-conds mem,omem --out runs/bl_b30_kvL"
wait
[ -f runs/b30_dpL.pt ] && run 14400 bl_b30_dpL "$DP --init runs/b30_dpL.pt --jsonl-eval $B11 --jsonl-n 500 --jsonl-conds mem,omem --out runs/bl_b30_dpL" &
sleep 90
run 14400 ext_base_dp "$DP --init ckpt/smixA_deep_ct_pres.pt --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_base_dp"
wait
echo "Q2-FINI $(date +%F_%T)" >> $L
