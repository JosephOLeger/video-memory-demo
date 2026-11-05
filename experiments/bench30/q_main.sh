#!/bin/bash
# bench30 main queue (agent bench30, 29/10). .135 CARD 3 ONLY. Self-chaining: smoke (gated), cache training,
# evals (two concurrent eval processes when both are short-context), deep arm, baselines.
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
while kill -0 2416342 2>/dev/null; do sleep 30; done           # baseline BABILong eval (q_base.sh) first
# 1. cache smoke, gated
run 2400 smoke_kv2 "$KV --preserve 1.0 --init $T/runs_bl/offl_topk_pres.pt --past-choices 2048,7100 --steps 30 --eval-every 0 --out runs/smoke_kv2"
ok smoke_kv2 || { echo "SMOKE FAILED, stop" >> $L; exit 1; }
# 2. cache training on bench30 (+25 % BABILong train stories), preservation on
run 36000 b30_kv "$KV --preserve 1.0 --init $T/runs_bl/offl_topk_pres.pt --steps 1500 --eval-every 0 --out runs/b30_kv"
if ok b30_kv; then
  run 14400 off_b30_kv "$KV --init runs/b30_kv.pt --official $BL10 --official-conds mem,omem --out runs/off_b30_kv" &
  sleep 90
  run 14400 ev_b30_kv "$KV --init runs/b30_kv.pt --jsonl-eval evalsets/b30_dev_4k.jsonl --jsonl-conds mem,omem --out runs/ev_b30_kv"
  wait
  run 14400 ext_b30_kv "$KV --init runs/b30_kv.pt --jsonl-eval evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_b30_kv" &
  sleep 90
fi
run 14400 ev_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval evalsets/b30_dev_4k.jsonl --jsonl-conds win,full,mem,omem --out runs/ev_base_kv"
wait
# 3. deep arm: smoke, then training from smixA_deep_ct_pres
run 2400 smoke_dp "$DP --preserve 1.0 --init ckpt/smixA_deep_ct_pres.pt --past-choices 2048,7100 --steps 20 --eval-every 0 --out runs/smoke_dp"
if ok smoke_dp; then
  run 36000 b30_dp "$DP --preserve 1.0 --init ckpt/smixA_deep_ct_pres.pt --steps 1500 --eval-every 0 --out runs/b30_dp"
  if ok b30_dp; then
    run 14400 off_b30_dp "$DP --init runs/b30_dp.pt --official $BL10 --official-conds mem,omem --out runs/off_b30_dp" &
    sleep 90
    run 14400 ev_b30_dp "$DP --init runs/b30_dp.pt --jsonl-eval evalsets/b30_dev_4k.jsonl --jsonl-conds mem,omem --out runs/ev_b30_dp"
    wait
    run 14400 ext_b30_dp "$DP --init runs/b30_dp.pt --jsonl-eval evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_b30_dp" &
    sleep 90
  fi
fi
run 14400 ev_base_dp "$DP --init ckpt/smixA_deep_ct_pres.pt --jsonl-eval evalsets/b30_dev_4k.jsonl --jsonl-conds mem,omem --out runs/ev_base_dp"
wait
# 4. baselines on the external benchmarks (window, full context, untrained-on-bench30 memories), long context: alone
run 18000 ext_base_kv "$KV --init $T/runs_bl/offl_topk_pres.pt --jsonl-eval evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl --jsonl-conds win,full,mem,omem --jsonl-gen 24 --out runs/ext_base_kv"
run 14400 ext_base_dp "$DP --init ckpt/smixA_deep_ct_pres.pt --jsonl-eval evalsets/lme_oracle.jsonl,evalsets/locomo.jsonl --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_base_dp" &
sleep 90
run 14400 off_base_dp "$DP --init ckpt/smixA_deep_ct_pres.pt --official qa9,qa10:4k --official-conds mem,omem --out runs/off_base_dp"
wait
echo "QMAIN-FINI $(date +%F_%T)" >> $L
