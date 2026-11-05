#!/bin/bash
# q_twostage.sh (03/11): two-stage training of the deep memory on the 30 types (tower card 1, after q_mdp_s2).
# The reader learned with oracle writes (w_dp) is FROZEN; only the write is trained (learned pooling, k/v projections,
# gates), with the same oracle-to-learned curriculum and steps as m_dp (joint) -> ts_w. Then the write is frozen and
# the reader is fine-tuned on the learned writes -> ts_r. Question: does freezing the reader stop the co-adaptation
# that lets the joint run ignore its memory (m_dp: mem 12.7 vs omem 11.4)?
cd ~/work/iclr/bench30
exec 9>qt1.lock; flock 9
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=1; PY=~/miniforge3/envs/titans/bin/python; L=~/work/iclr/bench30/jobs_tower.log
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
READ="q_proj,memory.q_proj,norm_r,up.,memory.memory.params"
WRITE="attn_pool,memory.k_proj,memory.v_proj,memory.alpha_head,memory.eta_head,memory.theta_head"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
D1=evalsets/b30_dev_4k_1ep_sp.jsonl; D2=evalsets/b30_dev_4k.jsonl
echo "QTWO-START $(date +%F_%T)" >> $L
B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 1800 ts_smoke "$A --freeze $READ --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/w_dp.pt --steps 20 --eval-every 0 --out runs/ts_smoke"
if ! grep -q "TRAIN-TEXT-AB-FINI" runs/ts_smoke.log || ! grep -q "freeze: " runs/ts_smoke.log; then echo "QTWO-ABORT $(date +%F_%T) smoke failed" >> $L; exit 1; fi
[ -f runs/ts_w.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 18000 ts_w "$A --freeze $READ --oracle-mix-steps 800 --oracle-mix-stream --preserve 1.0 --init runs/w_dp.pt --steps 2000 --eval-every 0 --out runs/ts_w"
run 7200 fin1_ts_w "$A --init runs/ts_w.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_ts_w"
run 10800 fin2_ts_w "$A --init runs/ts_w.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_ts_w"
[ -f runs/ts_r.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run 10800 ts_r "$A --freeze $WRITE --preserve 1.0 --init runs/ts_w.pt --steps 1000 --warmup 10 --eval-every 0 --out runs/ts_r"
run 7200 fin1_ts_r "$A --init runs/ts_r.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_ts_r"
run 10800 fin2_ts_r "$A --init runs/ts_r.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_ts_r"
echo "QTWO-FINI $(date +%F_%T)" >> $L
