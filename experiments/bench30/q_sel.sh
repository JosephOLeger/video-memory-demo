#!/bin/bash
# q_sel.sh (03/11): write-selection diagnostic, host surprise (nll) vs random, on the reader trained with oracle
# writes (w_dp), 300 dev records. After q_m512b (same lock); then releases the lock and runs q_mdp_long.sh.
# s3: sup load curve (supmix at N 10..512); s4: write strength of the learned write (m_dp).
# s1: budget = sup oracle count, token units (nll, nllsent, rand) + sup + omem.  s2: budget 64, sentence units.
cd ~/work/iclr/bench30
exec 9>q135width.lock; flock 9
export PYTHONPATH=~/work/iclr/bench30:~/work/iclr/text PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
T=~/work/iclr/text; CARD=3; PY=~/work/titans-mac/.venv/bin/python; L=~/work/iclr/bench30/jobs.log
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $T/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
E="--init runs/w_dp.pt --jsonl-eval evalsets/b30_dev_4k_1ep_sp.jsonl --jsonl-n 300"
run() { echo "START $(date +%F_%T) card$CARD $2" >> $L; CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30_sel.py $3 > runs/$2.log 2>&1; echo "END $(date +%F_%T) $2 rc=$?" >> $L; }
run 9000 sel_s1_w_dp "$A $E --jsonl-conds sup,omem,nll,nllsent,rand --sel-budget sup --out runs/sel_s1_w_dp"
run 9000 sel_s2_w_dp "$A $E --jsonl-conds sup,nll,rand --sel-budget 64 --sel-unit sent --out runs/sel_s2_w_dp"
run 10800 supmix_w_dp "$A $E --jsonl-conds sup,mem,omem,supmix --sel-budget-list 10,32,64,128,256,512 --out runs/supmix_w_dp"
run 5400 ws_m_dp "$A --init runs/m_dp.pt --jsonl-eval evalsets/b30_dev_4k_1ep_sp.jsonl --jsonl-n 300 --jsonl-conds mem,omem --log-write-strength --out runs/ws_m_dp"
echo "QSEL-FINI $(date +%F_%T)" >> $L
exec 9>&-
exec bash q_mdp_long.sh
