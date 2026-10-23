#!/bin/bash
# 15/10 03:30 : carte 0 rejoint la file ATLAS apres le QCM scratch (ablations Omega seul / phi seul du bras A).
D=$HOME/work/iclr/distill; U=$HOME/work/iclr/atlas; Q=$U/queue.txt
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
AT="--omega-window 8 --omega-stride 2 --feature-degree 2"
BASE="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500 $AT"
until grep -q "mcq : mcq_scratch_s0 fini" $D/orch.log; do sleep 120; done
while true; do
  line=$(flock $U/q.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
  [ -z "$line" ] && break
  name=${line%%|*}; args=${line#*|}
  log "atlas $name carte 0"
  CUDA_VISIBLE_DEVICES=0 $PY -m titans.graft.train_load $BASE $args --out $U/$name > $U/$name.log 2>&1
  log "atlas $name fini (code $?)"
done
