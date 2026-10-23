# usage: tgt_queue.sh <python> <data> <card> <shard>
P=$1; DATA=$2; CARD=$3; SH=$4
cd ~/work/titans-mac
while [ "$(nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 60; done
while [ ! -f "$DATA/.complete" ]; do sleep 60; done
HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$CARD $P -m titans.graft.distill_targets --data $DATA --basis cache/qwen_l24_pca128.pt --shard $SH --nshards 4 --out $HOME/work/iclr/distill/targets > $HOME/work/iclr/distill/targets_$SH.log 2>&1
