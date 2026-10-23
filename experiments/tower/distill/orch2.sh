# Orchestrateur v2 (tour). 1) attend les 4 shards de cibles (tour 0-2, .135 c3 : 3),
# rapatrie le shard 3 ; 2) U2 : 3 manches (seeds 0,1,2) x 3 bras sur les cartes 0-2 ;
# 3) distillation (3 runs). JAMAIS la carte 0 de .135.
# U2 : A = rappel seul ; B = rappel + loss dense (lambda 1, collapse appris) ;
#      C = comme B, eta et alpha bloques a leur init (intervention causale).
D=$HOME/work/iclr/distill
U=$HOME/work/iclr/u2
R=josephleger@172.31.4.135
PY=~/miniforge3/envs/titans/bin/python
mkdir -p $U
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
log "orch v2 start"
while true; do
  n=0
  for s in 0 1 2; do grep -q TARGETS-FINIE $D/targets_$s.log 2>/dev/null && n=$((n+1)); done
  ssh -o BatchMode=yes $R "grep -q TARGETS-FINIE work/iclr/distill/targets_3.log" 2>/dev/null && n=$((n+1))
  [ $n -ge 4 ] && break
  sleep 300
done
log "shards finis"
scp -q $R:work/iclr/distill/targets_3.pt $D/ || log "ECHEC rapatriement shard 3"
cd ~/work/titans-mac
UC="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --fill-choices 4,8,16 --eval-every 500 --eval-n 96 --eval-k 1,2,4 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500"
for seed in 0 1 2; do
  log "U2 manche seed $seed"
  HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $PY -m titans.graft.train_load $UC --seed $seed --dense-lambda 0 --out $U/A_s$seed > $U/A_s$seed.log 2>&1 &
  HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 $PY -m titans.graft.train_load $UC --seed $seed --dense-lambda 1 --out $U/B_s$seed > $U/B_s$seed.log 2>&1 &
  HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 $PY -m titans.graft.train_load $UC --seed $seed --dense-lambda 1 --clamp-eta 0.995 --clamp-alpha 1.27e-4 --out $U/C_s$seed > $U/C_s$seed.log 2>&1 &
  wait
done
log "U2 fini, distillation"
C="--targets $D/targets_[0-3].pt --basis cache/qwen_l24_pca128.pt --steps 8000"
for spec in "0 unit 0" "1 unit_squash 0" "2 unit 1"; do set -- $spec
  HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1 nohup $PY -m titans.graft.train_distill $C --muon $2 --seed $3 --out $D/dist_$2_s$3 > $D/dist_$2_s$3.log 2>&1 &
done
log "3 runs de distillation lances"
