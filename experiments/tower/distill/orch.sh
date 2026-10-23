# Orchestrateur (tourne sur la tour) : attend les 4 shards (tour 0-2, .135 carte 3 : 3),
# rapatrie le shard 3, lance 3 runs sur la tour. JAMAIS la carte 0 de .135 (partagee).
D=$HOME/work/iclr/distill
R=josephleger@172.31.4.135
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
log start
while true; do
  n=0
  for s in 0 1 2; do grep -q TARGETS-FINIE $D/targets_$s.log 2>/dev/null && n=$((n+1)); done
  for s in 3; do ssh -o BatchMode=yes $R "grep -q TARGETS-FINIE work/iclr/distill/targets_$s.log" 2>/dev/null && n=$((n+1)); done
  [ $n -ge 4 ] && break
  sleep 300
done
log "shards finis, echange"
scp -q $R:work/iclr/distill/targets_3.pt $D/ || log "ECHEC rapatriement"
log "echange fait"
C="--targets $D/targets_[0-3].pt --basis cache/qwen_l24_pca128.pt --steps 8000"
cd ~/work/titans-mac
for spec in "0 unit 0 " "1 unit_squash 0 " "2 unit 1 "; do set -- $spec
  HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1 nohup ~/miniforge3/envs/titans/bin/python -m titans.graft.train_distill $C --muon $2 --seed $3 --out $D/dist_$2_s$3 > $D/dist_$2_s$3.log 2>&1 &
done
log "3 runs lances sur la tour"
