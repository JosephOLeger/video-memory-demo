# Orchestrateur v3 (tour). Ajout du bras D (rappel seul, portes bloquees) : separe
# "bloquer les portes aide" de "empecher le collapse aide" (revue empirique 12/10).
# File de 12 runs (4 bras x 3 seeds) sur 3 cartes. 1) attend les 4 shards de cibles (tour 0-2, .135 c3 : 3),
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
log "orch v3 start"
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
UC="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --fill-choices 4,8,16 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500"
CL="--clamp-eta 0.995 --clamp-alpha 1.27e-4"
JOBS=()
for seed in 0 1 2; do
  JOBS+=("A_s$seed|--seed $seed --dense-lambda 0")
  JOBS+=("B_s$seed|--seed $seed --dense-lambda 1")
  JOBS+=("C_s$seed|--seed $seed --dense-lambda 1 $CL")
  JOBS+=("D_s$seed|--seed $seed --dense-lambda 0 $CL")
done
printf "%s\n" "${JOBS[@]}" > $U/queue.txt
worker() { # carte
  local card=$1
  while true; do
    local line
    line=$(flock $U/queue.lock sh -c 'l=$(head -1 '$U'/queue.txt); [ -n "$l" ] && sed -i 1d '$U'/queue.txt; echo "$l"')
    [ -z "$line" ] && break
    local name=${line%%|*} args=${line#*|}
    log "U2 $name carte $card"
    HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$card $PY -m titans.graft.train_load $UC $args --out $U/$name > $U/$name.log 2>&1
  done
}
worker 0 & worker 1 & worker 2 &
wait
log "U2 fini, distillation"
C="--targets $D/targets_[0-3].pt --basis cache/qwen_l24_pca128.pt --steps 8000"
for spec in "0 unit 0" "1 unit_squash 0" "2 unit 1"; do set -- $spec
  HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1 nohup $PY -m titans.graft.train_distill $C --muon $2 --seed $3 --out $D/dist_$2_s$3 > $D/dist_$2_s$3.log 2>&1 &
done
log "3 runs de distillation lances"
