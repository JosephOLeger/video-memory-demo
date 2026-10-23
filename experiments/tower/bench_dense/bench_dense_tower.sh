#!/bin/bash
# 20/10 : banc dense (cf. bench_dense_135.sh), graines 3-7 (10 runs) sur la tour.
# Cartes 1 et 2 : 3 runs a la fois des que la reprise couleur (C1c) de la carte est finie.
# Carte 0 : 2 runs a la fois apres C1_color_s3. Code et caches = copie exacte de .135 (smoke identique).
cd $HOME/work/iclr/bench_dense/tm
PY=$HOME/miniforge3/envs/titans/bin/python
export PYTHONPATH=src
B=$HOME/work/iclr/bench_dense; Q=$B/qtour.txt; O=$HOME/work/iclr/distill/orch.log
log() { echo "$(date +%F_%T) banc dense : $*" >> $O; }
[ -f $Q ] || for s in 3 4 5 6 7; do echo "unit10k $s"; echo "usq10k $s"; done > $Q
stream_on() { local p; for p in $(pgrep -u "$USER" -f train_stream); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
slot() { # carte
  local c=$1
  while true; do
    line=$(flock $Q.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
    [ -z "$line" ] && return
    set -- $line; local arm=$1 s=$2 M="--muon unit"; [ $arm = usq10k ] && M="--muon unit_squash --muon-scale 350"
    log "mu_${arm}d_s$s carte $c"
    CUDA_VISIBLE_DEVICES=$c $PY -m titans.worldmodel.echo --label mu_${arm}d_s$s --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt --horizon 24 --steps 10000 --seed $s --eta-init 0.995 --theta-scale 0.03 --alpha-init 0.002 $M --frozen-every 1000 --out $B/mu_${arm}d_s$s.json --frozen-out $B/mu_${arm}d_s$s.pt > $B/mu_${arm}d_s$s.log 2>&1
    log "mu_${arm}d_s$s fini (code $?)"
  done
}
card() { # carte nslots
  local c=$1 n=$2 i
  if [ $c = 0 ]; then until grep -q "couleur : C1_color_s3 fini" $O; do sleep 300; done
  else until grep -q "couleur : C1c_color_s$c fini" $O; do sleep 300; done; fi
  while stream_on $c; do sleep 120; done
  for i in $(seq $n); do slot $c & sleep 60; done; wait
}
card 1 3 & card 2 3 & card 0 2 & wait
log "FINI"
