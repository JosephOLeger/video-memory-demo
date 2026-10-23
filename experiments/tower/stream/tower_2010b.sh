#!/bin/bash
# 20/10 midi (Josep : « fais ça et lance le ») : couleur avec variantes (reponse non memorisable).
# Remplace cont_2010.sh (reprises C1c arretees : elles apprennent par coeur, cf. diag_color) et
# bench_dense_tower.sh (meme file qtour.txt, reprise ici).
# 1. cartes 1 et 2 : variantes de couleur (2 eclats, ~45 min) -> stream/tgv
# 2. file couleur : C1v_color_s0..7 (variantes, memes reglages que C1), 2 par carte
# 3. file banc dense (graines 3-7) : 1 place par carte, et les places couleur quand la file couleur est vide
# Carte 0 : rejoint apres « seeds8 : file videe » (fin de la file 8 graines + mesure des portes).
S=$HOME/work/iclr/stream; B=$HOME/work/iclr/bench_dense; O=$HOME/work/iclr/distill/orch.log
QC=$S/qcol.txt; QB=$B/qtour.txt
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $O; }
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
[ -f $QC ] || for s in 0 1 2 3 4 5 6 7; do echo "$s"; done > $QC
pop() { flock $1.lock sh -c 'l=$(head -1 '$1'); [ -n "$l" ] && sed -i 1d '$1'; echo "$l"'; }
color() { # carte graine
  log "couleur v : C1v_color_s$2 carte $1"
  ( cd $HOME/work/titans-mac && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1 $PY -m titans.graft.train_stream --targets "$S/tgc/*.pt" --variants "$S/tgv/*.pt" --kinds color "${CO[@]}" --seed $2 --out $S/C1v_color_s$2 > $S/C1v_color_s$2.log 2>&1 )
  log "couleur v : C1v_color_s$2 fini (code $?)"
}
bench() { # carte "arm seed"
  set -- $1 $2; local c=$1 arm=$2 s=$3 M="--muon unit"; [ $arm = usq10k ] && M="--muon unit_squash --muon-scale 350"
  log "banc dense : mu_${arm}d_s$s carte $c"
  ( cd $B/tm && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$c $PY -m titans.worldmodel.echo --label mu_${arm}d_s$s --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt --horizon 24 --steps 10000 --seed $s --eta-init 0.995 --theta-scale 0.03 --alpha-init 0.002 $M --frozen-every 1000 --out $B/mu_${arm}d_s$s.json --frozen-out $B/mu_${arm}d_s$s.pt > $B/mu_${arm}d_s$s.log 2>&1 )
  log "banc dense : mu_${arm}d_s$s fini (code $?)"
}
cslot() { local c=$1 l; while true; do l=$(pop $QC); if [ -n "$l" ]; then color $c $l; continue; fi
  l=$(pop $QB); [ -z "$l" ] && return; bench "$c $l"; done; }
bslot() { local c=$1 l; while true; do l=$(pop $QB); [ -z "$l" ] && return; bench "$c $l"; done; }
pre() { # carte eclat
  ( cd $HOME/work/titans-mac && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1 $PY -m titans.graft.stream_color_variants --tgc $S/tgc --data data/finevideo --out $S/tgv --shard $2 --nshards 2 > $S/tgv_$2.log 2>&1 )
  log "couleur v : variantes eclat $2 (code $?)"
}
log "couleur v : variantes (cartes 1-2)"
bslot 1 & sleep 30; bslot 2 & sleep 30
pre 1 0 & P1=$!; pre 2 1 & P2=$!; wait $P1 $P2
log "couleur v : $(ls $S/tgv | wc -l) videos de variantes"
cslot 1 & sleep 60; cslot 2 & sleep 60; cslot 1 & sleep 60; cslot 2 &
( until grep -q "seeds8 : file videe" $O; do sleep 300; done
  cslot 0 & sleep 60; cslot 0 & sleep 60; bslot 0 & wait ) &
wait
log "couleur v + banc dense tour : FINI"
