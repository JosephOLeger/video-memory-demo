#!/bin/bash
# 20/10 15h50 : 6 graines couleur v (s1 s2 s4-s7) mortes au lancement (OOM : 2 couleur + banc sur la meme carte).
# Relance : 1 couloir couleur par carte (1 et 2), chacun remplace la graine couleur en cours ;
# 4 couloirs banc, chacun remplace un run banc en cours. Concurrence par carte inchangee (1 couleur + 2 banc), qui tient.
S=$HOME/work/iclr/stream; B=$HOME/work/iclr/bench_dense; O=$HOME/work/iclr/distill/orch.log
QC=$S/qcol2.txt; QB=$B/qtour_hold.txt
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $O; }
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
pop() { flock $1.lock sh -c "l=\$(head -1 $1); [ -n \"\$l\" ] && sed -i 1d $1; echo \"\$l\""; }
waitpid() { while kill -0 $1 2>/dev/null; do sleep 60; done; }
color() { log "couleur v relance : C1v_color_s$2 carte $1"
  ( cd $HOME/work/titans-mac && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1 $PY -m titans.graft.train_stream --targets "$S/tgc/*.pt" --variants "$S/tgv/*.pt" --kinds color "${CO[@]}" --seed $2 --out $S/C1v_color_s$2 > $S/C1v_color_s$2.log 2>&1 )
  log "couleur v relance : C1v_color_s$2 fini (code $?)"; }
bench() { set -- $1 $2 $3; local c=$1 arm=$2 s=$3 M="--muon unit"; [ $arm = usq10k ] && M="--muon unit_squash --muon-scale 350"
  log "banc dense relance : mu_${arm}d_s$s carte $c"
  ( cd $B/tm && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$c $PY -m titans.worldmodel.echo --label mu_${arm}d_s$s --train cache/egolife_pca128_train.pt --val cache/egolife_pca128_val.pt --horizon 24 --steps 10000 --seed $s --eta-init 0.995 --theta-scale 0.03 --alpha-init 0.002 $M --frozen-every 1000 --out $B/mu_${arm}d_s$s.json --frozen-out $B/mu_${arm}d_s$s.pt > $B/mu_${arm}d_s$s.log 2>&1 )
  log "banc dense relance : mu_${arm}d_s$s fini (code $?)"; }
clane() { waitpid $2; sleep 30; local l; while l=$(pop $QC); [ -n "$l" ]; do color $1 $l; done; }
blane() { waitpid $2; sleep 30; local l; while l=$(pop $QB); [ -n "$l" ]; do bench "$1 $l"; done; }
log "relance_2010 : couleur s1 s2 s4-s7 + banc s5-s7"
clane 2 3201058 & clane 1 3199514 &
blane 1 2891458 & blane 1 3214654 & blane 2 2891598 & blane 2 3223808 &
wait; log "relance_2010 : FINI"
