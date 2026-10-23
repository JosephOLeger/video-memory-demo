#!/bin/bash
# 15/10 : entrainements FLUX apres le precalcul. R1 complet carte 0 ; R2 CE seule carte 1.
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
until grep -q "stream : cibles finies" $D/orch.log; do sleep 120; done
C=(--targets "$S/tg/*.pt" --basis cache/qwen_l24_pca128.pt --steps 2000 --batch 4 --eval-every 250 --max-eval 300)
go() { # nom carte args...
  local n=$1 c=$2; shift 2
  log "stream : $n carte $c"
  CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream "${C[@]}" "$@" --out $S/$n > $S/$n.log 2>&1
  log "stream : $n fini (code $?)"
}
go R1_full_s0 0 &
go R2_ce_s0 1 --lam-lat 0 --lam-con 0 &
wait
