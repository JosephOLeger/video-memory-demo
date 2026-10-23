#!/bin/bash
# 19/10 : graines 1 et 2 de la greffe en flux couleur puis formes (C1 -> C2), et formes sans
# pre-entrainement couleur (graine 0). Chaque carte attend la fin de la file seeds8 ET une carte libre.
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
card_busy() { local p; for p in $(pgrep -u "$USER" -f python); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
wait_card() { until grep -q "seeds8 : file videe" $D/orch.log; do sleep 300; done; while card_busy $1; do sleep 120; done; }
run() { # carte nom args...
  local c=$1 n=$2; shift 2
  log "couleur : $n carte $c"
  CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream "$@" "${CO[@]}" --out $S/$n > $S/$n.log 2>&1
  log "couleur : $n fini (code $?)"
}
w() { local c=$1 s=$2; wait_card $c
  run $c C1_color_s$s --targets "$S/tgc/*.pt" --kinds color --seed $s
  run $c C2_shape_s$s --targets "$S/tg/*.pt" --kinds shape --seed $s --init $S/C1_color_s${s}_at4000.pt; }
w 0 1 & w 1 2 &
( wait_card 2; run 2 C2scratch_shape_s0 --targets "$S/tg/*.pt" --kinds shape --seed 0 ) &
wait
log "couleur : graines 1-2 finies"
