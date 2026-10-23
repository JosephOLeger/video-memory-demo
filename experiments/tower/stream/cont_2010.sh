#!/bin/bash
# 20/10 matin : la couleur ne se reproduit pas (s1 26 %, s2 36 % contre s0 100 %), memes reglages.
# Remplace night_1910.sh (arrete) et les runs formes C2_shape_s{1,2} partis d'une couleur non apprise.
#   carte 1 : C1c_color_s1 = reprise de C1_color_s1_at4000, 4000 pas de plus (nouveau warmup + cosinus, lr 3e-3)
#   carte 2 : C1c_color_s2 = idem depuis C1_color_s2_at4000 (la graine 2 montait : 27 -> 36 %)
#   carte 0 : le worker seeds8 finit la file (H_s7, PN_s7) ; apres « seeds8 : file videe », C1_color_s3 (nouvelle graine)
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
card_busy() { local p; for p in $(pgrep -u "$USER" -f python); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
run() { # carte nom args...
  local c=$1 n=$2; shift 2
  log "couleur : $n carte $c"
  CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream "$@" "${CO[@]}" --out $S/$n > $S/$n.log 2>&1
  log "couleur : $n fini (code $?)"
}
for c in 1 2; do while card_busy $c; do sleep 20; done; done
run 1 C1c_color_s1 --targets "$S/tgc/*.pt" --kinds color --seed 1 --init $S/C1_color_s1_at4000.pt &
run 2 C1c_color_s2 --targets "$S/tgc/*.pt" --kinds color --seed 2 --init $S/C1_color_s2_at4000.pt &
( until grep -q "seeds8 : file videe" $D/orch.log; do sleep 300; done
  while card_busy 0; do sleep 120; done
  run 0 C1_color_s3 --targets "$S/tgc/*.pt" --kinds color --seed 3 ) &
wait
log "couleur 20/10 : reprises s1-s2 et graine 3 finies"
