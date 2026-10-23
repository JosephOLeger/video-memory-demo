#!/bin/bash
# 19/10 soir (Josep : « lance ce qu'il faut pour bien profiter des GPU », tour deconnectee jusqu'au 20/10).
# Remplace color_seeds.sh. Les graines couleur passent AVANT la fin de seeds8 pour la reunion :
#   carte 1 : attend H_d32_l10_s5 (PID 1818240), puis C1_color_s1 -> C2_shape_s1, puis reprend la file seeds8
#   carte 2 : attend P1_l10_s5 (PID 1800453),   puis C1_color_s2 -> C2_shape_s2, puis reprend la file seeds8
#   carte 0 : le worker seeds8 d'origine continue la file ; apres « seeds8 : file videe », C2scratch_shape_s0
# Les workers seeds8 des cartes 1 et 2 (PID 205518, 205519) ont ete arretes a la main (leurs python continuent).
D=$HOME/work/iclr/distill; S=$HOME/work/iclr/stream; U=$HOME/work/iclr/u2; L=$HOME/work/iclr/load
Q=$HOME/work/iclr/seeds8q.txt
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) $*" >> $D/orch.log; }
CO=(--basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300)
UC="--basis cache/qwen_l24_pca128.pt --bank $L/bank.pt --muon unit --steps 2500 --train-k 1,2,4 --fill-choices 4,8,16 --eval-every 500 --eval-n 96 --eval-k 1,2,4,8 --reach-n 48 --reach-after 0,2,4,8,16,24 --reach-fill 24 --save-every 500"
card_busy() { local p; for p in $(pgrep -u "$USER" -f python); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
wait_pid() { # pid nom
  while kill -0 $1 2>/dev/null; do sleep 60; done
  if grep -q TRAIN-LOAD-FINIE $U/$2.log; then log "seeds8 : $2 fini (code 0, worker remplace)"; else log "seeds8 : $2 fini SANS TRAIN-LOAD-FINIE"; fi
}
run() { # carte nom args...
  local c=$1 n=$2; shift 2
  log "couleur : $n carte $c"
  CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream "$@" "${CO[@]}" --out $S/$n > $S/$n.log 2>&1
  log "couleur : $n fini (code $?)"
}
drain() { # carte : reprend la file seeds8 (meme commande que seeds8_tower.sh)
  local c=$1 line
  while true; do
    line=$(flock $Q.lock sh -c 'l=$(head -1 '$Q'); [ -n "$l" ] && sed -i 1d '$Q'; echo "$l"')
    [ -z "$line" ] && break
    local rest=${line#*|}; local name=${rest%%|*} args=${rest#*|}
    log "seeds8 : $name carte $c (file reprise)"
    CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_load $UC $args --out $U/$name > $U/$name.log 2>&1
    log "seeds8 : $name fini (code $?)"
  done
}
lane() { # carte pid nom_en_cours graine
  local c=$1 pid=$2 cur=$3 s=$4
  wait_pid $pid $cur
  while card_busy $c; do sleep 60; done
  run $c C1_color_s$s --targets "$S/tgc/*.pt" --kinds color --seed $s
  run $c C2_shape_s$s --targets "$S/tg/*.pt" --kinds shape --seed $s --init $S/C1_color_s${s}_at4000.pt
  drain $c
}
lane 1 1818240 H_d32_l10_s5 1 &
lane 2 1800453 P1_l10_s5 2 &
( until grep -q "seeds8 : file videe" $D/orch.log; do sleep 300; done
  while card_busy 0; do sleep 120; done
  run 0 C2scratch_shape_s0 --targets "$S/tg/*.pt" --kinds shape --seed 0 ) &
wait
log "nuit 19/10 : couleur graines 1-2 et formes sans pre-entrainement finies"
