#!/bin/bash
# LVB des checkpoints de distillation (go de Joseph 14/10). Tour UNIQUEMENT, cartes liberees par ctrl.sh.
L=$HOME/work/iclr/lvb
O=$HOME/work/iclr/distill/orch.log
cd $HOME/work/titans-mac
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=src
C="--data $L/data --basis cache/qwen_l24_pca128.pt --frames 32 --mem-frames 256"
log(){ echo "$(date +%F_%T) lvb : $*" >> $O; }
# copie complete ?
while pgrep -f "rsync -a --partial --files-from=val_list.txt" > /dev/null; do sleep 60; done
n=$(python3 -c "
import json,os;d=json.load(open('$L/data/lvb_val.json'));print(sum(os.path.exists('$L/data/videos/'+x['video_path']) for x in d))")
log "copie finie, $n/1337 questions avec video"
run(){ # carte, attente (motif de fin dans orch.log ou vide), nom, args
  local card=$1 wait=$2 name=$3; shift 3
  if [ -n "$wait" ]; then until grep -q "$wait" $O; do sleep 120; done; fi
  log "$name carte $card"
  CUDA_VISIBLE_DEVICES=$card $PY $L/eval_lvb_clips.py --clip 16 $C "$@" --out $L/$name.json > $L/$name.log 2>&1
  log "$name fini (code $?)"
}
run 2 "" mem_s0_4000 --ckpt $L/ckpt/dist_unit_s0_at4000.pt &
run 1 "ctrl P1_l10_s2 fini" mis_s0_4000 --ckpt $L/ckpt/dist_unit_s0_at4000.pt --memory-source mismatched &
run 0 "ctrl H_d32_l10_s2 fini" full256 --frames 256 &
wait
log "tout fini"
