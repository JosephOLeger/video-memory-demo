#!/bin/bash
# 21/10 : telechargement des videos completes de CG-Bench (69 archives, 326 Go), reprise automatique,
# puis extraction archive par archive dans videos/ (les zips sont gardes).
C=$HOME/work/iclr/cgb; O=$HOME/work/iclr/distill/orch.log; HF=$HOME/miniforge3/envs/titans/bin/hf
log() { echo "$(date +%F_%T) cgb : $*" >> $O; }
unset HF_HUB_OFFLINE; export HF_HUB_ENABLE_HF_TRANSFER=0
log "telechargement lance"
for i in 1 2 3 4 5; do
  $HF download CG-Bench/CG-Bench --repo-type dataset --include "video_chunk_*.zip" --local-dir $C/raw && break
  log "telechargement interrompu (essai $i), reprise dans 5 min"; sleep 300
done
n=$(ls $C/raw/video_chunk_*.zip 2>/dev/null | wc -l); log "telechargement : $n archives"
mkdir -p $C/videos
for z in $C/raw/video_chunk_*.zip; do
  [ -f "$z" ] || continue; [ -f "$z.ok" ] && continue
  unzip -n -q "$z" -d $C/videos && touch "$z.ok" || log "extraction ECHEC $z"
done
log "extraction finie : $(find $C/videos -name "*.mp4" | wc -l) videos"
