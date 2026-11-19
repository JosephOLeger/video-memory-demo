#!/bin/bash
# CG-Bench videos (69 archives, 326 GB), resumable download, then extraction archive by archive into videos/.
# Usage: download.sh DEST
C=${1:?destination directory}; mkdir -p $C/raw $C/videos
unset HF_HUB_OFFLINE; export HF_HUB_ENABLE_HF_TRANSFER=0
for i in 1 2 3 4 5; do
  hf download CG-Bench/CG-Bench --repo-type dataset --include "video_chunk_*.zip" --local-dir $C/raw && break
  echo "download interrupted (try $i), resuming in 5 min"; sleep 300
done
for z in $C/raw/video_chunk_*.zip; do
  [ -f "$z.ok" ] && continue
  unzip -n -q "$z" -d $C/videos && touch "$z.ok" || echo "extraction failed: $z"
done
echo "$(find $C/videos -name '*.mp4' | wc -l) videos"
