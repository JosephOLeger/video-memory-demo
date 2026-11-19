#!/bin/bash
# CG-Bench memory module, end to end (October 2026). Usage: pipeline_cgb.sh CGB_DIR CARD
#   1. items: 1 frame / 4 s (64 s clips), window 64 s, query instants at end of evidence + 2 / 5 / 15 / 30 min + end;
#   2. targets: latents and three Qwen references per item (run one shard per card);
#   3. filters; 4. training of the memory at 64 KB, 256 KB and 1 MB of state (basis and width scale together).
set -u
C=${1:?CG-Bench directory (videos/ and cgbench.json)}; CARD=${2:-0}; V=$C/v2
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); H=$(dirname "$0"); PY=${PY:-python}
cd $ROOT; export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $V/items.jsonl ] || $PY $H/build_cgb.py --src $C/cgbench.json --out $V --window-s 64 --gaps 120,300,900,1800
CUDA_VISIBLE_DEVICES=$CARD $PY $H/cgb_targets.py --items $V/items.jsonl --videos $C/videos --out $V/tg --fps 0.25 --shard 0 --nshards 1
$PY $H/cgb_filter.py --tg $V/tg --out $V/filtre
CO=(--targets "$V/tg/*.pt" --kinds cgb --max-past 32 --lam-lat 0 --lam-con 0 --lr 3e-3 --batch 4
    --eval-every 500 --max-eval 300 --drop-blind --teacher-ok-only 1 --seed 0 --ckpt-seg 90 --steps 4000)
mkdir -p $C/runs
for cfg in "128 64 cgb_64kb" "256 128 cgb_256kb" "512 256 cgb_1mb"; do set -- $cfg
  CUDA_VISIBLE_DEVICES=$CARD $PY -m titans.graft.train_stream "${CO[@]}" --basis cache/qwen_l24_stream_pca$1.pt \
     --d-mem $1 --memory-hidden $2 --out $C/runs/$3 > $C/runs/$3.log 2>&1
done
