#!/bin/bash
cd ~/work/iclr/ptest
PY="$HOME/miniforge3/envs/titans/bin/python -I"
while kill -0 2356928 2>/dev/null; do sleep 60; done
rm -f check_out.jsonl
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_ptest_cached.py --questions smoke_q.jsonl --out check_out.jsonl --videos data/videos --check > check.log 2>&1
bad=$(grep CHECK check.log | awk "{split(\$7,a,\"/\"); if (a[1]!=a[2] || \$9>0.5) print}" | wc -l)
good=$(grep -c CHECK check.log)
echo "check: $good comparisons, $bad mismatches"
if [ "$good" -ge 10 ] && [ "$bad" -eq 0 ]; then S=eval_ptest_cached.py; else S=eval_ptest.py; fi
echo "using $S"
CUDA_VISIBLE_DEVICES=0 $PY scripts/$S --questions questions_rest.jsonl --out res_rest.jsonl
