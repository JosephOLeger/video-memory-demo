#!/usr/bin/env bash
# AHN on the thirty types and BABILong 4k, three variants, one card. Usage: CARD=1 bash run_full_ahn.sh [GDN DN Mamba2]
# Refuses to start below 12 GiB free; each process caps itself at MEMFRAC of the card.
# LoCoMo and LongMemEval (up to 24k tokens): a card with nothing else on it and MEMFRAC=0.9.
B=$(cd "$(dirname "$0")" && pwd); source $B/env_ahn.sh; cd $B
CARD=${CARD:-1}; MEMFRAC=${MEMFRAC:-0.45}; OUT=${OUT:-runs/full_w256}
VARS=${@:-GDN DN Mamba2}
EV=$B/../text/evalsets
BLT=qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10
mkdir -p $OUT logs
go () {  # go <variant> <conds> <data> [extra]
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $CARD)
  if [ "$free" -lt 12288 ]; then echo "$(date) card $CARD free ${free} MiB < 12 GiB: skip $1 $3"; return; fi
  echo "$(date) start $1 $2 $3"
  CUDA_VISIBLE_DEVICES=$CARD python ahn_eval.py --variant $1 --conds $2 --data $3 --out $OUT --mem-frac $MEMFRAC \
     --state-report ${@:4} 2>&1 | grep -E "SUMMARY|STATE|OOM|Error|error"
}
first=1
for V in $VARS; do
  # win and full do not depend on the AHN variant: computed once, with the first variant
  if [ $first = 1 ]; then C=win,full,mem,swa,omem; first=0; else C=mem,swa,omem; fi
  go $V $C $EV/b30_dev_4k_1ep_sp.jsonl
  go $V $C $EV/b30_dev_4k.jsonl
  go $V $C babilong:4k:$BLT
done
# standard benchmarks (RULER 4k, LongBench), as run in November:
#   for F in ruler_4k longbench_hotpotqa longbench_2wikimqa longbench_musique longbench_qasper longbench_multifieldqa_en longbench_passage_retrieval_en longbench_trec; do
#     CUDA_VISIBLE_DEVICES=$CARD python ahn_eval.py --variant GDN --conds win,full,mem,swa,omem --gen 32 --data $EV/std/$F.jsonl --out runs/std_w256 --mem-frac 0.9; done
