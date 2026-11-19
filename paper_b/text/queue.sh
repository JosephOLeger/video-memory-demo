# Shared header of the text-track queues. Source it, then call the functions below.
#   PY    python with torch and transformers        (default: python)
#   CARD  GPU index                                 (default: 0)
#   LOG   where START / END lines go                (default: runs/jobs.log)
#   ROOT  this directory, holding bl_data/, the PCA bases, evalsets/ and runs/
# Every job has a hard timeout, a failure never stops the queue, nothing starts below 8 GiB of free disk.
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PY=${PY:-python}; CARD=${CARD:-0}; LOG=${LOG:-$ROOT/runs/jobs.log}
cd "$ROOT"; mkdir -p runs
export PYTHONPATH=$ROOT/../../src:$ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEXT_GEN=gen_bench30
# the training and evaluation recipe of every model in the paper (arm B = learned write, PCA-128 basis at layer 24,
# sliding write of 256-token chunks, window 256, question hidden at write time)
A="--arm B --past 4096 --past-choices 2048,3400,7100 --long-p 7000 --dmax 1500 --eval-dists 300 --past-chunk 256 --past-mode slide --basis $ROOT/basis_l24_blf_pca{k}.pt --lr 3e-3 --long-q 1 --write-attnpool 8 --layout official --b30-content"
E=evalsets; D1=$E/b30_dev_4k_1ep_sp.jsonl; D2=$E/b30_dev_4k.jsonl
EXT=$E/lme_oracle.jsonl,$E/locomo.jsonl,$E/bl1120_4k_reconstructed.jsonl
BL10="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10:4k"
H=$E/host; HF=$H/mmlu.jsonl,$H/arc.jsonl,$H/triviaqa.jsonl,$H/pg19test.jsonl,$H/mmbench.jsonl,$H/pope.jsonl,$H/chartqa.jsonl,$H/textvqa.jsonl,$H/gsm8k.jsonl
SD=$E/std; STD=$(ls $SD/*.jsonl 2>/dev/null | tr "\n" "," | sed "s/,$//")

# run TIMEOUT NAME "ARGS": one train_b30.py job, log in runs/NAME.log
run() { av=$(df --output=avail -BG . | tail -1); av=${av//[!0-9]/}
        if [ "${av:-0}" -lt 8 ]; then echo "DISK-LOW $(date +%F_%T) skip $2 (${av}G)" >> $LOG; return; fi
        echo "START $(date +%F_%T) card$CARD $2" >> $LOG
        CUDA_VISIBLE_DEVICES=$CARD nice -n 10 timeout $1 $PY train_b30.py $3 > runs/$2.log 2>&1; rc=$?
        echo "END $(date +%F_%T) $2 rc=$rc" >> $LOG; }
# trn NAME INIT SEED STEPS [flags]: continue a learned-write model (the recipe of m_dp_long and its seeds)
trn() { [ -f runs/$1.pt ] || B30_NEP=1:0.4,2:0.3,3:0.2,4:0.1 run $((10 * $4 + 3600)) $1 "$A --preserve 1.0 --init $2 --seed $3 --steps $4 --warmup 10 --eval-every 0 $5 --out runs/$1"; }
# ev NAME: the thirty types, dev sets of 600 and 900 records
ev()  { [ -f runs/$1.pt ] || return
        run 7200  fin1_$1 "$A --init runs/$1.pt --jsonl-eval $D1 --jsonl-conds sup,orc,mem,omem --out runs/fin1_$1"
        run 10800 fin2_$1 "$A --init runs/$1.pt --jsonl-eval $D2 --jsonl-conds mem,omem --out runs/fin2_$1"; }
# evx NAME: official BABILong 4k qa1 to qa10, 8k to 32k qa1 to qa4, LongMemEval, LoCoMo, reconstructed qa11 to qa20
evx() { [ -f runs/$1.pt ] || return
        run 10800 finbl_$1  "$A --init runs/$1.pt --official $BL10 --official-conds mem,omem,orc --out runs/finbl_$1"
        run 21600 bllong_$1 "$A --init runs/$1.pt --official qa1,qa2,qa3,qa4:8k,16k,32k --official-conds mem,omem --out runs/bllong_$1"
        run 21600 ext_$1    "$A --init runs/$1.pt --jsonl-eval $EXT --jsonl-conds mem,omem --jsonl-gen 24 --out runs/ext_$1"; }
# std NAME: RULER 4k, LongBench, InfiniteBench En.Dia (score with score_std.py)
std() { [ -f runs/$1.pt ] || return
        run 43200 std_$1 "$A --init runs/$1.pt --jsonl-eval $STD --jsonl-conds mem,omem --jsonl-gen 32 --out runs/std_$1"; }
# ref NAME SET: model-free references (window only, full context) and two untrained memories (notes, BM25)
ref() { run 43200 ref_$1 "$A --init runs/m_dp_long.pt --jsonl-eval $2 --jsonl-conds win,full,summ,bm25 --out runs/ref_$1"; }
# host NAME: the cost of the graft on the host outside its task
host() { [ -f runs/$1.pt ] || return
        run 43200 host_$1 "$A --init runs/$1.pt --host-eval $HF --host-conds init,carry,mem,omem --out runs/host_$1"; }
waitpid() { while kill -0 $1 2>/dev/null; do sleep 300; done; }
