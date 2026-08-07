#!/bin/bash
# Run a multi-phase study from a single cron entry, one phase at a time.
#
#   */10 * * * * /bin/bash $HOME/work/titans-mac/scripts/chain.sh
#
# Phases differ in what they need: echo cells are ~6.4 GiB and want two per card,
# the margin probe is a single fast pass, the length probe is light but long.
# Encoding that here rather than swapping the crontab between phases means an
# unattended handover cannot be missed, and the guard cannot be left at the previous
# phase's value -- the mistake that cost 74 jobs on 2026-07-29 and 22 on 07-31.
#
# **A phase starts only when the previous one's jobs are actually gone**, not when
# its queue empties. Gating on "queue empty" starts the next phase while the last
# cells still hold cards, which is what caused the in-flight OOMs on 07-29.
cd ~/work/titans-mac || exit 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=~/venvs/titans/bin/python
VAL=cache/egolife_pca128_val.pt
LOG=logs/chain.log
mkdir -p logs frozen_vv

# --- phase 1: the V;V sweep (heavy, two per card) ----------------------------
# Guard 11000: these cells peak at ~6.4 GiB, so a card takes two and refuses a
# third (free falls to ~6.6 GiB). It also keeps workers off a card another user
# is holding.
if [ -s jobs_vv.txt ]; then
  JOBS=jobs_vv.txt GUARD=11000 PREFIX=V SETTLE=120 TMO=21600 WORKERS=8 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[w]orldmodel.echo" >/dev/null; then
  exit 0                      # queue drained but cells still running
fi

# --- phase 2: stability margin on every artifact, and queue the length probe --
# One pass over all of them; seconds each, no training. Writing the marker file
# last makes this idempotent under a cron that fires every ten minutes.
if [ ! -f vv_margins.json ] && ls frozen_vv/*.pt >/dev/null 2>&1; then
  echo "$(date '+%F %T') chain: phase 2, marges" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 $P -m titans.worldmodel.margin \
      --frozen 'frozen_vv/*.pt' --val "$VAL" --out vv_margins.json >> "$LOG" 2>&1

  # One length probe per arm (not per seed): the drift is an architecture property
  # and 46 of them would cost more than the sweep that produced them.
  #
  # The arm's *lowest available* seed, not seed 0: arms whose seed grid starts
  # elsewhere (the eta top-ups start at 3) matched nothing under a `*_s0.pt` glob
  # and were silently skipped. Already-computed probes are skipped too, so a
  # re-run after adding cells costs only the new arms.
  : > jobs_vvlong.txt
  for lab in $(ls frozen_vv/*.pt 2>/dev/null | sed 's|.*/||; s|_s[0-9]*\.pt$||' | sort -u); do
    [ -f "vvlong_${lab}.json" ] && continue
    f=$(ls frozen_vv/${lab}_s*.pt 2>/dev/null | sort -V | head -1)
    [ -n "$f" ] || continue
    echo "$P -m titans.worldmodel.long_run --frozen $f --val $VAL --reps 1000 --pool 4 --diag-every 10 --label $lab --out vvlong_${lab}.json" >> jobs_vvlong.txt
  done
  echo "$(date '+%F %T') chain: $(wc -l < jobs_vvlong.txt) sondes de longueur en file" >> "$LOG"
  exit 0
fi

# --- phase 3: length probe (light, forward-only) -----------------------------
if [ -s jobs_vvlong.txt ]; then
  JOBS=jobs_vvlong.txt GUARD=6000 PREFIX=L SETTLE=30 TMO=21600 WORKERS=4 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[w]orldmodel.long_run" >/dev/null; then
  exit 0
fi

# --- phase 4: convergence, one long trajectory per arm -----------------------
# TMO 36000, not the usual 21600: these are 10000 steps, ~6.2 h at the reference
# cell's rate, so a 6 h timeout would kill and requeue every one of them forever --
# the treadmill that ate an entire sweep on 2026-07-29. WORKERS 3 because there are
# three arms and nothing is gained by more.
if [ -s jobs_cv.txt ]; then
  JOBS=jobs_cv.txt GUARD=11000 PREFIX=C SETTLE=120 TMO=36000 WORKERS=3 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[f]rozen-every" >/dev/null; then
  exit 0                      # convergence trajectories still running
fi
# Score the snapshots. Nothing else does, and without it the trajectories are
# written and never read.
if [ ! -f cv_converge.json ] && ls frozen_cv/*_at*.pt >/dev/null 2>&1; then
  echo "$(date '+%F %T') chain: convergence, lecture des instantanes" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 $P -m titans.worldmodel.converge \
      --frozen 'frozen_cv/*_at*.pt' --val "$VAL" --out cv_converge.json >> "$LOG" 2>&1
  exit 0
fi

# --- phase 5: the denominator ------------------------------------------------
# Every run in the PCA space so far is at seg=8 with the memory on, so neither
# ablating the memory nor widening the attention window has ever been measured
# here. Without both, "the memory helps" has no comparison point and the claim
# that matters -- distant events *without widening the context* -- has never been
# put against the trivial alternative, which is to widen the context.
# GUARD 13000: seg=64 holds a 4x wider attention window than the reference cell.
if [ -s jobs_base.txt ]; then
  JOBS=jobs_base.txt GUARD=13000 PREFIX=B SETTLE=120 TMO=21600 WORKERS=8 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[w]orldmodel.echo" >/dev/null; then
  exit 0
fi

# --- phase 6: the long-stream study, which is what tests the motivating claim --
# Contiguous 256-latent streams rebuilt from EgoLife's own segments, full BPTT over
# the doubled 512 so the echo's copy distance actually receives gradient. At T=256
# a full context's KV weighs 512 KiB against 516 KiB of memory state -- exactly the
# crossover, and 4x past where every earlier measurement sat.
# GUARD 15000 and WORKERS 4: a cell peaks at 12.9 GiB, so one per card. TMO 36000
# because 2500 steps at 9.5 s is 6.6 h, past the usual 6 h timeout.
if [ -s jobs_long.txt ]; then
  JOBS=jobs_long.txt GUARD=15000 PREFIX=G SETTLE=180 TMO=36000 WORKERS=4 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[l]ong256" >/dev/null; then
  exit 0
fi

# --- phase 7: capacity -------------------------------------------------------
# How many items an associative memory holds has never been measured. The memory
# MLP's width is the direct axis; the retention curve read off these artifacts
# afterwards gives capacity x horizon together.
if [ -s jobs_cap.txt ]; then
  JOBS=jobs_cap.txt GUARD=13000 PREFIX=K SETTLE=120 TMO=21600 WORKERS=8 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[w]orldmodel.echo" >/dev/null; then
  exit 0
fi

# --- phase 8: retention curves on everything that has been trained since ------
# The episodic horizon is the metric the project is actually about, so it must be
# read on the long-stream and capacity artifacts too, not only on the V;V ones.
# Evaluation only. Each arm's lowest seed, skipping what is already measured.
for d in frozen_long frozen_base frozen_cap; do
  [ -d "$d" ] || continue
  for lab in $(ls $d/*.pt 2>/dev/null | sed 's|.*/||; s|_s[0-9]*\.pt$||' | sort -u); do
    out="retention_out/${d#frozen_}_${lab}.json"
    [ -f "$out" ] && continue
    f=$(ls $d/${lab}_s*.pt 2>/dev/null | sort -V | head -1)
    [ -n "$f" ] || continue
    v="$VAL"
    [ "$d" = frozen_long ] && v=cache/egolife_long256_val.pt
    echo "$(date '+%F %T') chain: retention $lab" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0 $P -m titans.worldmodel.retention --frozen "$f" \
        --val "$v" --n-eval 48 --out "$out" >> "$LOG" 2>&1
    exit 0                    # one per cron tick, so a card is never monopolised
  done
done
