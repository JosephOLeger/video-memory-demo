#!/bin/bash
# Find the critical accumulation horizon tau*, unattended, from one cron entry.
#
#   */10 * * * * /bin/bash $HOME/work/titans-mac/scripts/tau_chain.sh
#
# tau* is bracketed by the existing arms between tau_acc = 20 (eta 0.95, held 64 000
# tokens) and 50 (eta 0.98, NaN at 320). This trains the intermediate points and then
# probes **every seed**, which matters because the divergence is documented as
# seed-dependent -- and because every survival number in the project so far came from
# a single seed per arm (the earlier chain ran one length probe per *arm*, not per
# seed), so the whole stability column is currently n=1.
#
# Two phases, gated on jobs being *gone* rather than on the queue being empty -- the
# in-flight cells of a finished queue still hold cards, and starting the next phase
# then is what caused the OOMs of 2026-07-29.
cd ~/work/titans-mac || exit 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=~/venvs/titans/bin/python
VAL=cache/egolife_pca128_val.pt
LOG=logs/tau_chain.log
mkdir -p logs frozen_tau frozen_cas frozen_traj taucrit

# --- theta/alpha audit -------------------------------------------------------
# The gates the outer loop *chose* are the measurement, not a by-product: the cascade
# arms land ~19x higher on theta*lambda_hidden and ~3x lower on alpha than the scalar
# ones, and 2x higher on ||h0|| (d=2.11, the cleanest separator), which is what puts
# them past the stability bound. A few seconds per artefact, refreshed whenever a
# new one lands, so the audit is never stale when the report is pulled.
if [ -n "$(find frozen_vv frozen_cas frozen_tau -name '*.pt' -newer theta_audit.json \
           -print -quit 2>/dev/null)" ] || [ ! -f theta_audit.json ]; then
  CUDA_VISIBLE_DEVICES=${AUDIT_GPU:-3} $P -m titans.worldmodel.margin \
    --frozen 'frozen_vv/*.pt' --frozen 'frozen_cas/*.pt' --frozen 'frozen_tau/*.pt' \
    --val "$VAL" --nseq 8 --out theta_audit.json >> "$LOG" 2>&1 || true
fi

# --- phase 1: train the new eta points ---------------------------------------
# GUARD 8000: a T=64 batch-8 cell peaks at 6.6 GiB, so a card takes one of these
# alongside whatever the other queues are running and refuses a second.
if [ -s jobs_tau.txt ]; then
  # WORKERS 6: a cell peaks at 6.6 GiB, so the guard lets a card take two and
  # refuses a third (free falls to 9.8 GiB, above the guard, but the third would
  # leave 3.2 -- pool4's per-card lock is what makes that sequence safe to rely on).
  JOBS=jobs_tau.txt GUARD=8000 PREFIX=E SETTLE=180 TMO=21600 WORKERS=6 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[e]cho --train.*--label E_e" >/dev/null; then
  exit 0
fi

# --- phase 2: probe survival, per seed, for every eta arm --------------------
# 200 repetitions = 12 800 tokens. Every arm that has ever diverged did so before
# 400 tokens, so this is 30x past the event and a fifth of the cost of the 1000-rep
# probe; arms that clear it are recorded as "held 12 800", not as "stable forever".
# loop and cycle only: cycle is the harsher stream (eta 0.90 survives loop for 64 000
# tokens and dies at 5 760 in cycle), and shuffle behaves like cycle.
: > jobs_tauprobe.txt
for d in frozen_tau frozen_vv; do
  for f in $d/nocas_e*_s*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f" .pt)
    [ -f "taucrit/${b}.json" ] && continue
    echo "$P -m titans.worldmodel.long_run --frozen $f --val $VAL --reps 200 --pool 4 --diag-every 2 --modes loop,cycle --label ${b%_s*} --out taucrit/${b}.json" >> jobs_tauprobe.txt
  done
done
if [ -s jobs_tauprobe.txt ]; then
  echo "$(date '+%F %T') tau_chain: $(wc -l < jobs_tauprobe.txt) sondes en file" >> "$LOG"
  JOBS=jobs_tauprobe.txt GUARD=4000 PREFIX=Q SETTLE=45 TMO=7200 WORKERS=4 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[l]ong_run.*taucrit" >/dev/null; then
  exit 0
fi

# --- phase 3: does the cascade compensate a dilution? ------------------------
# The cascade arms sit ~19x higher on theta*lambda_hidden and 2x higher on ||h0|| than
# the scalar ones, and the compensation is monotone in the leak rate -- which is exactly why sweeping
# the leak was flat in survival. The hypothesis is that the chain's diffusion drains
# the visible reservoir, and the outer loop pays for it in theta. These arms change the
# structure and leave theta and alpha freely learned, so the gates themselves are the
# measurement. Predictions written before launching: results/predictions_cascade_dilution.md
if [ -s jobs_cas.txt ]; then
  JOBS=jobs_cas.txt GUARD=8000 PREFIX=C SETTLE=180 TMO=21600 WORKERS=4 \
    bash "$HERE/keepalive.sh"
  exit 0
fi
if pgrep -f "[e]cho .*--label cas" >/dev/null; then
  exit 0
fi

# --- phase 4: trajectories on the eta values that bracket tau* ---------------
# The probes above locate tau*(2500 steps). This asks whether tau* itself moves as the
# outer loop trains -- 2500 steps is only ~30% of the asymptotic mem_effect, so the
# threshold we just measured is a snapshot of an unconverged model. 10 000 steps with a
# snapshot every 1250 matches frozen_cv/ and frozen_nh/, so the new arms drop straight
# into the existing trajectory comparisons.
#
# The builder refuses to emit anything if the probes did not actually bracket the
# transition, so a failed phase 2 stops here instead of burning 50 GPU-hours.
mkdir -p frozen_traj
if [ ! -f jobs_traj.txt ]; then
  PYTHONPATH=src $P scripts/build_traj_queue.py --root . --out jobs_traj.txt \
    >> "$LOG" 2>&1
  [ -f jobs_traj.txt ] || : > jobs_traj.txt   # unbracketed: mark it done, don't retry
fi

# Both remaining queues share the cards: jobs_later is 5 cells that finish the length
# sweep, jobs_traj is the 9-cell trajectory study. 6 workers at ~7 GiB is the mix that
# ran phase 1 without a single requeue.
if [ -s jobs_traj.txt ]; then
  JOBS=jobs_traj.txt GUARD=8000 PREFIX=T SETTLE=180 TMO=36000 WORKERS=4 \
    bash "$HERE/keepalive.sh"
fi
# --- phase 5: the length sweep's latent-matched controls ---------------------
# Held back so tau* got the cards first. They are what separates "a longer training
# sequence helps" from "more latents seen helps": same number of latents as T=256 at
# 2500 steps, reached with 4x the steps at T=64 and 2x at T=128.
if [ -s jobs_later.txt ]; then
  JOBS=jobs_later.txt GUARD=8000 PREFIX=Z SETTLE=180 TMO=36000 WORKERS=2 \
    bash "$HERE/keepalive.sh"
fi
exit 0
