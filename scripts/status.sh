#!/bin/bash
# One clean status line for an unattended sweep monitor.
#
#   usage: status.sh <results-glob> <jobs-file> [worker-regex]
#   prints: <done> <queued> <running> <workers> <drops> <gpu_busy>
#
#   scripts/status.sh 'echo_ref64_*.json' jobs_ref64.txt 'pool4\.sh jobs_ref64'
#
# This lives in a script rather than inline in a monitor's ssh command, because
# inlining is how the counting broke: the previous version used
# `$(pgrep -fc ... || echo 0)`, and since `pgrep -c` prints 0 *and* exits 1 when it
# matches nothing, the `||` appended a second 0, the arithmetic then failed, and the
# worker count silently became 0 -- a false "all workers dead" alarm while all eight
# were running. An unattended monitor that cries wolf is worse than none, so the
# counting is somewhere it can be read and tested.
cd ~/work/titans-mac || exit 1

GLOB=${1:-'echo_*.json'}
JOBS=${2:-jobs.txt}
WRE=${3:-'pool4\.sh'}

done_=$(ls $GLOB 2>/dev/null | wc -l)
queued=0; [ -f "$JOBS" ] && queued=$(wc -l < "$JOBS")

# each job is two processes (the `timeout` wrapper and the python it runs)
running=$(ps -eo args | grep -c '[t]itans.worldmodel.echo'); running=$(( running / 2 ))
workers=$(pgrep -fc "$WRE"); workers=${workers:-0}
drops=$(grep -h 'DROP after' logs/pool/w*.log 2>/dev/null | wc -l); drops=${drops:-0}
# a card counts as busy above 10% utilisation
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 > 10 {n++} END {print n+0}')

echo "$done_ $queued $running $workers $drops $gpu"
