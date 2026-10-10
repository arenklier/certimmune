#!/bin/bash
# Stop queue worker(s) and the job they are running, cleanly.
#
#   stop_worker.sh <phase> <worker_id>
#
# Two traps this script exists to avoid, both hit for real:
#   1. The launch wrapper's command line contains "qworker.py <phase> <id>"
#      too, so a naive grep matches the wrapper and `head -1` can pick it.
#      Killing the wrapper leaves the real worker running -- which is how two
#      worker Bs ended up live at once and the card dropped to 370 MiB free.
#      We therefore match only lines whose command ENDS with the worker id
#      (the wrapper has redirections after it) and act on every match.
#   2. shell=True puts a shell between the worker and the real process, so
#      killing the worker alone orphans the GPU job. The patched qworker
#      handles SIGTERM by killing its job's whole process group; this script
#      falls back to the recorded pgid if the worker does not exit.

set -u
PHASE=${1:?phase required}
WID=${2:?worker id required}
Q=/mnt/data/certimmune/queue

mapfile -t pids < <(ps -eo pid,cmd | grep "[q]worker\.py ${PHASE} ${WID}\$" | awk '{print $1}')
if [ ${#pids[@]} -eq 0 ]; then
  echo "no worker ${PHASE}/${WID} running"
  exit 0
fi

echo "stopping worker ${PHASE}/${WID}: pids ${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null

for _ in $(seq 1 20); do
  sleep 1
  alive=0
  for p in "${pids[@]}"; do ps -o pid= -p "$p" >/dev/null 2>&1 && alive=1; done
  [ "$alive" -eq 0 ] && break
done

for p in "${pids[@]}"; do
  if ps -o pid= -p "$p" >/dev/null 2>&1; then
    echo "worker $p did not exit; killing its job groups and the worker"
    for c in "$Q/claimed/$PHASE"/*/pgid; do
      [ -e "$c" ] || continue
      kill -9 -- "-$(cat "$c")" 2>/dev/null
    done
    kill -9 "$p" 2>/dev/null
  fi
done

echo "remaining claims: $(ls "$Q/claimed/$PHASE" 2>/dev/null | tr '\n' ' ')"
echo "stray certify/metacloak processes: $(ps -eo cmd | grep -cE '[c]ertify\.py|[m]etacloak\.py')"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
