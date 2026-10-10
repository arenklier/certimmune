#!/bin/bash
# Wait until the GPU is effectively idle (the user's own jobs have finished),
# then run one phase of the CertImmune job queue.
#
#   start_when_free.sh <phase> <workers> [free_mib_threshold] [mib_per_worker]
#
# Workers are launched one at a time and only while there is room for another,
# so a phase whose per-job footprint turns out larger than expected degrades to
# fewer workers instead of pushing the card into OOM.

set -u
ROOT=/mnt/data/certimmune
PHASE=${1:?phase required}
WORKERS=${2:-1}
THRESH=${3:-5000}
PER_WORKER=${4:-8000}
RESERVE=4000

used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

echo "[$(date -Is)] waiting for GPU to drop below ${THRESH} MiB (now $(used) MiB)"
until [ "$(used)" -lt "$THRESH" ]; do sleep 120; done
echo "[$(date -Is)] GPU free ($(used) MiB used) -- starting phase ${PHASE}"

pids=()
for i in $(seq 1 "$WORKERS"); do
  if [ "$i" -gt 1 ]; then
    # only add a worker if there is still room for it plus a reserve
    avail=$(free_mib)
    if [ "$avail" -lt $((PER_WORKER + RESERVE)) ]; then
      echo "[$(date -Is)] stopping at $((i-1)) workers (${avail} MiB free)"
      break
    fi
  fi
  "$ROOT/venv/bin/python" "$ROOT/src/qworker.py" "$PHASE" "$i" &
  pids+=($!)
  echo "[$(date -Is)] launched worker $i (pid ${pids[-1]})"
  sleep 90   # let it allocate before deciding on the next one
done

for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date -Is)] phase ${PHASE} complete"
D=$(ls "$ROOT/queue/done/$PHASE" 2>/dev/null | wc -l)
F=$(ls "$ROOT/queue/failed/$PHASE" 2>/dev/null | wc -l)
echo "[$(date -Is)] done=${D} failed=${F}"
