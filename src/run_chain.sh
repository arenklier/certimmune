#!/bin/bash
# Run a sequence of queue phases on one worker slot, autonomously.
#
#   run_chain.sh <worker_id> <phase> [phase ...]
#
# Before starting a phase, waits until no other worker is already running it,
# so a chain can be launched while an earlier worker is still finishing that
# same phase. A phase whose jobs are all done exits immediately, which makes it
# safe to list a phase that is already complete.
#
# A worker exits both when a phase is finished AND when it is stopped by hand,
# and those look identical from here. Advancing on the second case is how a
# chain once jumped from a halted p1 straight into p3 and started certifying
# MetaCloak images that had not been generated yet. So after each phase the
# chain checks that every job carries a done marker; if some are merely
# unclaimed it retries the phase, and if some genuinely failed it stops rather
# than carrying a broken prerequisite into the next phase.

set -u
ROOT=/mnt/data/certimmune
WID=${1:?worker id required}
shift
[ $# -gt 0 ] || { echo "no phases given"; exit 1; }

cd "$ROOT/src" || exit 1

for PH in "$@"; do
  T=$(grep -c . "$ROOT/queue/jobs_${PH}.txt")
  attempt=0
  while :; do
    while ps -eo cmd | grep -q "[q]worker\.py ${PH} "; do sleep 60; done
    attempt=$((attempt + 1))
    echo "[$(date -Is)] chain ${WID}: phase ${PH} (attempt ${attempt})"
    "$ROOT/venv/bin/python" qworker.py "$PH" "$WID"

    D=$(ls "$ROOT/queue/done/$PH" 2>/dev/null | wc -l)
    F=$(ls "$ROOT/queue/failed/$PH" 2>/dev/null | wc -l)
    echo "[$(date -Is)] chain ${WID}: ${PH} done=${D}/${T} failed=${F}"

    [ "$D" -ge "$T" ] && break
    if [ "$F" -gt 0 ]; then
      echo "[$(date -Is)] chain ${WID}: ${PH} has ${F} failed job(s); stopping the chain"
      echo "    failed: $(ls "$ROOT/queue/failed/$PH" | tr '\n' ' ')"
      exit 1
    fi
    if [ "$attempt" -ge 3 ]; then
      echo "[$(date -Is)] chain ${WID}: ${PH} still incomplete after ${attempt} attempts; stopping"
      exit 1
    fi
    echo "[$(date -Is)] chain ${WID}: ${PH} incomplete and nothing failed -- retrying"
    sleep 30
  done
done

echo "[$(date -Is)] chain ${WID}: all phases complete"
