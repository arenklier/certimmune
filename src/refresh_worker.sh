#!/bin/bash
# Wait until a worker finishes the job it is currently running, then restart it
# so it picks up the regenerated job list.
#
#   refresh_worker.sh <phase> <worker_id> <job_id_to_wait_for>
#
# qworker reads the job file once at startup, so a worker started before the
# file was regenerated keeps running the old commands from memory. Restarting
# it between jobs is the only way to pick up the new ones without discarding
# work already in flight.

set -u
ROOT=/mnt/data/certimmune
PHASE=${1:?phase}
WID=${2:?worker id}
WAIT_JOB=${3:?job id}
Q=$ROOT/queue

echo "[$(date -Is)] waiting for ${WAIT_JOB} to finish before refreshing worker ${WID}"
until [ -e "$Q/done/$PHASE/$WAIT_JOB" ] || [ -e "$Q/failed/$PHASE/$WAIT_JOB" ]; do
  sleep 60
  # if the worker died on its own there is nothing left to wait for
  ps -eo cmd | grep -q "[q]worker.py ${PHASE} ${WID}" || { echo "[$(date -Is)] worker gone"; break; }
done
echo "[$(date -Is)] ${WAIT_JOB} finished; restarting worker ${WID}"

bash "$ROOT/src/stop_worker.sh" "$PHASE" "$WID" || true
sleep 5
cd "$ROOT/src" || exit 1
setsid nohup "$ROOT/venv/bin/python" qworker.py "$PHASE" "$WID" \
  > "$Q/${PHASE}_worker${WID}.log" 2>&1 < /dev/null &
echo "[$(date -Is)] worker ${WID} restarted with the refreshed job list"
