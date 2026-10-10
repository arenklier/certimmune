#!/bin/bash
# identity_19 PROTECTED was the one cell of the rank-4 redo that hit CUDA OOM
# (other users held ~39 GB at the time). Wait for room, then run just that cell.
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results
NEED=20000
POLL=120
MAX=180

i=0
while [ $i -lt $MAX ]; do
  if [ -f "$R/lora_calib_identity_19_protected_r4.json" ]; then echo "already present"; exit 0; fi
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -ge "$NEED" ]; then
    echo "[gpu] ${FREE} MiB free -- running identity_19 protected"
    $PY lora_calibration.py \
        --identity-dir  $D/identity_19_protected \
        --reference-dir $D/identity_19 \
        --rank 4 --steps 1200 --lr 2e-4 \
        --conditions diffpure_t200 --n-generate 20 \
        --out $R/lora_calib_identity_19_protected_r4.json \
      && { echo "identity_19 protected COMPLETE"; exit 0; }
    echo "[retry] run failed, waiting before the next attempt"
    sleep 300
  else
    echo "[gpu] ${FREE} MiB free (< ${NEED}) -- waiting (poll $i/$MAX)"
    sleep $POLL
  fi
  i=$((i+1))
done
echo "timed out waiting for GPU"
