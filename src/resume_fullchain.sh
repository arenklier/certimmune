#!/bin/bash
# Wait-and-resume driver for the three identities that hit CUDA OOM when other
# users' jobs occupied the GPU. Waits for enough free VRAM before each stage and
# retries an identity up to 3 times before giving up on it.
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results
NEED=26000          # MiB free required before starting a stage
POLL=120            # seconds between polls
MAXWAIT=180         # max polls (~6 h)

wait_for_gpu () {
  local i=0
  while [ $i -lt $MAXWAIT ]; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge "$NEED" ]; then
      echo "[gpu] ${FREE} MiB free (>= ${NEED}) -- proceeding"
      return 0
    fi
    echo "[gpu] ${FREE} MiB free (< ${NEED}) -- waiting ${POLL}s (poll $i/$MAXWAIT)"
    sleep $POLL
    i=$((i+1))
  done
  echo "[gpu] timed out waiting for free VRAM"
  return 1
}

for ID in 15 19 37; do
  echo "############ identity_${ID} ############"
  for ATTEMPT in 1 2 3; do
    echo "--- attempt ${ATTEMPT} for identity_${ID} ---"
    wait_for_gpu || break

    if [ ! -d "$D/identity_${ID}_protected" ]; then
      $PY protect_identity_set.py \
          --identity-dir $D/identity_${ID} \
          --out-dir      $D/identity_${ID}_protected \
        || { echo "PROTECT FAILED $ID (attempt $ATTEMPT)"; rm -rf $D/identity_${ID}_protected; sleep 300; continue; }
    fi

    wait_for_gpu || break
    $PY lora_calibration.py \
        --identity-dir $D/identity_${ID} \
        --conditions diffpure_t200 --n-generate 20 \
        --out $R/lora_calib_identity_${ID}_unprotected_fixedref.json \
      || { echo "UNPROT FAILED $ID (attempt $ATTEMPT)"; sleep 300; continue; }

    wait_for_gpu || break
    $PY lora_calibration.py \
        --identity-dir  $D/identity_${ID}_protected \
        --reference-dir $D/identity_${ID} \
        --conditions diffpure_t200 --n-generate 20 \
        --out $R/lora_calib_identity_${ID}_protected_fixedref.json \
      || { echo "PROT FAILED $ID (attempt $ATTEMPT)"; sleep 300; continue; }

    echo "identity_${ID} COMPLETE"
    break
  done
  echo
done
echo "RESUME DONE"
