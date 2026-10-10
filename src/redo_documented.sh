#!/bin/bash
# Re-run the whole nine-identity full chain with the protocol the paper
# documents (rank 8, 1200 steps, lr 2e-4).
#
# Why: today's scale-up was run at the script's DEFAULTS (rank 4, 300 steps,
# lr 1e-4). A controlled probe on identity_17 unprotected gives
#   defaults          : 0.054 / 0.055 / 0.052   (run-to-run sd ~0.001)
#   documented setting: 0.189
# so the defaults produce a ~3.5x weaker attack, and the "attack establishes on
# only four of nine" result is very likely an artifact of that, not a property
# of the pipeline. Variance is small, so one run per cell suffices once the
# settings are right.
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results
ARGS="--rank 4 --steps 1200 --lr 2e-4 --conditions diffpure_t200 --n-generate 20"

for ID in 3 17 38 9 8 37 2 15 19; do
  echo "############ identity_${ID} (documented protocol) ############"
  echo "--- UNPROTECTED ---"
  $PY lora_calibration.py --identity-dir $D/identity_${ID} $ARGS \
      --out $R/lora_calib_identity_${ID}_unprotected_r4.json || echo "UNPROT FAILED $ID"
  echo "--- PROTECTED (fixed clean reference) ---"
  $PY lora_calibration.py --identity-dir $D/identity_${ID}_protected \
      --reference-dir $D/identity_${ID} $ARGS \
      --out $R/lora_calib_identity_${ID}_protected_r4.json || echo "PROT FAILED $ID"
  echo
done
echo "DOCUMENTED REDO DONE"
