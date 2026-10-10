#!/bin/bash
# Re-run the three protected full-chain conditions with the fixed clean reference.
# Unprotected runs are unaffected (their identity-dir IS the clean set).
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results

for ID in 3 17 38; do
  echo "=========== identity_${ID} protected (FIXED reference) ==========="
  $PY lora_calibration.py \
      --identity-dir  $D/identity_${ID}_protected \
      --reference-dir $D/identity_${ID} \
      --conditions diffpure_t200 \
      --n-generate 20 \
      --out $R/lora_calib_identity_${ID}_protected_fixedref.json
  echo
done
echo "ALL DONE"
