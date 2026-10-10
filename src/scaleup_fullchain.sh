#!/bin/bash
# Extend the full-chain (real-attack) validation from 3 to 9 identities.
# Pool = MetaCloak sets with >=3/4 images face-detectable (relaxed screen,
# results/metacloak_relaxed_screen.json). The old >=4/4 screen admitted only 3.
# Every run uses the FIXED clean reference (--reference-dir) for the protected
# condition, so generations are never scored against the protected images.
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results

for ID in 2 8 9 15 19 37; do
  echo "############ identity_${ID} ############"

  if [ ! -d "$D/identity_${ID}_protected" ]; then
    echo "--- generating EOT protection for identity_${ID} ---"
    $PY protect_identity_set.py \
        --identity-dir $D/identity_${ID} \
        --out-dir      $D/identity_${ID}_protected || { echo "PROTECT FAILED $ID"; continue; }
  fi

  echo "--- identity_${ID} UNPROTECTED ---"
  $PY lora_calibration.py \
      --identity-dir $D/identity_${ID} \
      --conditions diffpure_t200 --n-generate 20 \
      --out $R/lora_calib_identity_${ID}_unprotected_fixedref.json || echo "UNPROT FAILED $ID"

  echo "--- identity_${ID} PROTECTED (fixed clean reference) ---"
  $PY lora_calibration.py \
      --identity-dir  $D/identity_${ID}_protected \
      --reference-dir $D/identity_${ID} \
      --conditions diffpure_t200 --n-generate 20 \
      --out $R/lora_calib_identity_${ID}_protected_fixedref.json || echo "PROT FAILED $ID"
  echo
done
echo "SCALEUP DONE"
