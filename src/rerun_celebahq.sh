#!/bin/bash
# Re-run the single-image CelebA-HQ full-chain attempt with the FIXED clean
# reference. The original run (2 Sep) predates the --reference-dir fix (6 Sep),
# so its "erratic" outcome is confounded by the very bug we since removed; the
# paper cannot cite it as evidence until it is re-measured.
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/celebahq_fullchain
R=../results

for ID in 01 05 07 10; do
  echo "############ celebahq identity_${ID} ############"
  echo "--- UNPROTECTED ---"
  $PY lora_calibration.py \
      --identity-dir $D/identity_${ID}_unprotected \
      --conditions diffpure_t200 --n-generate 20 \
      --out $R/lora_calib_celebahq_${ID}_unprotected_fixedref.json || echo "UNPROT FAILED $ID"

  echo "--- PROTECTED (fixed clean reference) ---"
  $PY lora_calibration.py \
      --identity-dir  $D/identity_${ID}_protected \
      --reference-dir $D/identity_${ID}_unprotected \
      --conditions diffpure_t200 --n-generate 20 \
      --out $R/lora_calib_celebahq_${ID}_protected_fixedref.json || echo "PROT FAILED $ID"
  echo
done
echo "CELEBAHQ RERUN DONE"
