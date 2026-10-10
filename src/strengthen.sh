#!/bin/bash
# Two pre-submission strengthening runs (rebuttal_prep.md A1 + A2).
#
# JOB 1 -- raise the Monte-Carlo sample size on identity_00 from n=30 to n=100.
#   The reported floor alpha=0.095 is 1-0.05^(1/30), i.e. a SAMPLE-SIZE CEILING,
#   not a property of the perturbation. At n=100 the same zero-failure result
#   certifies at 0.030.
#
# JOB 2 -- raise the calibration evidence from 18 rows (9 identities) to 34
#   (17 identities), by admitting the MetaCloak sets with >=2/4 images face-
#   detectable. The paper's own argument for relaxing 4/4 -> >=3/4 (the reference
#   template skips undetectable images; LoRA training needs no detection at all)
#   applies unchanged at >=2/4. The calibration term falls 0.153 -> 0.084.
#
# CONSISTENCY (load-bearing): the 8 new identities must be protected with
# protect_identity_set.py at ITS DEFAULTS -- exactly how the existing 9 were made
# by scaleup_fullchain.sh -- and calibrated with the DOCUMENTED LoRA protocol
# (--rank 4 --steps 1200 --lr 2e-4), exactly how redo_documented.sh produced the
# _r4 files that Table V actually reports. Mixing either would repeat the
# defaults-vs-documented error this project already made once.
set -u
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results
CAL_ARGS="--rank 4 --steps 1200 --lr 2e-4 --conditions diffpure_t200 --n-generate 20"

echo "########## JOB 1: n=100 certification on identity_00 ##########"
date
$PY certify.py --n-seeds 100 --bonferroni --out $R/certificate_n100.json \
  && echo "JOB1 OK" || echo "JOB1 FAILED"
date

echo
echo "########## JOB 2: calibration 18 -> 34 rows ##########"
for ID in 4 13 14 18 21 23 28 31; do
  echo "===== identity_${ID} ====="
  date
  if [ ! -d "$D/identity_${ID}_protected" ]; then
    echo "--- EOT protection (script defaults, as for the existing nine) ---"
    $PY protect_identity_set.py \
        --identity-dir $D/identity_${ID} \
        --out-dir      $D/identity_${ID}_protected || { echo "PROTECT FAILED $ID"; continue; }
  fi

  echo "--- unprotected ---"
  $PY lora_calibration.py --identity-dir $D/identity_${ID} $CAL_ARGS \
      --out $R/lora_calib_identity_${ID}_unprotected_r4.json || echo "UNPROT FAILED $ID"

  echo "--- protected (fixed clean reference) ---"
  $PY lora_calibration.py --identity-dir $D/identity_${ID}_protected \
      --reference-dir $D/identity_${ID} $CAL_ARGS \
      --out $R/lora_calib_identity_${ID}_protected_r4.json || echo "PROT FAILED $ID"
  echo
done
date
echo "STRENGTHEN DONE"
