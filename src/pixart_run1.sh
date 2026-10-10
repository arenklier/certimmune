#!/bin/bash
# Third-architecture pass, job 1: the cross-method table on identity_00.
# One uniform n=30 for every arm (the DiffPure/GrIDPure columns were run at a
# mix of n=20 and n=30 over many months; the new column is run in one sitting,
# so there is no reason to inherit that unevenness).
set -e
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python

$PY certify_pixart.py --n-seeds 30 --with-unprotected \
    --arms ours=delta_immunized_eot.png

$PY certify_pixart.py --n-seeds 30 \
    --arms photoguard=photoguard_baseline_immunized.png \
           glaze=glaze_baseline_immunized.png \
           antidreambooth=antidreambooth_baseline_immunized.png \
           antipure=antipure_baseline_immunized.png \
           ours_wcmatch17=delta_computematched17_immunized.png
echo "JOB1 DONE"
