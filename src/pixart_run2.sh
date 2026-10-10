#!/bin/bash
# Third-architecture pass, job 2: the 19-identity cohort, at the SAME n=20 the
# DiffPure cohort used, so the two columns are directly comparable rather than
# comparable-after-an-argument. identity_03 stays excluded for the reason
# Sec. VII-A already gives.
set -e
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
../venv/bin/python certify_pixart.py --n-seeds 20 --with-unprotected \
    --cohort 01,02,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20
echo "JOB2 DONE"
