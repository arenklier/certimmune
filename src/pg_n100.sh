#!/bin/bash
# PhotoGuard on identity_00 at n=100, to match fast-EOT's n=100 run there.
# identity_00 is the single identity where fast-EOT is behind at n=30, and
# fast-EOT's 0.229 there drops to 0.095 at n=100 -- comparing that against
# PhotoGuard's n=30 number would be comparing two different sample sizes.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
R=../results
while pgrep -f 'rerun_all|budget_curve|cohort60|pg_complete' > /dev/null; do sleep 60; done
O=$R/certificate_photoguard_00_n100.json
[ -f "$O" ] || ../venv/bin/python certify.py \
    --clean ../data/celebahq_sample/identity_00.png \
    --immunized $R/photoguard_baseline_immunized.png \
    --n-seeds 100 --bonferroni --out "$O" > /dev/null 2>&1
echo PG_N100_DONE
