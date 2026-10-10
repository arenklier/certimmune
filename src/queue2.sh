#!/bin/bash
# Restart what the CRLF-mangled repro_and_seeds.sh never ran, and find out why
# 14 of the 39 cohort identities failed.
#
# Order is by what the paper is blocked on. The reproducibility test and the
# multi-seed cohort decide how the head-to-head headline is worded, so they go
# first once the diagnostic is in hand; the cohort expansion is a scale answer
# and can follow.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
R=../results

echo "########## 0. why does identity_41 fail? full stderr, no redirect ##########"
../venv/bin/python run_certification_experiment.py \
    --identity ../data/celebahq_sample/identity_41.png --tag diag_41 --n-seeds 20
echo "diag_41 exit: $?"

echo "########## 1. repro + seeds ##########"
bash repro_and_seeds.sh 2>&1 | sed "s/^/[rs] /"
echo QUEUE2_DONE
