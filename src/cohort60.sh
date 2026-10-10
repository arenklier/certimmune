#!/bin/bash
# Take the certification cohort from 20 identities to ~59.
#
# Certification is the cheap tier -- the whole reason the paper has a two-tier
# design -- so the scale objection is answerable here for a couple of hours of
# GPU rather than the weeks the full chain would need. The protocol is exactly
# the one the existing cohort used (run_certification_experiment.py, eps=8/255,
# n=20/point, K=7), so these are an extension of the same experiment and not a
# new one.
#
# No identity is excluded up front. Where the unprotected control turns out to
# be degenerate, that is reported per identity the way the supplement already
# reports it for the third architecture, rather than screened out by hand.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample

while pgrep -f 'fasteot80' > /dev/null; do sleep 60; done

for id in 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
          40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59; do
  O=$R/certificate_identity_${id}_eps8.json
  [ -f "$O" ] && continue
  [ -f $D/identity_${id}.png ] || { echo "  missing image $id"; continue; }
  $PY run_certification_experiment.py --identity $D/identity_${id}.png \
      --tag identity_${id}_eps8 --n-seeds 20 > /dev/null 2>&1 || echo "FAILED $id"
  echo "  certified identity_${id}"
done
echo "COHORT60_DONE"
