#!/bin/bash
# Give the baseline the same treatment the method got.
#
# fast-EOT now carries three independent optimization seeds per identity, and
# the certificate turns out to be stable across them (18 of 20 identical).
# PhotoGuard is still a single draw, so the head-to-head table would be giving
# our own method three chances and the baseline one. That asymmetry is the
# first thing a referee would circle, and it is cheap to remove: PhotoGuard
# optimization is 50.3 s per identity.
#
# Part A repeats the reproducibility probe on PhotoGuard itself, so the
# non-determinism claim in Section 6 is stated about sign-based PGD in general
# rather than only about our two optimizers.
#
# Part C retries the 14 cohort identities that died in the 17 Sept run. That
# was not an identity-specific fault -- identity_41 certified cleanly when re-run
# by hand -- so they are simply attempted again.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample
IDS="00 01 02 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"

echo "########## A. is PhotoGuard reproducible either? ##########"
for rep in r1 r2; do
  [ -f $R/photoguard_baseline_pgrepro_${rep}_immunized.png ] && continue
  $PY photoguard_baseline.py --identity $D/identity_00.png --tag _pgrepro_${rep} \
      > /dev/null 2>&1 || echo "FAILED pgrepro $rep"
  echo "  pgrepro $rep"
done

echo "########## B. PhotoGuard at two further optimization seeds ##########"
for sd in 1 2; do
for id in $IDS; do
  P=$R/photoguard_baseline_${id}_d${sd}_immunized.png
  O=$R/certificate_photoguard_${id}_d${sd}.json
  [ -f "$O" ] && continue
  [ -f "$P" ] || $PY photoguard_baseline.py --identity $D/identity_${id}.png \
      --seed $sd --tag _${id}_d${sd} > /dev/null 2>&1 \
      || { echo "FAILED opt $id $sd"; continue; }
  $PY certify.py --clean $D/identity_${id}.png --immunized "$P" --n-seeds 30 --out "$O" \
      > /dev/null 2>&1 && echo "  seed $sd identity $id" || echo "FAILED cert $id $sd"
done
done

echo "########## C. retry the 14 cohort identities that died ##########"
for id in 41 42 43 45 47 51 52 53 54 55 56 57 58 59; do
  O=$R/certificate_identity_${id}_eps8.json
  [ -f "$O" ] && continue
  [ -f $D/identity_${id}.png ] || { echo "  missing image $id"; continue; }
  $PY run_certification_experiment.py --identity $D/identity_${id}.png \
      --tag identity_${id}_eps8 --n-seeds 20 > /dev/null 2>&1 \
      && echo "  certified identity_${id}" || echo "FAILED $id"
done
echo QUEUE3_DONE
