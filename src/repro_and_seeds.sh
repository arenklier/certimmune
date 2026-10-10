#!/bin/bash
# Two questions the non-reproducible perturbation raises.
#
# A. How far does it reach? fast-EOT at 80 steps, same identity, same seed,
#    produced two perturbations differing on 90.6% of pixels. Both optimizers
#    use sign-based PGD through an fp16 backward pass, so the original
#    EOT-over-Theta objective the paper headlines is very likely the same. Two
#    fresh runs of each, identical arguments, then compare the images.
#
# B. What does it cost the headline? "19 of 20 at the floor" is one draw. The
#    cohort is re-run at two further optimization seeds so the claim can carry
#    the spread instead of pretending there is none.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample
IDS="00 01 02 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"

while pgrep -f 'rerun_all|cohort60|budget_curve|pg_n100' > /dev/null; do sleep 60; done

echo "########## A. is the ORIGINAL EOT delta reproducible? ##########"
# delta_optimize.py writes to a fixed filename and would overwrite
# delta_immunized_eot.png, the paper's own artifact, so the EOT side goes
# through run_certification_experiment.py, which takes a --tag, runs the same
# EOT-over-Theta optimizer at the cohort's settings, and certifies as well.
for rep in r1 r2; do
  [ -f $R/immunized_eotrepro_${rep}.png ] && continue
  $PY run_certification_experiment.py --identity $D/identity_00.png       --tag eotrepro_${rep} --n-seeds 30 > /dev/null 2>&1 || echo "FAILED eot $rep"
  echo "  eot $rep"
done
for rep in r1 r2; do
  [ -f $R/delta_immunized_fastrepro_${rep}.png ] && continue
  $PY delta_optimize_fast.py --identity $D/identity_00.png --steps 80 --k-theta 3       --tag _fastrepro_${rep} > /dev/null 2>&1 || echo "FAILED fast $rep"
  echo "  fast $rep"
done

echo "########## B. the cohort at two further optimization seeds ##########"
for sd in 1 2; do
for id in $IDS; do
  P=$R/delta_immunized_fasteot80_${id}_d${sd}.png
  O=$R/certificate_fasteot80_${id}_d${sd}.json
  [ -f "$O" ] && continue
  [ -f "$P" ] || $PY delta_optimize_fast.py --identity $D/identity_${id}.png --steps 80 \
      --k-theta 3 --seed $sd --tag _fasteot80_${id}_d${sd} > /dev/null 2>&1 \
      || { echo "FAILED opt $id $sd"; continue; }
  $PY certify.py --clean $D/identity_${id}.png --immunized "$P" --out "$O" > /dev/null 2>&1 \
      || echo "FAILED cert $id $sd"
  echo "  seed $sd identity $id"
done
done
echo "REPRO_AND_SEEDS_DONE"
