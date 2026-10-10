#!/bin/bash
# Is fast-EOT's one failure at the matched budget an over-optimization effect or
# an unlucky draw?
#
# At 50 steps identity_00 certifies at the floor; at 80 it records 2/30 at
# t*=100 and lands at 0.229. Those two runs differ in BOTH the step count and
# the theta sequence they drew, so the question cannot be answered from them.
# Two sweeps separate the variables:
#
#   budget  -- the same seed across 20..160 steps. If the certificate degrades
#              monotonically past some budget, fast-EOT over-fits its one-step
#              approximation of the purifier and the matched-budget claim needs
#              that stated.
#   seed    -- the same 80-step budget across five seeds. If most seeds reach
#              the floor, the 80-step failure was a draw.
#
# Run on three identities, not one: identity_00 is where the effect appeared,
# identity_17 has the weakest unprotected control of the cohort, identity_11 is
# the one the eps8 cohort found hardest.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample

while pgrep -f 'fasteot80|cohort60' > /dev/null; do sleep 60; done

echo "########## budget sweep (seed 0) ##########"
for id in 00 17 11; do
  for st in 20 40 60 80 100 160; do
    O=$R/certificate_fastbudget_${id}_s${st}.json
    [ -f "$O" ] && continue
    $PY delta_optimize_fast.py --identity $D/identity_${id}.png --steps $st --k-theta 3 \
        --tag _fastbudget_${id}_s${st} > /dev/null 2>&1 || { echo "FAILED opt $id $st"; continue; }
    $PY certify.py --clean $D/identity_${id}.png \
        --immunized $R/delta_immunized_fastbudget_${id}_s${st}.png --out "$O" > /dev/null 2>&1 \
        || echo "FAILED cert $id $st"
    echo "  id $id steps $st"
  done
done

echo "########## seed sweep at 80 steps ##########"
for id in 00 17 11; do
  for sd in 1 2 3 4; do
    O=$R/certificate_fastseed_${id}_d${sd}.json
    [ -f "$O" ] && continue
    $PY delta_optimize_fast.py --identity $D/identity_${id}.png --steps 80 --k-theta 3 \
        --seed $sd --tag _fastseed_${id}_d${sd} > /dev/null 2>&1 || { echo "FAILED opt $id $sd"; continue; }
    $PY certify.py --clean $D/identity_${id}.png \
        --immunized $R/delta_immunized_fastseed_${id}_d${sd}.png --out "$O" > /dev/null 2>&1 \
        || echo "FAILED cert $id $sd"
    echo "  id $id seed $sd"
  done
done
echo "BUDGET_CURVE_DONE"
