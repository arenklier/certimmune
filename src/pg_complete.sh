#!/bin/bash
# Complete the PhotoGuard side of the matched-budget head-to-head at a uniform n=30.
#
# PhotoGuard was certified on identity_00 at n=20 (floor 0.139) and on 04 and
# 12-20 at n=30 (floor 0.095); on 01, 02 and 05-11 its perturbations exist but
# were only ever scored on the raw proxy, never certified. Comparing across
# those is comparing against two different floors, so every cell is redone at
# n=30 here. The perturbations themselves are not regenerated -- these are the
# same 1000-step official-recipe images already on disk.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample

while pgrep -f 'cohort60|budget_curve' > /dev/null; do sleep 60; done

for id in 00 01 02 05 06 07 08 09 10 11; do
  O=$R/certificate_photoguard_${id}_n30.json
  [ -f "$O" ] && continue
  P=$R/photoguard_baseline_${id}_immunized.png
  [ "$id" = "00" ] && P=$R/photoguard_baseline_immunized.png
  if [ ! -f "$P" ]; then
    $PY photoguard_baseline.py --identity $D/identity_${id}.png --tag _${id} > /dev/null 2>&1 \
        || { echo "FAILED opt $id"; continue; }
    P=$R/photoguard_baseline_${id}_immunized.png
  fi
  $PY certify.py --clean $D/identity_${id}.png --immunized "$P" --n-seeds 30 --out "$O" \
      > /dev/null 2>&1 || echo "FAILED cert $id"
  echo "  photoguard n30 $id"
done
echo "PG_COMPLETE_DONE"
