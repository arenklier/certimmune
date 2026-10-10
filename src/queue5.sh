#!/bin/bash
# A second optimization seed for the cross-architecture comparison.
#
# GrIDPure and the DiT now carry twenty identities for both methods, but at one
# optimization seed each. The DiffPure campaign measured what that costs: over
# three seeds fast-EOT returns an identical alpha on 18 of 20 identities and
# PhotoGuard on 10 of 20, so PhotoGuard's single draw is the less settled of the
# two and the one a referee would press on.
#
# Both methods' seed-1 perturbations already exist, so this is certification
# only. DiT first because it is roughly four times faster per identity, so the
# cheaper half of the answer lands first.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample
IDS="00 01 02 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"

echo '########## A. DiT, seed 1, both methods ##########'
for id in $IDS; do
  for m in fasteot80 photoguard; do
    if [ "$m" = fasteot80 ]; then P=delta_immunized_fasteot80_${id}_d1.png; else P=photoguard_baseline_${id}_d1_immunized.png; fi
    [ -f $R/certificate_pixart_${m}_${id}_d1.json ] && continue
    [ -f $R/$P ] || { echo "  no delta $m $id"; continue; }
    $PY certify_pixart.py --n-seeds 30 --clean $D/identity_${id}.png         --arms ${m}_${id}_d1=$P > /dev/null 2>&1         && echo "  dit $m $id" || echo "FAILED dit $m $id"
  done
done

echo '########## B. GrIDPure, seed 1, both methods ##########'
for id in $IDS; do
  for m in fasteot80 photoguard; do
    if [ "$m" = fasteot80 ]; then P=delta_immunized_fasteot80_${id}_d1.png; else P=photoguard_baseline_${id}_d1_immunized.png; fi
    O=$R/certificate_gridpure_${m}_${id}_d1.json
    [ -f "$O" ] && continue
    [ -f $R/$P ] || { echo "  no delta $m $id"; continue; }
    $PY certify_gridpure.py --clean $D/identity_${id}.png --immunized $R/$P --out "$O"         > /dev/null 2>&1 && echo "  gridpure $m $id" || echo "FAILED gridpure $m $id"
  done
done
echo QUEUE5_DONE
