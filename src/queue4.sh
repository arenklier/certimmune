#!/bin/bash
# Close the last asymmetry in the cross-architecture comparison.
#
# fast-EOT carries 20-identity cohorts on GrIDPure and on the PixArt DiT;
# PhotoGuard carries one identity on each. Putting "fast-EOT reaches the floor
# on 20 of 20 under GrIDPure" next to a single-identity baseline number repeats,
# on the other two architectures, exactly the asymmetry that was just removed
# from the DiffPure comparison.
#
# No optimization is needed: the twenty PhotoGuard perturbations are the ones
# already on disk from the official 1000-step recipe. This is certification only,
# at the same n and the same grids the fast-EOT cohorts used.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample
IDS="00 01 02 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"

pgp () { if [ "$1" = "00" ]; then echo "photoguard_baseline_immunized.png"; else echo "photoguard_baseline_$1_immunized.png"; fi; }

echo "########## A. PhotoGuard under GrIDPure (n=30) ##########"
for id in $IDS; do
  O=$R/certificate_gridpure_photoguard_${id}.json
  [ -f "$O" ] && continue
  P=$R/$(pgp $id)
  [ -f "$P" ] || { echo "  missing perturbation $id"; continue; }
  $PY certify_gridpure.py --clean $D/identity_${id}.png --immunized "$P" --out "$O" \
      > /dev/null 2>&1 && echo "  gridpure $id" || echo "FAILED gridpure $id"
done

echo "########## B. PhotoGuard under the PixArt DiT (n=30) ##########"
for id in $IDS; do
  [ -f $R/certificate_pixart_photoguard_${id}.json ] && continue
  P=$R/$(pgp $id)
  [ -f "$P" ] || { echo "  missing perturbation $id"; continue; }
  $PY certify_pixart.py --n-seeds 30 --clean $D/identity_${id}.png \
      --arms photoguard_${id}=$(pgp $id) > /dev/null 2>&1 \
      && echo "  pixart $id" || echo "FAILED pixart $id"
done
echo QUEUE4_DONE
