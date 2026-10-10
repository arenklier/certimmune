#!/bin/bash
# fast-EOT at PhotoGuard's ACTUAL matched budget, at the paper's full cohort scale.
#
# Why 80 and not 50: the existing fast-EOT runs used 50 steps, matched against a
# per-step cost of ~1.1 s measured on a shared GPU. Three exclusive-GPU repeats
# now put it at 0.6231-0.6249 s/step against PhotoGuard's 0.0502-0.0505, so
# PhotoGuard's 1000-step budget is 50.3 +- 0.2 s and buys 80 fast-EOT steps, not
# 50. The old numbers were therefore run at 60% of the budget they claimed to
# match -- conservative, but not the comparison we want to headline.
#
# Every phase writes one file per identity and skips what exists, so this
# resumes after an interruption.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample
IDS="00 01 02 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"

echo "########## 1a. fast-EOT 80-step deltas ##########"
for id in $IDS; do
  [ -f $R/delta_immunized_fasteot80_${id}.png ] && continue
  $PY delta_optimize_fast.py --identity $D/identity_${id}.png --steps 80 --k-theta 3 \
      --tag _fasteot80_${id} > /dev/null 2>&1 || echo "FAILED opt $id"
  echo "  delta $id"
done

echo "########## 1b. certify DiffPure (n=30) ##########"
for id in $IDS; do
  O=$R/certificate_fasteot80_${id}.json
  [ -f "$O" ] && continue
  $PY certify.py --clean $D/identity_${id}.png \
      --immunized $R/delta_immunized_fasteot80_${id}.png --out "$O" > /dev/null 2>&1 \
      || echo "FAILED diffpure $id"
  echo "  diffpure $id"
done

echo "########## 1c. PhotoGuard identity-matched, the ten not yet run ##########"
for id in 04 12 13 14 15 16 17 18 19 20; do
  P=$R/photoguard_baseline_${id}_immunized.png
  if [ ! -f "$P" ]; then
    $PY photoguard_baseline.py --identity $D/identity_${id}.png --tag _${id} > /dev/null 2>&1 \
        || echo "FAILED pg-opt $id"
  fi
  O=$R/certificate_photoguard_${id}.json
  [ -f "$O" ] && continue
  [ -f "$P" ] && $PY certify.py --clean $D/identity_${id}.png --immunized "$P" --out "$O" \
      > /dev/null 2>&1 || echo "FAILED pg-cert $id"
  echo "  photoguard $id"
done

echo "########## 1d. certify the DiT (n=30) ##########"
$PY certify_pixart.py --n-seeds 30 --cohort 00,01,02,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20 \
    --cohort-protected "delta_immunized_fasteot80_%s.png" > $R/fasteot80_pixart.log 2>&1 \
    || echo "FAILED pixart"

echo "########## 1e. n=100 on identity_00 (DiffPure) ##########"
O=$R/certificate_fasteot80_00_n100.json
[ -f "$O" ] || $PY certify.py --clean $D/identity_00.png \
    --immunized $R/delta_immunized_fasteot80_00.png --n-seeds 100 --bonferroni --out "$O" \
    > /dev/null 2>&1 || echo "FAILED n100"

echo "########## 1f. certify GrIDPure ##########"
for id in $IDS; do
  O=$R/certificate_gridpure_fasteot80_${id}.json
  [ -f "$O" ] && continue
  $PY certify_gridpure.py --clean $D/identity_${id}.png \
      --immunized $R/delta_immunized_fasteot80_${id}.png --out "$O" > /dev/null 2>&1 \
      || echo "FAILED gridpure $id"
  echo "  gridpure $id"
done

echo "FASTEOT80_DONE"
