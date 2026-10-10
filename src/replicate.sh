#!/bin/bash
# Replicate every full-chain cell that feeds the calibration term.
#
# The term (Sec. VII-G: k=2 disagreements of n=34 rows -> 0.174) is 86% of the
# headline bound, and each of its 34 rows is a SINGLE run of a quantity we have
# now measured to wander with sd 0.039 -- one of the two disagreeing rows is
# decided by one generation out of twenty. Two further seeds per row turn each
# verdict into a majority of three.
#
# The runs already on disk are seed 0, so only seeds 1 and 2 are run here:
# 17 identities x 2 arms x 2 seeds = 68 runs at ~5.5 min. Each writes its own
# file and is skipped if that file exists, so this resumes after an interruption.
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
D=../data/metacloak_identities
R=../results
ARGS="--rank 4 --steps 1200 --lr 2e-4 --conditions diffpure_t200 --n-generate 20"

for SEED in 1 2; do
for ID in 3 17 38 9 8 37 2 15 19 4 13 14 18 21 23 28 31; do
  O=$R/lora_calib_identity_${ID}_unprotected_r4_s${SEED}.json
  [ -f "$O" ] || $PY lora_calibration.py --identity-dir $D/identity_${ID} $ARGS \
      --seed $SEED --out $O || echo "FAILED unprot $ID s$SEED"
  O=$R/lora_calib_identity_${ID}_protected_r4_s${SEED}.json
  [ -f "$O" ] || $PY lora_calibration.py --identity-dir $D/identity_${ID}_protected \
      --reference-dir $D/identity_${ID} $ARGS --seed $SEED --out $O || echo "FAILED prot $ID s$SEED"
  echo "### done identity_${ID} seed ${SEED} ($(date +%H:%M))"
done
done
echo "REPLICATION DONE"
