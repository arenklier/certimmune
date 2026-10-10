#!/bin/bash
# Two questions the staleness audit forced open:
#  (1) How large is run-to-run variance of the full-chain measurement?
#      Re-running identity_17 unprotected gave real_mean 0.066 where the stored
#      value is 0.120. If that spread is typical, every per-identity number in
#      Table V -- and the 4-of-9 "attack established" split -- is a single draw
#      from a wide distribution.
#  (2) Do the settings the paper documents (rank 8, 1200 steps, lr 2e-4)
#      reproduce the stored values better than the code defaults
#      (rank 4, 300 steps, lr 1e-4)?
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results

echo "########## (1) variance at CODE DEFAULTS: identity_17 unprotected x3 ##########"
for REP in 1 2 3; do
  echo "--- repeat ${REP} (rank 4, 300 steps, lr 1e-4) ---"
  $PY lora_calibration.py \
      --identity-dir $D/identity_17 \
      --conditions diffpure_t200 --n-generate 20 \
      --out $R/varprobe_id17_default_rep${REP}.json || echo "FAILED rep${REP}"
done

echo
echo "########## (2) DOCUMENTED settings: rank 8, 1200 steps, lr 2e-4 ##########"
$PY lora_calibration.py \
    --identity-dir $D/identity_17 \
    --conditions diffpure_t200 --n-generate 20 \
    --rank 8 --steps 1200 --lr 2e-4 \
    --out $R/varprobe_id17_documented.json || echo "FAILED documented"

echo "VARIANCE PROBE DONE"
