#!/bin/bash
# Close the two genuine staleness flags found by the audit:
#  (1) identity_3/17/38 UNPROTECTED full-chain values predate the
#      --reference-dir patch. The patch should not affect them (for an
#      unprotected run the reference IS the identity dir), but that is a
#      claim, and the point of the audit is not to trust such claims.
#  (2) transfer_all_baselines.json (PhotoGuard / Anti-DreamBooth GrIDPure
#      transfer, Table IV) predates a 15 Aug edit to its producing script.
cd /mnt/data/certimmune/src
PY=/mnt/data/certimmune/venv/bin/python
D=../data/metacloak_identities
R=../results

echo "########## (1) unprotected full-chain re-runs ##########"
for ID in 3 17 38; do
  echo "--- identity_${ID} UNPROTECTED (current script) ---"
  $PY lora_calibration.py \
      --identity-dir $D/identity_${ID} \
      --conditions diffpure_t200 --n-generate 20 \
      --out $R/lora_calib_identity_${ID}_unprotected_recheck.json || echo "FAILED $ID"
done

echo
echo "########## (2) GrIDPure transfer re-run, current script ##########"
$PY transfer_test_all_baselines.py \
    --methods photoguard antidreambooth \
    --out $R/transfer_all_baselines_recheck.json || echo "TRANSFER FAILED"

echo "STALENESS RERUNS DONE"
