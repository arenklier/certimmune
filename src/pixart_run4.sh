#!/bin/bash
# Plan-B hardening, chained behind the PixArt jobs.
#   D2: does monotonicity survive BETWEEN grid points (31-point sweep)?
#   D3: does the attack work at the bottom of Theta and below it? This is what
#       makes t_min defensible once monotonicity puts all the weight on it.
#       identity_19 has the strongest unprotected full-chain attack of the nine
#       (0.206), so it gives the cleanest contrast.
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
while pgrep -f 'pixart_run2.sh|pixart_run3.sh' > /dev/null; do sleep 60; done

$PY dense_sweep.py --n-seeds 10 --step 10
echo "D2 DONE"

D=../data/metacloak_identities
R=../results
ARGS="--rank 4 --steps 1200 --lr 2e-4 --n-generate 20 --conditions clean diffpure_t50 diffpure_t100 diffpure_t200"
$PY lora_calibration.py --identity-dir $D/identity_19 $ARGS \
    --out $R/lora_calib_identity_19_unprotected_lowt.json
$PY lora_calibration.py --identity-dir $D/identity_19_protected --reference-dir $D/identity_19 $ARGS \
    --out $R/lora_calib_identity_19_protected_lowt.json
echo "D3 DONE"
