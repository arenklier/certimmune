#!/bin/bash
# Is the documented (strong) LoRA setting as reproducible as the weak one?
#
# The project's own note justifies one full-chain run per cell with a measured
# run-to-run sd of ~0.001 -- but that was measured at the script DEFAULTS
# (rank 4, 300 steps, lr 1e-4), where the attack is ~3.5x weaker. The paper
# reports the documented setting (1200 steps, lr 2e-4). identity_19 unprotected
# at t*=200 has now come out at 0.2064 (isolated) and 0.1234 (inside a
# four-condition invocation), which that sd cannot explain. Three isolated,
# single-condition repeats of the identical command separate run-to-run variance
# from an order effect.
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
D=../data/metacloak_identities
R=../results
ARGS="--rank 4 --steps 1200 --lr 2e-4 --conditions diffpure_t200 --n-generate 20"
while pgrep -f 'pixart_run5.sh' > /dev/null; do sleep 30; done
for K in 1 2 3; do
  $PY lora_calibration.py --identity-dir $D/identity_19 $ARGS \
      --out $R/lora_calib_identity_19_unprotected_rep${K}.json
done
echo "REPEATS DONE"
