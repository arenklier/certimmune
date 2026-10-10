#!/bin/bash
# The DiT phase of the fast-EOT campaign, as its own script.
#
# fasteot80.sh's own phase 1d calls certify_pixart.py in --cohort mode, which
# names its output certificate_pixart_identity_XX.json -- files that already
# exist from the eps8 cohort, so it skips every one of them and does nothing.
# Rather than edit a script while bash is reading it, the phase is redone here
# in --arms mode, which gives these runs their own filenames.
set -u
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
PY=../venv/bin/python
R=../results
D=../data/celebahq_sample
IDS="00 01 02 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"

while pgrep -f fasteot80.sh > /dev/null; do sleep 60; done

echo "########## DiT (n=30), fast-EOT 80-step ##########"
for id in $IDS; do
  [ -f $R/certificate_pixart_fasteot80_${id}.json ] && continue
  [ -f $R/delta_immunized_fasteot80_${id}.png ] || { echo "  no delta for $id"; continue; }
  $PY certify_pixart.py --n-seeds 30 --clean $D/identity_${id}.png \
      --arms fasteot80_${id}=delta_immunized_fasteot80_${id}.png > /dev/null 2>&1 \
      || echo "FAILED pixart $id"
  echo "  pixart $id"
done
echo "FASTEOT80_DIT_DONE"
