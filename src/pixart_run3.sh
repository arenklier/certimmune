#!/bin/bash
# Third-architecture pass, job 3: raise identity_00 to n=100 on PixArt, the same
# sample size Sec. VII-F uses for DiffPure, so the third column's floor is a
# measured floor rather than a small-n artifact. Chained behind job 2 so the two
# never share the GPU.
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
while pgrep -f pixart_run2.sh > /dev/null; do sleep 60; done
../venv/bin/python certify_pixart.py --n-seeds 100 --with-unprotected \
    --arms ours_n100=delta_immunized_eot.png
echo "JOB3 DONE"
