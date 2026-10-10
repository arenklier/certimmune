#!/bin/bash
# Wall-clock is now load-bearing for a headline claim, so it gets the same
# treatment every other load-bearing number in this paper gets: repeated, with
# the spread reported. Three exclusive-GPU repeats.
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
for K in 1 2 3; do
  echo "########## repeat $K ##########"
  ../venv/bin/python timing_exclusive.py 2>&1 | grep -E 's/step|x$|budget|matched steps'
  cp ../results/timing_exclusive.json ../results/timing_exclusive_rep${K}.json
done
echo TIMING_REPS_DONE
